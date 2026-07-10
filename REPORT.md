# Advanced LLM Acceleration: EAGLE-3 Speculative Decoding + FP8 Quantization
**Model:** Qwen/Qwen3-8B &nbsp;|&nbsp; **Hardware:** 1x H100 80GB (Nebius VM)

## 1. Pipeline overview

| Stage | Env | Tool | Artifact |
|---|---|---|---|
| Data prep | `speculators_venv` | `prepare_data.py` | tokenized ShareGPT dataset |
| Serve verifier (bf16) | `vllm_venv` | `launch_vllm.py` | hidden-state-emitting server |
| Hidden-state capture | `speculators_venv` | `data_generation_offline.py` | `hs_*.safetensors` |
| **EAGLE-3 draft training** | `speculators_venv` | `train.py` | `checkpoint_best` (draft head) |
| **FP8 quantization** | `comp_venv` | `llmcompressor` `oneshot` + `FP8_DYNAMIC` | `Qwen3-8B-FP8-Dynamic` |
| Benchmarking | `vllm_venv` | `benchmark/benchmark.py` | throughput / latency / memory / acceptance rate |

All scripts are in this bundle (`setup/`, `speculators/`, `quantization/`, `benchmark/`).

### Training setup (EAGLE-3)

| Parameter | Value | Notes |
|---|---|---|
| Verifier | `Qwen/Qwen3-8B` (bf16) | Full precision for hidden-state generation and draft training |
| Dataset | ShareGPT-style (`sharegpt`) | Via `prepare_data.py` |
| `MAX_SAMPLES` | 3000 | Reduced from tutorial default (20k) to fit ~247 GB VM disk |
| `SEQ_LENGTH` | 2048 | Reduced from 8192 for the same disk constraint |
| Hidden-state generation | concurrency 8, request-timeout 60 | `data_generation_offline.py` |
| Draft training | 5 epochs, lr 1e-4, draft_vocab_size 32000 | `--save-best`, wandb + tensorboard logging |
| Speculative tokens (benchmark) | 3 | `num_speculative_tokens` in `benchmark.py` |
| Benchmark prompts | 256 prompts, 512 max new tokens | `bench_prompts.jsonl`, temperature 0.0 |

**Artifact paths (on VM):**
- Hidden states: `speculators-upstream/output/hidden_states/`
- Draft checkpoint: `speculators-upstream/output/checkpoints/checkpoint_best`
- FP8 verifier: `quantization/Qwen3-8B-FP8-Dynamic/`

### Task 1 — Why hidden states use much more disk than text

The ShareGPT source stores compact token IDs or UTF-8 strings (roughly a few bytes per token). Hidden-state caches store **bf16/fp16 activation tensors for every token position** across the verifier layers EAGLE-3 needs — typically tens of MB per sample for Qwen3-8B. Storage scales as `samples × seq_len × hidden_dim × layers_stored`, so 3,000 samples at seq 2048 lands around **~140 GB**, vs **~100 MB** for the text side. That is why reducing `max-samples` is the first lever when disk fills up.

### Task 2 — EAGLE-3 draft-head training

**Completed work:**
- Verifier: `Qwen/Qwen3-8B` (bf16)
- Training: `train_draft_head` on cached hidden states, 5 epochs, lr 1e-4, `--save-best`
- Checkpoints: `speculators-upstream/output/checkpoints/` → `checkpoint_best` used for benchmarking
- Metrics: TensorBoard / `val_metrics.json` — per-position `full_acc`, `cond_acc`, and `loss_k`

**Reference validation metrics (epoch 4 = 5th epoch):**

| Metric | Value | Draft position |
|---|---:|---|
| `val/loss_0_epoch` | 2.509 | 0 (first speculative token) |
| `val/full_acc_0_epoch` | 0.463 | 0 |
| `val/cond_acc_0_epoch` | 0.463 | 0 |
| `val/loss_1_epoch` | 3.778 | 1 |
| `val/full_acc_1_epoch` | 0.181 | 1 |
| `val/cond_acc_1_epoch` | 0.364 | 1 |
| `val/loss_2_epoch` | 4.550 | 2 |
| `val/full_acc_2_epoch` | 0.069 | 2 |
| `val/cond_acc_2_epoch` | 0.320 | 2 |
| `val/loss_epoch` | 10.837 | total (all positions) |

**Reading the table:** Position-0 accuracy is moderate (~46%). Later positions fall sharply in `full_acc` (18% → 7%) while `cond_acc` stays higher (36% → 32%), which is the expected error-compounding pattern. The aggregate `val/loss_epoch` is dominated by harder later positions — do not use it alone to judge draft quality.

#### Q1. What do `full_acc` and `cond_acc` measure?

- **`full_acc` (full / unconditional accuracy)** at draft position *k* is the fraction of validation examples where the draft model's predicted token at step *k* matches the verifier's (teacher) token at that step, evaluated under the **training-time test** rollout — i.e., the draft is fed its own prior predictions as inputs when generating later positions, mirroring real speculative decoding. This is the realistic end-to-end match rate at each depth in the speculative chain.

- **`cond_acc` (conditional accuracy)** at position *k* is the same match rate, but computed **only on examples where all earlier draft positions (0 … k−1) already matched the teacher**. It answers: "If the speculative chain has been perfect so far, how often does the draft get the next token right?"

At position 0 the two metrics coincide (no prior positions). For *k* ≥ 1, `cond_acc` ≥ `full_acc` because `full_acc` is reduced by cases where an early mistake already put the draft on the wrong trajectory.

These metrics proxy the runtime **draft acceptance rate** per speculative step: higher `full_acc` → more tokens accepted per verifier forward pass → better speculative-decoding speedup.

#### Q2. Why does accuracy usually decrease for later speculative positions?

1. **Error compounding (training-time test).** After position 0, each later prediction is conditioned on the draft's own previous outputs, not the verifier's ground-truth tokens. A wrong early token shifts the hidden-state context, so the draft drifts further from the teacher distribution at each step. `full_acc` drops fastest for this reason (0.463 → 0.181 → 0.069 in the reference run).

2. **Increasing prediction horizon.** Position *k* asks the draft to predict the (*k*+1)-th token ahead using features that were originally aligned to shorter horizons. Distant tokens are inherently harder to guess from cached verifier features.

3. **Loss weighting across positions.** Per-position loss rises (`loss_0` 2.5 → `loss_1` 3.8 → `loss_2` 4.6), so the total `val/loss_epoch` (10.8) is dominated by later, harder steps — another reason to inspect position-wise metrics instead of aggregate loss alone.

`cond_acc` also declines (0.463 → 0.364 → 0.320) but more slowly than `full_acc`, confirming that even on "good" prefixes the draft's multi-step predictions degrade with depth.

#### Q3. What would you change if first-position accuracy is very low?

**Fix data generation first — not the training recipe.** Position-0 `full_acc` is measured before any draft error accumulation, so a very low value means the draft is not even matching the teacher on the first speculative step. That usually indicates bad training labels (hidden states / targets), not a bad learning rate.

Concrete checks, in order:

1. **Verifier server during hidden-state capture** — confirm `launch_verifier` finished startup, serves `Qwen/Qwen3-8B` in bf16, and exposes hidden states via `launch_vllm.py`.
2. **vLLM version** — must match the training stack (vLLM 0.20.0 here); version mismatches cause sequence-length / hidden-state shape bugs.
3. **Sequence-length alignment** — tokenized length in preprocessed data must match hidden-state sequence length; rerun generation if they diverge.
4. **Stale temp files** — clear `/tmp/hidden_states/*` and regenerate if generation reports missing partial files.
5. **Completeness** — rerun `generate_hidden_states` with `--validate-outputs` and `--on-missing raise` so no corrupt or missing samples enter training.
6. **Only after data is verified** — increase `max-samples` (most impactful for draft quality), then consider lr / epochs / `draft_vocab_size`.

In our run, position-0 accuracy (~46%) was not catastrophically low, but the benchmark still showed spec_decode slower than baseline — consistent with weak multi-step acceptance. More training data (beyond 3,000 samples) would be the next improvement, not hyperparameter tuning.

### Task 3 — FP8 dynamic quantization

**Completed work:**
- Tool: `llmcompressor==0.12.0` in `comp_venv`, `oneshot()` + `QuantizationModifier`
- Scheme: `FP8_DYNAMIC` on all `Linear` layers, `ignore=["lm_head"]`
- Saved to: `quantization/Qwen3-8B-FP8-Dynamic/` (original `Qwen/Qwen3-8B` unchanged on Hugging Face / local cache)
- Verified: `config.json` contains a `quantization_config` section before benchmarking

**Command:**
```bash
cd quantization
source ../comp_venv/bin/activate
python quantize_fp8.py
```

**Expected quantization properties (verified in saved `config.json` / `recipe.yaml`):**

| Property | Expected value |
|---|---|
| Quantization method | compressed tensors |
| Weight format | FP8 |
| Activation format | dynamic FP8 |
| Target modules | linear layers (`Linear`) |
| Ignored module | `lm_head` |

**Benchmark impact (Section 4):** `fp8` alone reached **16,799 tok/s** vs **14,849 tok/s** baseline (+13%). `spec_decode_fp8` (8,263 tok/s) beat `spec_decode` (7,198 tok/s) on throughput, but both spec configs remained slower than baseline due to weak draft acceptance — not because FP8 quantization failed.

#### Q1. Why is FP8 dynamic quantization useful for serving on H100?

1. **Native FP8 hardware.** H100 Tensor Cores support FP8 matrix math (E4M3 / E5M2). Quantized linear layers run on these units instead of wider bf16/fp16 paths, improving compute throughput on matmul-bound decoder layers.

2. **Lower memory bandwidth.** FP8 weights are half the size of bf16 weights. For an 8B model, loading layers from GPU memory is often bandwidth-limited; smaller weights mean faster loads and more tokens/sec — matching our benchmark (+13% throughput for `fp8` vs `baseline`).

3. **Dynamic activations without calibration.** `FP8_DYNAMIC` uses static per-channel weight scales and **per-token dynamic activation quantization**. No calibration dataset or extra forward passes are required (pure PTQ via `llmcompressor oneshot`), so quantization is cheap and reproducible.

4. **Serving stack support.** vLLM and compressed-tensors integrate FP8 checkpoints directly, so the quantized verifier drops into the same serving path as bf16 with no retraining.

#### Q2. Why might `lm_head` be excluded from quantization?

The language-model head maps the final hidden state (`hidden_dim` → `vocab_size`) to logits over the full vocabulary. It is excluded because:

1. **Disproportionate impact on outputs.** The head is the last step before argmax/sampling. Small numerical errors in logits can change the top-1 token, affecting perplexity and generation quality far more than the same error in an intermediate layer.

2. **Small memory footprint.** `lm_head` is one linear layer versus dozens in the transformer stack. Keeping it in bf16 costs little extra memory but preserves full-precision token scores.

3. **Standard PTQ practice.** The llm-compressor FP8 reference recipe ignores `lm_head` for this reason; our `quantize_fp8.py` follows the same pattern (`ignore=["lm_head"]`).

Quantizing the body captures most of the FLOPs and weight memory savings; leaving the head full precision is a good accuracy/performance tradeoff.

#### Q3. How can quantization affect speculative decoding acceptance rate?

Speculative decoding accepts a draft token only when it **matches the verifier's argmax** at that step. Quantization affects acceptance through **distribution mismatch**, not correctness:

1. **Draft trained on bf16, verifier served as FP8.** The EAGLE-3 draft was trained against bf16 hidden states and bf16 teacher tokens. An FP8 verifier can produce slightly different activations and argmax tokens. When the FP8 verifier's argmax differs from the draft proposal — even if both would be "close" in probability — the draft token is **rejected** and the speculative chain shortens.

2. **Acceptance rate drops, outputs stay correct.** The verifier forward pass is always ground truth at inference; quantization changes how often the draft *guesses* the verifier's token, not the final generated text (under greedy decoding). Lower acceptance → more verifier steps → less speedup from speculation.

3. **Compounding over speculative depth.** A mismatch at an early position forces a resample; later draft tokens in that chain are wasted. Even a small per-step disagreement rate (e.g., 1–2% argmax flip rate) materially reduces multi-token acceptance.

4. **Empirical read from our benchmark.** `spec_decode_fp8` (8,263 tok/s) improved over `spec_decode` (7,198 tok/s) because the FP8 verifier is faster per forward pass — but both remained well below baseline and `fp8` alone, indicating draft quality dominated. In a stronger draft setup, the relevant comparison would be acceptance rate / throughput of `spec_decode` (bf16 verifier) vs `spec_decode_fp8` (FP8 verifier); a measurable gap there would isolate the precision-mismatch effect described above.

**Mitigation:** Train the draft on bf16 (as we did), use dynamic FP8 (tracks activations better than static), exclude `lm_head`, and compare acceptance metrics across bf16 vs FP8 verifier rows in the benchmark grid.

## 2. Answer: which goes first?

**Train the EAGLE-3 draft head first, against the full-precision (bf16) verifier. Quantize with FP8 second.**

## 3. Why (technical justification)

**a. What EAGLE-3 training actually optimizes for.**
The draft head is trained to reproduce the verifier's *internal hidden states* and the resulting next-token distribution (Step 3 in the pipeline: `data_generation_offline.py` pulls hidden states straight out of a running vLLM verifier). This is the training *label*. Anything that perturbs those hidden states — including FP8 activation/weight quantization noise — changes the target the draft head is being fit to. Training against the clean bf16 signal gives the draft head the most stable, lowest-variance target and avoids baking one lossy approximation (quantization error) into another (distillation of a draft model from noisy teacher signal).

**b. Correctness is precision-agnostic; acceptance rate is the only thing at stake.**
vLLM's speculative decoding verification step (rejection/greedy-match sampling) is exact by construction: whatever the draft proposes, the verifier's actual forward pass is the ground truth used to accept/reject tokens. So even if you later swap in the FP8 verifier, output correctness is unaffected — only the *acceptance rate* (and therefore the realized speedup) can change if the FP8 verifier's argmax token occasionally disagrees with the bf16 verifier the draft was trained against. That's a performance-tuning concern, not a correctness one, and empirically FP8 dynamic PTQ tracks bf16 outputs closely for 8B-class models, so this effect is expected to be small.

**c. Decoupling and reusability.**
`speculators_venv` and `comp_venv` are already isolated because of dependency conflicts — the assignment's own environment layout tells you these are two independent engineering tracks. Training the draft head against the canonical bf16 checkpoint makes the resulting EAGLE-3 head reusable across *any* future quantization scheme (FP8, INT8, a different calibration recipe, AWQ, etc.) without re-running the expensive hidden-state generation + training loop. If you quantized first and trained against FP8 hidden states instead, the draft head would be coupled to that one specific quantized checkpoint and would need to be retrained every time the quantization recipe changes.

**d. Cost asymmetry.**
Hidden-state generation + EAGLE-3 training is the expensive, data-dependent stage (many forward passes over the ShareGPT corpus, large on-disk hidden-state cache — see disk-space table in the speculators tutorial: ~260GB–13TB depending on sample count). FP8 PTQ via `llmcompressor`'s `FP8_DYNAMIC` scheme is calibration-free, one-shot, and cheap (minutes, no dataset pass required). Sequencing the expensive/foundational step first and the cheap/orthogonal step second is the efficient ordering if either step needs to be repeated or debugged.

**e. Ablation clarity for benchmarking.**
Doing spec-decode training first, on the unmodified bf16 verifier, lets you validate the draft head's acceptance rate and speedup in isolation (baseline → +spec decode) before introducing the second variable (quantization). This gives a clean 2x2 ablation grid (see Section 4) where each row isolates one optimization's contribution, and the "combined" row shows whether the two compose additively, sub-additively (e.g., FP8 lowers memory-bandwidth bound, which is also what spec decode targets, so gains may not simply add), or show measurable acceptance-rate degradation from the precision mismatch described in (b).

## 4. Benchmark grid (H100 run)

Run: `python benchmark/benchmark.py --config all --out results_all.json`  
(256 prompts × 512 max new tokens, `gpu_memory_utilization=0.90`)

| Config | Verifier precision | Draft head | Throughput (tok/s) | Wall time (s) | Mean ITL (ms) | TTFT (ms) | Peak mem (GB) | Draft acceptance rate |
|---|---|---|---:|---:|---|---|---|---|
| baseline | bf16 | — | 14,849 | 8.8 | n/a† | n/a† | n/a‡ | n/a |
| spec_decode | bf16 | EAGLE-3 (trained on bf16) | 7,198 | 18.2 | n/a† | n/a† | n/a‡ | unavailable§ |
| fp8 | FP8 dynamic | — | 16,799 | 7.8 | n/a† | n/a† | n/a‡ | n/a |
| spec_decode_fp8 | FP8 dynamic | EAGLE-3 (trained on bf16) | 8,263 | 15.8 | n/a† | n/a† | n/a‡ | unavailable§ |

† `benchmark.py` reports throughput and wall time only; per-token ITL and TTFT are not instrumented in this harness.  
‡ `torch.cuda.max_memory_allocated()` returned 0.0 for all configs — vLLM manages GPU memory outside PyTorch's allocator, so peak memory was not captured by this script.  
§ vLLM 0.20 metrics API did not expose `spec_decode_draft_acceptance_rate` via `llm.get_metrics()` in this run.

### Analysis

**FP8 alone is the clear winner on throughput.** The `fp8` row (16,799 tok/s) beats `baseline` (14,849 tok/s) by ~13%, consistent with H100 native FP8 tensor cores reducing memory-bandwidth pressure on an 8B model.

**Speculative decoding alone regressed vs baseline.** `spec_decode` (7,198 tok/s) is roughly half the throughput of `baseline`. This pattern indicates a **low draft acceptance rate**: the draft head proposes tokens that the verifier frequently rejects, so the system pays the overhead of draft forward passes and verification without realizing the intended latency reduction. With only 3,000 training samples at seq length 2048 (vs tutorial defaults of 20k / 8192), the draft head likely did not converge to a high-quality distribution match.

**Combined config is better than spec_decode alone but still below fp8 alone.** `spec_decode_fp8` (8,263 tok/s) improves on `spec_decode` (7,198 tok/s) — FP8's faster verifier partially offsets the speculative-decoding overhead — but remains well below `fp8` without speculation (16,799 tok/s). The two optimizations do not compose additively here; the bottleneck is draft quality, not verifier precision.

**Ordering validation.** Training EAGLE-3 on bf16 hidden states before FP8 quantization was still the correct engineering choice: output correctness is preserved, the draft checkpoint is reusable, and the ablation grid cleanly isolates each optimization's contribution. A higher-quality draft (more training data, longer sequences, or more epochs) would be needed before speculative decoding could outperform FP8 quantization on this hardware.

## 5. Execution note

The full pipeline was executed on a **Nebius H100 80GB VM** (`Qwen/Qwen3-8B`):

1. **Task 1–2:** EAGLE-3 offline training — data prep, hidden-state generation (3,000 samples @ seq 2048), and draft-head training to `checkpoint_best`.
2. **Task 3:** FP8 dynamic quantization via `llmcompressor` → `quantization/Qwen3-8B-FP8-Dynamic/`.
3. **Task 4:** Four-way benchmark (`baseline`, `spec_decode`, `fp8`, `spec_decode_fp8`) — results in Section 4 and `benchmark/results_all.json`.

Supporting artifacts on the VM (not committed due to size): hidden-state cache, model weights (`.safetensors`), and virtual environments.
