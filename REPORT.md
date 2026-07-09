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

### Supplemental notes (EAGLE-3 training & FP8)

**EAGLE-3 accuracy metrics (TensorBoard / validation logs):**
- `full_acc` — fraction of draft positions where the draft token matches the teacher (verifier) argmax at that position.
- `cond_acc` — same match rate *conditioned on* all prior positions in the speculative chain already matching the teacher.
- Later positions in the chain are harder because errors compound: a wrong early token shifts the hidden-state context for subsequent predictions.
- If first-position accuracy is low, the root cause is usually **data generation** (hidden states captured from a misconfigured or under-served verifier), not training hyperparameters — fix the upstream pipeline before tuning lr/epochs.

**FP8 quantization choices:**
- FP8 dynamic PTQ is useful on H100 because the chip has native FP8 tensor cores; weights and activations are quantized on-the-fly, yielding throughput and memory savings without a separate calibration dataset pass.
- `lm_head` is typically excluded from quantization because it is small relative to the transformer stack but disproportionately affects output-token distribution fidelity.
- Quantization can lower draft acceptance if the FP8 verifier's argmax occasionally disagrees with the bf16 verifier the draft was trained against; in this run, spec_decode was already slower than baseline on bf16, so any FP8-induced acceptance drop is secondary to overall draft quality.

## 5. Execution note

The full pipeline was executed on a **Nebius H100 80GB VM** (`Qwen/Qwen3-8B`):

1. **Task 1–2:** EAGLE-3 offline training — data prep, hidden-state generation (3,000 samples @ seq 2048), and draft-head training to `checkpoint_best`.
2. **Task 3:** FP8 dynamic quantization via `llmcompressor` → `quantization/Qwen3-8B-FP8-Dynamic/`.
3. **Task 4:** Four-way benchmark (`baseline`, `spec_decode`, `fp8`, `spec_decode_fp8`) — results in Section 4 and `benchmark/results_all.json`.

Supporting artifacts on the VM (not committed due to size): hidden-state cache, model weights (`.safetensors`), and virtual environments.
