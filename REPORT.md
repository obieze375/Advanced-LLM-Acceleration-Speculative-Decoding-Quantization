# Advanced LLM Acceleration: EAGLE-3 Speculative Decoding + FP8 Quantization
**Model:** Qwen/Qwen3-8B &nbsp;|&nbsp; **Hardware:** 1x H100 80GB

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

## 4. Benchmark grid (to fill in from your H100 run)

Run `benchmark/benchmark.py --config all --out results.json` after completing both stages. Expected structure:

| Config | Verifier precision | Draft head | Throughput (tok/s) | Mean ITL (ms) | TTFT (ms) | Peak mem (GB) | Draft acceptance rate |
|---|---|---|---|---|---|---|---|
| baseline | bf16 | — | *measured* | *measured* | *measured* | *measured* | n/a |
| spec_decode | bf16 | EAGLE-3 (trained on bf16) | *measured* | *measured* | *measured* | *measured* | *measured* |
| fp8 | FP8 dynamic | — | *measured* | *measured* | *measured* | *measured* | n/a |
| spec_decode_fp8 | FP8 dynamic | EAGLE-3 (trained on bf16) | *measured* | *measured* | *measured* | *measured* | *measured* |

**What to look for when you fill this in:**
- `fp8` alone should cut peak memory roughly in half vs `baseline` and improve throughput (memory-bandwidth-bound regime for an 8B model on H100).
- `spec_decode` alone should improve mean inter-token latency substantially with only a small memory overhead (the draft head is small relative to the 8B verifier).
- `spec_decode_fp8` (combined) should show the largest throughput/latency win, but check `draft_acceptance_rate` against the `spec_decode` (bf16 verifier) row — a drop here is exactly the precision-mismatch effect described in 3(b), and quantifying it is a good bonus analysis for the writeup.

## 5. Honest scope note

This bundle is a complete, runnable pipeline (commands verified against the current `speculators` offline-training tutorial and `llm-compressor` FP8 `README`), but it was assembled in a CPU-only sandbox with no GPU and no access to the Hugging Face Hub — so the numbers in Section 4 were **not** actually measured here. Run the scripts in order on your H100 and drop the real numbers into that table; the reasoning in Section 3 is independent of the specific measured values.
