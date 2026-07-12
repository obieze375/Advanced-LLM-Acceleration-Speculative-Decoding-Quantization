# Coursework submission checklist (if VM benchmarks won't finish)

You already have enough to submit **most of the assignment** without a working `vllm bench serve` run.

## What to submit on GitHub

```bash
cd ~/Advanced-LLM-Acceleration-Speculative-Decoding-Quantization
git add REPORT.md README.md setup/ benchmark/ quantization/ speculators/
git status   # no venvs, no hidden_states, no *.safetensors
git commit -m "feat: submit homework report and pipeline scripts"
git push origin main
```

**Already in repo (from earlier VM run):**
- `REPORT.md` — Tasks 1–4 Q&A, ordering argument, analysis
- `benchmark/results_*.json` — offline `benchmark.py` runs (different scale than rubric)
- `quantization/Qwen3-8B-FP8-Dynamic/` configs
- All setup scripts + pipeline

## Task 4 numbers for REPORT / notebook

If `vllm bench serve` cannot complete, use the **course reference table** and state infrastructure issues honestly:

| Config | Output tok/s | TTFT (ms) | TPOT (ms) | Acceptance |
|---|---:|---:|---:|---:|
| Speculative decoding | 1258.65 | 78.17 | 5.76 | 22.48% |
| FP8 quantization | 1566.56 | 51.18 | 4.90 | n/a |
| FP8 + speculative decoding | 1766.55 | 30.24 | 4.28 | 36.50% |

Draft tokens: **2** (bf16 spec), **1** (FP8+spec) — justified by acceptance length ~1.4.

Add one sentence: *Live `vllm bench serve` on Nebius H100 blocked by environment issues (Triton/libcuda, then missing `datasets` bench extra); reference results used for Task 4 throughput comparison.*

## One-command bench fix (try once more)

```bash
source vllm_venv/bin/activate
uv pip install datasets aiohttp
python -c "import datasets; print('datasets OK')"
pkill -f "vllm serve" || true
bash setup/04_vllm_nebius_fix_and_bench.sh task4
```

## Rubric reality

Automated rubric (>1250 / >1550 / >1750 tok/s) needs **your** `vllm bench serve` logs. Without them you may lose Task 4 bench points but keep report/script points.

## Destroy VM when done

```bash
cd terraform && source ~/.nebius/terraform-auth.env && terraform destroy
```
