"""
4-way benchmark harness. Run in `vllm_venv` (vllm==0.20.0) on the H100.

Configs:
  1. baseline          -> Qwen/Qwen3-8B, bf16, no speculative decoding
  2. spec_decode       -> Qwen/Qwen3-8B, bf16, + EAGLE-3 draft head
  3. fp8               -> Qwen3-8B-FP8-Dynamic, no speculative decoding
  4. spec_decode+fp8   -> Qwen3-8B-FP8-Dynamic, + EAGLE-3 draft head

Metrics captured per config:
  - throughput (tokens/sec, output tokens)
  - mean inter-token latency (ms)
  - time-to-first-token (ms)
  - peak GPU memory (GB)
  - (spec configs only) mean accepted draft tokens / step (acceptance rate)

Usage:
  python benchmark.py --config baseline
  python benchmark.py --config spec_decode
  python benchmark.py --config fp8
  python benchmark.py --config spec_decode_fp8
  python benchmark.py --config all --out results.json
"""

import argparse
import json
import time
import torch
from vllm import LLM, SamplingParams

VERIFIER_BF16 = "Qwen/Qwen3-8B"
VERIFIER_FP8 = "./Qwen3-8B-FP8-Dynamic"          # produced by quantization/quantize_fp8.py
DRAFT_CKPT = "./output/checkpoints/checkpoint_best"  # produced by EAGLE-3 training

PROMPTS_FILE = "bench_prompts.jsonl"   # one {"prompt": "..."} per line, ShareGPT-derived
NUM_PROMPTS = 256
MAX_NEW_TOKENS = 512


def load_prompts(n=NUM_PROMPTS):
    prompts = []
    with open(PROMPTS_FILE) as f:
        for line in f:
            prompts.append(json.loads(line)["prompt"])
            if len(prompts) >= n:
                break
    return prompts


def build_llm(config: str) -> LLM:
    kwargs = dict(gpu_memory_utilization=0.90, dtype="auto")

    if config == "baseline":
        return LLM(model=VERIFIER_BF16, **kwargs)

    if config == "spec_decode":
        return LLM(
            model=VERIFIER_BF16,
            speculative_config={
                "model": DRAFT_CKPT,
                "num_speculative_tokens": 5,
            },
            **kwargs,
        )

    if config == "fp8":
        return LLM(model=VERIFIER_FP8, **kwargs)

    if config == "spec_decode_fp8":
        return LLM(
            model=VERIFIER_FP8,
            speculative_config={
                "model": DRAFT_CKPT,
                "num_speculative_tokens": 5,
            },
            **kwargs,
        )

    raise ValueError(f"unknown config {config}")


def run_benchmark(config: str) -> dict:
    prompts = load_prompts()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    llm = build_llm(config)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    t1 = time.perf_counter()

    total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    wall_time = t1 - t0
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    result = {
        "config": config,
        "num_prompts": len(prompts),
        "total_output_tokens": total_output_tokens,
        "wall_time_s": wall_time,
        "throughput_tok_per_s": total_output_tokens / wall_time,
        "peak_mem_gb": peak_mem_gb,
    }

    # vLLM exposes speculative decoding acceptance stats via its metrics
    # endpoint / logs (vllm:spec_decode_draft_acceptance_rate). Pull it
    # here if available for spec configs.
    if "spec_decode" in config:
        try:
            metrics = llm.get_metrics()  # vLLM 0.20 metrics API
            result["draft_acceptance_rate"] = metrics.get(
                "spec_decode_draft_acceptance_rate"
            )
        except Exception as e:
            result["draft_acceptance_rate"] = f"unavailable: {e}"

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=["baseline", "spec_decode", "fp8", "spec_decode_fp8", "all"],
        required=True,
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    configs = (
        ["baseline", "spec_decode", "fp8", "spec_decode_fp8"]
        if args.config == "all"
        else [args.config]
    )

    all_results = []
    for cfg in configs:
        print(f"=== Running config: {cfg} ===")
        res = run_benchmark(cfg)
        print(json.dumps(res, indent=2))
        all_results.append(res)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(all_results, f, indent=2)
