"""
FP8 dynamic quantization of the Qwen/Qwen3-8B verifier.

Run in `comp_venv` (llmcompressor==0.12.0), AFTER the EAGLE-3 draft head has
already been trained against the full-precision verifier's hidden states
(see ../speculators/run_eagle3_pipeline.sh and ../REPORT.md for why this
ordering was chosen).

Scheme: FP8_DYNAMIC
  - static, per-channel weight quantization
  - dynamic, per-token activation quantization
  - no calibration dataset required (pure PTQ)

Reference: vllm-project/llm-compressor examples/quantization_w8a8_fp8
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-8B"

def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Quantize all Linear layers to FP8, leave the LM head unquantized
    # (standard practice: the head is small relative to total FLOPs/memory
    # but disproportionately affects output-distribution fidelity).
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
    )

    oneshot(model=model, recipe=recipe)

    save_dir = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Saved FP8-quantized verifier to: {save_dir}")


if __name__ == "__main__":
    main()
