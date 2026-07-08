# README — Advanced LLM Acceleration: Speculative Decoding & Quantization

This README maps the homework assignment's requirements directly onto the
scripts in this folder: what to run, in what order, and **where to write
your answers**. It assumes you are running on a provisioned VM with
**1x NVIDIA H100 80GB** (Colab does not provide H100s, so this cannot be run
there — see setup step 0 below).

```
lab/
├── README.md                  <- you are here
├── REPORT.md                  <- your written answer + results table go here
├── setup/
│   ├── 00_setup_speculators_venv.sh
│   ├── 01_setup_vllm_venv.sh
│   └── 02_setup_comp_venv.sh
├── speculators/
│   └── run_eagle3_pipeline.sh
├── quantization/
│   └── quantize_fp8.py
└── benchmark/
    └── benchmark.py
```

---

## 0. Prerequisites

- A Linux VM with 1x H100 80GB, SSH/root access (AWS, Lambda Labs, RunPod, CoreWeave, GCP, etc.)
- Python 3.12 available (`python3.12 --version`)
- `git`
- A Hugging Face account/token if `Qwen/Qwen3-8B` requires auth in your region:
  ```bash
  pip install --user huggingface_hub
  huggingface-cli login
  ```
- Upload/clone this entire `lab/` folder onto the VM. All commands below assume
  you `cd` into `lab/` first, then into the relevant subfolder.

---

## 1. Build the three required environments

The assignment requires **three separate venvs** (dependency conflicts between
the training and serving stacks mean one shared env is not expected to work).
This step satisfies the "Required Library Versions" table in the assignment.

```bash
cd setup
bash 00_setup_speculators_venv.sh   # speculators_venv: speculators @ git tag v0.5.0, editable install
bash 01_setup_vllm_venv.sh          # vllm_venv: vllm==0.20.0, fastapi<0.137
bash 02_setup_comp_venv.sh          # comp_venv: llmcompressor==0.12.0
cd ..
```

Confirm each script's sanity-check output before moving on. Do **not** submit
the `speculators_venv/`, `vllm_venv/`, or `comp_venv/` folders themselves —
only the scripts and your results/report (per the assignment's "Do not submit
the virtual environments" instruction).

---

## 2. Get a ShareGPT-style dataset

The assignment requires a ShareGPT-style conversational dataset as the
training data source. Example:

```bash
source speculators_venv/bin/activate
huggingface-cli download anon8231489123/ShareGPT_Vicuna_unfiltered \
  --repo-type dataset --local-dir ./sharegpt_raw
```

If your course provided a specific dataset link instead, use that and point
`speculators/run_eagle3_pipeline.sh`'s `prepare_data` step at it.

---

## 3. Train the EAGLE-3 draft head (`speculators/`)

This satisfies the assignment's **"train an EAGLE-3 speculative decoding draft
head"** requirement, following the offline training tutorial referenced in
the assignment.

Run these four sub-steps in order (two terminals needed for steps 2–3, since
the verifier server must stay running while hidden states are generated):

```bash
cd speculators

# Terminal A
bash run_eagle3_pipeline.sh prepare_data
bash run_eagle3_pipeline.sh launch_verifier      # leave running

# Terminal B (new SSH session, cd lab/speculators)
bash run_eagle3_pipeline.sh generate_hidden_states

# Back in Terminal A: Ctrl+C to stop the verifier server, then:
bash run_eagle3_pipeline.sh train_draft_head
bash run_eagle3_pipeline.sh smoke_test
```

**Output:** a trained draft head checkpoint at `speculators/output/checkpoints/checkpoint_best`.

**Where this feeds into your answer:** the *order* in which this step is run
relative to Section 4 (quantization) is exactly the assignment's core
question. See `REPORT.md` Section 2–3 for the answer and reasoning — the
short version is: **this step is run first**, against the full-precision
(bf16) verifier, before quantization.

---

## 4. Quantize the verifier with FP8 dynamic quantization (`quantization/`)

This satisfies the assignment's **"quantize the verifier model with FP8
dynamic quantization"** requirement, following the `llm-compressor`
`FP8_DYNAMIC` reference linked in the assignment.

```bash
cd ../quantization
source ../comp_venv/bin/activate
python quantize_fp8.py
```

**Output:** `quantization/Qwen3-8B-FP8-Dynamic/` — the quantized verifier checkpoint.

---

## 5. Benchmark all four configurations (`benchmark/`)

This satisfies the assignment's **"benchmark baseline, speculative decoding,
quantization, and the combined setup"** requirement.

```bash
cd ../benchmark
source ../vllm_venv/bin/activate
```

Create `bench_prompts.jsonl` (one `{"prompt": "..."}` per line) from a
held-out slice of your ShareGPT data (not the slice used for training).

Run each config (one at a time, to avoid GPU memory contention):

```bash
python benchmark.py --config baseline         --out results_baseline.json
python benchmark.py --config spec_decode      --out results_spec.json
python benchmark.py --config fp8              --out results_fp8.json
python benchmark.py --config spec_decode_fp8  --out results_combined.json
```

Or all four in one call:

```bash
python benchmark.py --config all --out results_all.json
```

**Where the answer goes:** copy the four rows of measured throughput,
latency, memory, and (for spec configs) draft acceptance rate into the
**benchmark table in `REPORT.md`, Section 4**.

---

## 6. Write up your final answer (`REPORT.md`)

`REPORT.md` is already structured to match every deliverable the assignment
asks for. Fill in / finalize each section as follows:

| Assignment requirement | Where to put it in `REPORT.md` |
|---|---|
| "explain which optimization should be applied first and why" | **Section 2** (short answer) + **Section 3** (technical justification) — already drafted, review and edit in your own words if required by your course |
| "Your answer must be supported by... benchmark results" | **Section 4** — replace every `*measured*` placeholder with your real numbers from `results_all.json` |
| "...the training setup" | **Section 1** (pipeline/environment overview) — already summarizes venvs, tools, and artifacts; adjust if you changed any hyperparameters (epochs, seq length, num speculative tokens, etc.) |
| Honest accounting of what was/wasn't run in this sandbox | **Section 5** — replace with a note confirming you ran the full pipeline on your own H100 VM, once you've done so |

**Do not submit:**
- `speculators_venv/`, `vllm_venv/`, `comp_venv/` (per assignment instructions)
- large intermediate artifacts (`output/hidden_states/`, raw dataset downloads) unless your course asks for them

**Do submit:**
- This `lab/` folder (scripts) with your edits
- `REPORT.md` with the completed answer and filled-in benchmark table
- `results_all.json` (or the four separate result files) as supporting evidence
- The trained draft head checkpoint and/or FP8 checkpoint, if your course requires artifact submission (check file size limits — these can be large)
