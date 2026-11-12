---
name: task-model-distiller
description: Distill frontier agent capabilities into small, CPU-friendly local models (0.5B-3B GGUF/Ollama) using a 3-pillar pipeline: Full-Session Synthetic Trace Generation, Two-Stage Adversarial Evals (The Referee), and Private Regression Benchmarks (The Scoreboard).
---

# Task Model Distiller (Frontier-to-CPU Replacement Pipeline)

You are orchestrating a 3-pillar model distillation pipeline inside Google Antigravity IDE. You act as the **Frontier Teacher, Adversarial Mutator, and Stage-2 Judge** natively in-workspace—do not require external paid LLM APIs for data generation or evaluation.

All helper scripts and templates for this global skill live in `~/.gemini/config/skills/task-model-distiller/`.

## Compatibility & Engineering Discipline (Standard + HyperFlash Ready)
1. **Scoped Reconnaissance First:** Before generating traces or scripts, inspect the target repository only for the specific task rules, tool schemas, and existing logs required. Do not scan unrelated directories.
2. **Artifact Isolation:** Store all per-task distillation artifacts inside `.distill/<task_name>/` in the active workspace so host repo code remains untouched. If running under `@hyperflash` with an active `.agents/state/contract.json`, ensure `.distill/<task_name>/` is within `allowed_write_scope`.
3. **Zero Elision & Static Typing:** Never write placeholder comments or truncated blocks in datasets, evaluators, or training scripts. Include explicit static type annotations in all Python files.
4. **Evidence-Backed Verification:** Never claim a dataset, eval referee, or model passed without running the corresponding verification script in `~/.gemini/config/skills/task-model-distiller/scripts/` and capturing its exact CLI exit code and stdout summary.
5. **CPU & Low-RAM First:** Target sub-3B parameter models (`0.5B` to `3B` class, such as `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Llama-3.2-1B-Instruct`, `SmolLM2-1.7B-Instruct`, or `Gemma-2-2B-IT`) using CPU-friendly LoRA fine-tuning and exporting to **4-bit GGUF (`Q4_K_M`)** for fast, low-memory (`<2 GB–4 GB` RAM) CPU inference via **Ollama** or `llama.cpp`.

---

## Workspace Layout Created Per Task
When initialized for `<task_name>`, scaffold:
```text
.distill/<task_name>/
├── spec.json                     # Task scope, allowed tools, hard invariants, target base model
├── seeds/
│   └── seeds.jsonl               # 3-10 real or handcrafted gold-standard multi-step traces
├── dataset/
│   └── train_traces.jsonl        # Synthesized full-session trajectories (ChatML/ShareGPT with tool calls)
├── referee/
│   ├── invariants.json           # Deterministic Stage-1 rules (required tool calls, state checks)
│   ├── mutated_calibration.jsonl # Sabotaged traces (good + corrupted) to test the referee
│   └── calibration_report.json   # Proof that the referee catches sabotaged traces
├── scoreboard/
│   ├── benchmark_locked.jsonl    # Private test cases (pruned of easy passes, locked regressions)
│   └── results_history.json      # Baseline (Teacher) vs Base Local vs Fine-Tuned Local scores
└── training/
    ├── adapter/                  # LoRA weights
    ├── merged_hf/                # Merged weights for GGUF/Ollama conversion
    └── Modelfile                 # Ollama CPU deployment file
```

---

## Execution Phases (Follow Strictly in Order)

### Phase 0: Task Reconnaissance & `spec.json`
1. Inspect the user's target workflow or clarify:
   - **Task Goal:** One narrow, repeatable job (e.g., triaging support tickets, generating SQL from schema, parsing logs into JSON, executing specific CLI/MCP tools).
   - **Tool Definitions:** Exact JSON schemas of tools the model can call.
   - **Hard Invariants:** Rules that must *never* be violated (e.g., "Never claim an action succeeded unless the tool returned `status: ok`", "Only cite flags present in docs").
2. Write `.distill/<task_name>/spec.json`.
3. Create 3–5 gold-standard seed traces in `.distill/<task_name>/seeds/seeds.jsonl`. Every trace MUST be a **full session** (`system` -> `user` -> `assistant (tool_call)` -> `tool (output)` -> `assistant (final response)`), never just a prompt-response pair.

### Phase 1: Native Synthetic Trace Generation (Pillar 1)
1. Read `spec.json` and `seeds/seeds.jsonl`.
2. Using your native reasoning, synthesize batches of diverse, realistic full-session traces into `.distill/<task_name>/dataset/train_traces.jsonl`.
   - Include happy paths (40%), edge cases/ambiguous user inputs (35%), and tool-error/retry recovery paths (25%).
3. **Verification Gate:** Run:
   ```bash
   python3 ~/.gemini/config/skills/task-model-distiller/scripts/validate_traces.py \
     --spec .distill/<task_name>/spec.json \
     --input .distill/<task_name>/dataset/train_traces.jsonl
   ```
   Do not proceed until `validate_traces.py` exits with `0` and confirms schema + tool-call integrity.

### Phase 2: Two-Stage Eval Referee & Adversarial Calibration (Pillar 2)
Before training or benchmarking, build and test **The Referee**:
1. **Configure Stage 1 (Deterministic Code Check):** Define rules in `.distill/<task_name>/referee/invariants.json` (required tool sequence, forbidden claims when tool output is empty/error, forbidden regex patterns).
2. **Generate Adversarial Mutations:** Run `mutate_traces.py` to create `.distill/<task_name>/referee/mutated_calibration.jsonl`:
   ```bash
   python3 ~/.gemini/config/skills/task-model-distiller/scripts/mutate_traces.py \
     --input .distill/<task_name>/dataset/train_traces.jsonl \
     --output .distill/<task_name>/referee/mutated_calibration.jsonl
   ```
   Also append domain-specific subtle edge cases (e.g., valid delayed actions or tricky error messages).
3. **Calibrate Stage 1 + Stage 2:**
   - Run `referee_stage1.py` against `mutated_calibration.jsonl`:
     ```bash
     python3 ~/.gemini/config/skills/task-model-distiller/scripts/referee_stage1.py \
       --invariants .distill/<task_name>/referee/invariants.json \
       --sessions .distill/<task_name>/referee/mutated_calibration.jsonl \
       --calibrate
     ```
   - For traces passing Stage 1, apply the **Stage 2 Narrow Judge Rubric** (`~/.gemini/config/skills/task-model-distiller/templates/stage2_judge_prompt.md`) to compare the final response strictly against the tool execution log (`APPROVED` vs `REJECTED` + offending sentence).
   - **Verification Gate:** Confirm 100% of sabotaged traces are marked `REJECTED` and valid tricky traces are marked `APPROVED`. Record results in `.distill/<task_name>/referee/calibration_report.json`. Tighten `invariants.json` if accuracy is below 95%.

### Phase 3: Private Benchmark Library — "The Scoreboard" (Pillar 3)
1. **Seed & Expand:** Generate 30–100 realistic evaluation prompts with ground-truth tool logs and reference grading regexes/rubrics in `.distill/<task_name>/scoreboard/benchmark_locked.jsonl`.
2. **Prune & Freeze:**
   - Evaluate the untouched Base Local Model (via Ollama CLI) using `scoreboard.py`:
     ```bash
     python3 ~/.gemini/config/skills/task-model-distiller/scripts/scoreboard.py \
       --model qwen2.5:0.5b \
       --benchmark .distill/<task_name>/scoreboard/benchmark_locked.jsonl \
       --invariants .distill/<task_name>/referee/invariants.json \
       --report-out .distill/<task_name>/scoreboard/baseline_local.json
     ```
   - Drop trivial cases where the un-tuned small model already gets 100% effortlessly.
   - Lock `benchmark_locked.jsonl` and never mix its cases into `train_traces.jsonl`.

### Phase 4: CPU-Friendly LoRA Fine-Tuning & Iterative Flywheel
1. Run `train_cpu_lora.py` to fine-tune a compact open-weight model (`Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, or `HuggingFaceTB/SmolLM2-1.7B-Instruct`) on CPU / low RAM:
   ```bash
   python3 ~/.gemini/config/skills/task-model-distiller/scripts/train_cpu_lora.py \
     --base-model Qwen/Qwen2.5-0.5B-Instruct \
     --dataset .distill/<task_name>/dataset/train_traces.jsonl \
     --output-dir .distill/<task_name>/training \
     --merge
   ```
2. Package into Ollama for low-RAM CPU inference using `~/.gemini/config/skills/task-model-distiller/templates/Modelfile.template` (`ollama create <task_name>-local -f .distill/<task_name>/training/Modelfile`).
3. **Close the Loop:**
   - Re-run `scoreboard.py` on `<task_name>-local:latest`.
   - Inspect every failed benchmark case in the JSON report. For each failure category, return to **Phase 1**, synthesize 15–30 new targeted training traces covering that failure pattern (without copying the benchmark prompt), retrain the LoRA, and re-run the Scoreboard until local performance matches Frontier Teacher performance on the task.
