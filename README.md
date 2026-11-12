# Task Model Distiller ⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](pyproject.toml)
[![Architecture: 3--Pillar](https://img.shields.io/badge/Architecture-3--Pillar%20Distillation-orange.svg)](#the-3-pillar-architecture)
[![Inference: CPU--Friendly](https://img.shields.io/badge/Inference-CPU%20%2F%20Ollama%20%2F%20ONNX-purple.svg)](#deployment-targets)

> **Distill frontier agent capabilities into small, CPU-friendly local models (`0.5B`–`3B` GGUF / Ollama / ONNX) using Full-Session Synthetic Trace Generation, Two-Stage Adversarial Evals (The Referee), and Private Regression Benchmarks (The Scoreboard).**

---

## 🎯 The Problem & Vision

Calling frontier models (GPT-4, Claude 3.5 Sonnet, Gemini 1.5 Pro) for narrow, repetitive agent workflows costs **\$20–\$100+ per million tokens** and incurs significant cloud latency.

Most production agent tasks:
- Call only **3 to 7 specific tools**.
- Follow strict deterministic schema and invariant rules.
- Require zero creative hallucination.

**Task Model Distiller** transfers narrow frontier capabilities into compact open-weight models (`Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `SmolLM2-1.7B`, `Llama-3.2-1B`) that run locally on **ordinary CPUs at <1.5 GB RAM** with sub-50ms token latency.

---

## 🏛️ The 3-Pillar Architecture

```
                  ┌────────────────────────────────────────┐
                  │    Frontier Teacher (In-Workspace)     │
                  └───────────────────┬────────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
  ┌─────────────────────────┐                   ┌─────────────────────────┐
  │        PILLAR 1         │                   │        PILLAR 2         │
  │   Full-Session Traces   │                   │   Adversarial Referee   │
  │   (Happy + Recovery)    │                   │   (Stage 1 + Stage 2)   │
  └────────────┬────────────┘                   └────────────┬────────────┘
               │                                             │
               │         ┌─────────────────────────┐         │
               └────────►│  CPU LoRA Fine-Tuning   │◄────────┘
                         │   (train_cpu_lora.py)   │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │        PILLAR 3         │
                         │     The Scoreboard      │
                         │    (Private Locked)     │
                         └────────────┬────────────┘
                                      │ Failures
                                      ▼
                         ┌─────────────────────────┐
                         │   Targeted Flywheel     │
                         │  (Synthesize Fix Traces)│
                         └─────────────────────────┘
```

1. **Pillar 1 — Full-Session Synthetic Trace Generation:**
   Synthesizes full multi-turn conversational trajectories (`system` $\rightarrow$ `user` $\rightarrow$ `assistant (tool_call)` $\rightarrow$ `tool (output)` $\rightarrow$ `assistant (final response)`), covering happy paths, edge cases, and tool recovery.

2. **Pillar 2 — Two-Stage Adversarial Referee:**
   - **Stage 1 (Deterministic Check):** Hard rules enforcing required tool calls before claims, catching phantom executions and silent error masking.
   - **Stage 2 (Execution Referee):** Strict comparison between the final assistant message and the tool output log.
   - **Adversarial Calibration:** Uses `mutate_traces.py` to inject poisoned/hallucinated traces and proves the referee catches 100% of lies before training begins.

3. **Pillar 3 — Private Regression Scoreboard:**
   Evaluates base and fine-tuned models against locked private test cases, outputting categorized failure reports to fuel iterative data flywheel loops.

---

## 📁 Repository Structure

```text
task-model-distiller/
├── SKILL.md                          # Antigravity IDE skill definition & orchestrator guide
├── pyproject.toml                    # Package metadata & build configuration
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
├── README.md                         # Documentation & guides
├── scripts/
│   ├── __init__.py
│   ├── validate_traces.py            # Validates synthetic dataset against spec.json
│   ├── referee_stage1.py             # Deterministic log referee enforcing hard invariants
│   ├── mutate_traces.py              # Adversarial trace mutator for referee calibration
│   ├── scoreboard.py                 # Private benchmark runner & failure categorizer
│   └── train_cpu_lora.py             # Low-RAM / CPU LoRA fine-tuning & weight merger
├── templates/
│   ├── stage2_judge_prompt.md        # Stage 2 narrow referee prompt template
│   └── Modelfile.template            # Ollama CPU deployment Modelfile
└── examples/
    └── incident-triager/
        ├── spec.json                 # Example task specification & tool schemas
        └── seeds.jsonl               # Example multi-turn seed trajectories
```

---

## 🚀 Quick Start

### 1. Installation

Clone and install dependencies:
```bash
git clone https://github.com/Virat-Ghatge/task-model-distiller.git
cd task-model-distiller
pip install -r requirements.txt
```

To install as a global skill inside Google Antigravity IDE:
```bash
mkdir -p ~/.gemini/config/skills/task-model-distiller
cp -r * ~/.gemini/config/skills/task-model-distiller/
```

### 2. The 5-Phase Workflow

#### Phase 0: Define Task Scope & Seeds
Initialize `.distill/<task_name>/spec.json` and create 3–5 gold-standard seeds in `.distill/<task_name>/seeds/seeds.jsonl`.

#### Phase 1: Synthesize & Validate Traces
Synthesize 50–100 diverse sessions into `.distill/<task_name>/dataset/train_traces.jsonl`, then validate:
```bash
python scripts/validate_traces.py \
  --spec .distill/<task_name>/spec.json \
  --input .distill/<task_name>/dataset/train_traces.jsonl
```

#### Phase 2: Calibrate the Adversarial Referee
Generate sabotaged traces (phantom tool calls, silent failures) and verify the referee:
```bash
python scripts/mutate_traces.py \
  --input .distill/<task_name>/dataset/train_traces.jsonl \
  --output .distill/<task_name>/referee/mutated_calibration.jsonl

python scripts/referee_stage1.py \
  --invariants .distill/<task_name>/referee/invariants.json \
  --sessions .distill/<task_name>/referee/mutated_calibration.jsonl \
  --calibrate
```

#### Phase 3: Benchmark Base Local Model
```bash
python scripts/scoreboard.py \
  --model qwen2.5:0.5b \
  --benchmark .distill/<task_name>/scoreboard/benchmark_locked.jsonl \
  --invariants .distill/<task_name>/referee/invariants.json \
  --report-out .distill/<task_name>/scoreboard/baseline_local.json
```

#### Phase 4: CPU-Friendly LoRA Fine-Tuning
```bash
python scripts/train_cpu_lora.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset .distill/<task_name>/dataset/train_traces.jsonl \
  --output-dir .distill/<task_name>/training \
  --epochs 3 \
  --merge
```

---

## 📦 Multi-Target Deployment

The fine-tuning script exports standard Hugging Face weights (`safetensors`) in `.distill/<task>/training/merged_hf/`. You can deploy to any target:

| Target | Command | RAM / Latency |
| :--- | :--- | :--- |
| **Ollama (4-bit GGUF)** | `ollama create my-model -f Modelfile` | `<1.2 GB RAM`, CPU-ready |
| **llama.cpp** | `python llama.cpp/convert_hf_to_gguf.py merged_hf/` | Native C++ edge execution |
| **ONNX Runtime** | `optimum-cli export onnx --model merged_hf/ onnx/` | C#, C++, Rust, DirectML |
| **vLLM / SGLang** | Load directly from `merged_hf/` | High-throughput GPU serving |
| **Apple Silicon (MLX)** | `python -m mlx_lm.convert --hf-path merged_hf/` | Unified memory M-series |

---

## 🛡️ License

Released under the [MIT License](LICENSE).
Created with ⚡ by [Virat Ghatge](https://github.com/Virat-Ghatge).
