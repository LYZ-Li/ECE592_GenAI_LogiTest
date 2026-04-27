# ECE592 GenAI Spring 2026 — LogiTest: Memory Compression in Long-Horizon LLM Planning

**Project:** *When Memory Fails the Plan: Logical Dependency Preservation Under Context Compression in Long-Horizon LLM Planning*

---

## Overview

This repository contains all code for an empirical study investigating whether memory compression strategies—truncation and summarization—preserve the logical dependencies required for long-horizon robot planning tasks.

Nine language models are evaluated in a **stepwise rollout harness** across five memory policies and two prompting modes on synthetic PDDL logistics tasks. At every step the model receives the task description, the current symbolic world state, and a compressed memory representation of prior history; its output is parsed and symbolically validated before the next step begins. A fully automated symbolic constraint checker measures goal satisfaction, precondition violations, and a novel dependency-boundary violation metric.

**Key finding:** Planning capability and memory robustness are distinct thresholds. Four models fail entirely due to output-compliance deficits that are independent of memory policy. For capable models, compressed memory induces genuine dependency-boundary violations and—at the longest horizons—long-horizon tracking collapse.

---

## Key Results

| Model | Attempted / Expected | Goals | GSR | Tier |
|---|---|---|---|---|
| GLM-5 | 41 / 50 | 41 | **100%** | Capable |
| Qwen3.6-plus | 32 / 50 | 32 | **100%** | Capable |
| DeepSeek-v3.2 | 40 / 50 | 36 | 90% | Functional–fragile |
| Minimax-M2.5 | 41 / 50 | 37 | 90% | Functional–fragile |
| Gemma4-27b | 50 / 50 | 21 | 42% | Parse-limited |
| LLaMA-3.2-3b | 41 / 50 | 0 | 0% | Baseline-limited |
| LLaMA3-8b | 31 / 50 | 0 | 0% | Baseline-limited |
| Qwen3.5-4b | 50 / 50 | 0 | 0% | Baseline-limited |
| Qwen3.5-9b | 25 / 50 | 0 | 0% | Baseline-limited |

*GSR = goal-satisfaction rate on attempted cells. Total: 351 trials, 167 successes (47.6%).*

**Failure breakdown (351 trials):**
- 122 terminal parse errors (empty output, wrong JSON key, hallucinated action names)
- 59 terminal precondition violations
- 3 long-horizon tracking collapses (DeepSeek, 120-step cap, loop depth 12–13)
- 48 dependency-boundary events across 23 trials (capable models only)

**Difficulty progression (functional cohort — GLM-5, Qwen3.6-plus, DeepSeek, Minimax):**

| Tier | Oracle steps | Attempted | Goals |
|---|---|---|---|
| Easy | 6 | 40 | 40 |
| Medium | 18 | 40 | 39 |
| Hard | 29 | 40 | 40 |
| Very hard | 49 | 32 | 25 |
| Extreme | 94 | 2 | 2 |

---

## Code Organisation

```
ECE592_GenAI_LogiTest/
├── configs/
│   ├── experiment.yaml          # Master experiment config (memory policies, dataset, eval)
│   └── models/                  # Per-model config overlays (one YAML per model)
│       ├── aliyun_deepseek_v32.yaml
│       ├── aliyun_glm5.yaml
│       ├── aliyun_minimax_m25.yaml
│       ├── aliyun_qwen36_plus.yaml
│       ├── gemma4_27b.yaml
│       ├── llama32_3b.yaml
│       ├── llama3_8b.yaml
│       ├── qwen35_4b.yaml
│       └── qwen35_9b.yaml
├── data/
│   ├── instances/               # Pre-generated PDDL task instances (JSON)
│   └── plans/                   # Cached Fast Downward oracle plans (auto-created on first run)
├── results/                     # Experiment outputs — gitignored, created at runtime
│   └── <model-label>/
│       ├── results/results.jsonl   # Per-trial evaluation records
│       └── traces/traces.jsonl     # Per-step model call records
├── src/
│   ├── common/
│   │   ├── contracts.py         # Core dataclasses: PlanStep, PlanTrace, PlanningTaskInstance, etc.
│   │   ├── metrics.py           # plan_accuracy, ordering_violations (canonical implementations)
│   │   ├── logging.py           # Structured logging setup
│   │   └── reproducibility.py  # Global seed management
│   ├── generator/
│   │   ├── generate.py          # LinearLogisticsTaskGenerator — PDDL instance synthesis
│   │   └── nl_instruction.py   # Natural-language task description formatter
│   ├── runner/
│   │   ├── orchestrator.py      # ProposalExperimentRunner — stepwise rollout loop & batch runner
│   │   ├── memory.py            # FullContextPolicy, RecentWindowPolicy, SummarizationPolicy
│   │   ├── inference.py         # PromptBuilder, StrictActionParser
│   │   ├── planner.py           # FastDownwardPlannerBackend, SymbolicPlanExecutor, OracleTraceBuilder
│   │   ├── engine.py            # ModelBackend protocol, TransformersQwenBackend
│   │   ├── openai_backend.py    # OpenAICompatibleBackend (API and local llama.cpp serving)
│   │   ├── api_backend.py       # ClaudeAPIBackend (legacy)
│   │   ├── plan_cache.py        # Oracle plan caching and loading
│   │   └── config.py            # ProposalConfig, MemoryPolicyConfig, load_config
│   ├── eval/
│   │   └── check.py             # HardConstraintEvaluator — symbolic constraint checker
│   └── analysis/
│       ├── canonical.py         # CanonicalBundle — coverage-correct result loader
│       ├── python_figures.py    # Matplotlib figure generation for all paper figures
│       └── plot.py              # Entry point: build canonical tables + generate figures
├── tests/                       # pytest test suite
├── requirements.txt
└── README.md
```

### Module responsibilities at a glance

| Module | What it does |
|---|---|
| `src/generator/generate.py` | Generates synthetic PDDL logistics task instances at five difficulty tiers |
| `src/runner/orchestrator.py` | Runs one trial (stepwise rollout) or a full batch across all tasks × conditions × modes |
| `src/runner/memory.py` | Implements all five memory policies; applies them at each rollout step |
| `src/runner/planner.py` | Calls Fast Downward to produce oracle plans; validates action traces symbolically step by step |
| `src/eval/check.py` | Evaluates completed traces: goal satisfaction, precondition violations, ordering violations |
| `src/analysis/canonical.py` | Loads JSONL results with correct coverage accounting (missing cells ≠ failures) |
| `src/analysis/python_figures.py` | Produces all paper figures from the canonical result bundle |

---

## Installation

### 1. Python environment

Python 3.10 or later is required.

```bash
# Create and activate a conda environment (recommended)
conda create -n memory-exp python=3.10
conda activate memory-exp

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Fast Downward (oracle planner)

Fast Downward is required to generate oracle plans. It is **not** installed via pip.

```bash
git clone https://github.com/aibasel/downward.git ~/downward
cd ~/downward && python build.py
```

Verify the installation:
```bash
~/downward/fast-downward.py --version
```

### 3. Model serving

**API-accessed models** (GLM-5, Qwen3.6-plus, DeepSeek-v3.2, Minimax-M2.5) require an Alibaba Cloud API key:
```bash
export ALIYUN_API_KEY="your-key-here"
```

**Locally-served models** (Gemma4-27b, LLaMA variants, Qwen3.5 variants) were run on Google Colab A100 instances using `llama.cpp` GGUF quantizations behind an OpenAI-compatible endpoint (default `http://localhost:8000`). Uncomment the relevant lines in `requirements.txt` if you plan to use `transformers` directly.

---

## Running the Experiment End to End

### Step 1 — Generate task instances

The `data/instances/` directory already contains the pre-generated instances used in the paper (seed 42). To regenerate or create new instances:

```bash
python -m src.generator.generate \
    --n-instances 5 \
    --domain logistics \
    --seed 42 \
    --output-dir data/
```

**Output:** One JSON file per instance in `data/instances/`, e.g. `logistics-easy-42-0.json`.

**Instance JSON format:**
```json
{
  "task_id": "logistics-easy-42-0",
  "domain_pddl": "...",
  "problem_pddl": "...",
  "goal_text": "Deliver pkg_1 to room_2.",
  "metadata": {
    "difficulty": "easy",
    "instance_index": 0,
    "action_schemas": { "move": [...], "pickup": [...], ... }
  }
}
```

### Step 2 — Run the experiment

```bash
python -m src.runner \
    --config configs/experiment.yaml \
    --model-config configs/models/aliyun_deepseek_v32.yaml
```

To run a single memory condition or prompt mode:
```bash
python -m src.runner \
    --config configs/experiment.yaml \
    --model-config configs/models/aliyun_glm5.yaml \
    --condition full_context \
    --mode direct_action
```

Results are written incrementally; interrupted runs resume automatically from the last completed `(task_id, condition, mode)` triple.

**Output — `results/<model>/results/results.jsonl`** (one JSON object per trial):
```json
{
  "task_id": "logistics-very_hard-42-0",
  "difficulty": "very_hard",
  "condition": "truncation_1024",
  "mode": "direct_action",
  "goal_satisfied": false,
  "valid_plan": false,
  "failure_reason": "goal_not_satisfied",
  "executed_steps": 120,
  "precondition_violations": 0,
  "parse_errors": 0,
  "plan_accuracy": 0.12,
  "exact_match": false,
  "loop_depth": 12,
  ...
}
```

**Output — `results/<model>/traces/traces.jsonl`** (one JSON object per model call):
```json
{
  "task_id": "logistics-very_hard-42-0",
  "step_index": 63,
  "attempt_type": "primary",
  "memory_policy": "truncation_1024",
  "finish_reason": "stop",
  "raw_output": "{\"action_name\": \"move\", ...}",
  "parse_error": null,
  "symbolic_error": null,
  "missing_preconditions": [],
  "replay_result": { "goal_satisfied": false, ... },
  ...
}
```

### Step 3 — Evaluate

The symbolic evaluator re-runs constraint checking over completed traces and produces an `EvaluationBreakdown` for each trial. In normal operation this runs automatically inside the rollout. To re-evaluate from saved traces:

```bash
python -m src.eval \
    --results results/ \
    --reference data/plans/
```

### Step 4 — Generate figures and tables

```bash
python -m src.analysis \
    --results results/
```

This produces all figures from the paper as PNG files in `results/figures/`. The canonical result loader correctly accounts for coverage: cells that were never attempted are marked `not_attempted` rather than being counted as failures.

---

## Configuration Reference

`configs/experiment.yaml` controls the full experimental grid:

```yaml
experiment:
  seed: 42                          # Global random seed (all generation is deterministic)
  prompt_modes: [direct_action, cot]

dataset:
  instances_per_level: 1            # One instance per difficulty tier
  difficulty_levels: [easy, medium, hard, very_hard, extreme]
  include_distractors: true         # 10 irrelevant incident records prepended to each prompt
  max_plan_steps: 60                # Max oracle plan steps; rollout cap = 2 × this value

memory_policies:
  - {name: full_context,       type: full}
  - {name: truncation_1024,    type: truncation,    max_context_tokens: 1024}
  - {name: truncation_2048,    type: truncation,    max_context_tokens: 2048}
  - {name: summarization_1024, type: summarization, max_context_tokens: 1024, recent_window_tokens: 512}
  - {name: summarization_2048, type: summarization, max_context_tokens: 2048, recent_window_tokens: 768}

evaluation:
  enable_parse_recovery: true       # One CoT→direct_action recovery attempt on parse fail
  enable_symbolic_repair: true      # One symbolic repair attempt on precondition violation
  max_symbolic_repair_attempts: 1
  max_rollout_steps_multiplier: 2   # Rollout cap = 2 × oracle plan length
```

Model-specific settings (API endpoint, token budget, temperature) are set in `configs/models/*.yaml` and overlaid at runtime via `--model-config`.

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite covers the generator, symbolic evaluator, memory policies, action parser, plan cache, and orchestrator without requiring a live model endpoint (all model calls are mocked).

---

## Reproducibility

All results in the paper are fully reproducible from the configurations and pre-generated instances in this repository:

- Random seed: `42` throughout (task generation, prompt distractor sampling)
- Temperature: `0.0` for all models (deterministic outputs)
- Oracle plans: cached in `data/plans/` after first generation; subsequent runs load from cache
- Memory policy parameters: fully specified in `configs/experiment.yaml`

The `data/instances/` directory contains the exact task instances used in the paper. Running the experiment with the same model config will reproduce the same trial sequence.
