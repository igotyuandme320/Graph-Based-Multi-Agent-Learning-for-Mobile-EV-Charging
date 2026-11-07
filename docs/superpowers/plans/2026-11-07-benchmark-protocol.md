# Reproducible Benchmark Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a multi-seed, equal-budget benchmark command for MLP/GNN comparison with auditable CSV summaries.

**Architecture:** A new benchmark module orchestrates existing `run_experiment` calls without changing learning code. It writes episode-level results, aggregates first by `(policy, train_seed, eval_seed)`, then reports strategy-level mean/std; CLI only parses and validates arguments.

**Tech Stack:** Python 3.12, pandas, pytest, existing PyTorch simulation runner.

**Spec:** `docs/superpowers/specs/2026-11-07-benchmark-protocol-design.md`

## Global Constraints

- Train and evaluate all policies with exactly the caller-supplied shared budgets and seed lists.
- A policy's evaluation must load only that policy and training seed's checkpoint pair.
- Strategy-level standard deviation uses per-seed-run rows and is 0 for exactly one run.
- No long experiment or performance conclusion belongs in this implementation task.

## Review Focus

- A duplicate or blank seed token must fail rather than silently changing run count; Task 1 test owns this.
- MLP/GNN checkpoints must be isolated by policy and training seed; Task 2 test owns this.
- Evaluation must use each training model for every evaluation seed, not retrain inside evaluation; Task 2 test owns this.
- Summary must aggregate episode rows before aggregating seed runs; Task 2 test owns this.
- A one-run benchmark must emit numeric zero standard deviations rather than NaN; Task 2 test owns this.

---

### Task 1: Benchmark parsing and aggregation

**Files:**
- Create: `marl_ev_mcs/benchmark.py`
- Create: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `parse_seed_list(value: str) -> list[int]`, `summarize_episode_metrics(frame: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]`.

- [x] **Step 1: Write failing tests** for valid/invalid seed lists and two-stage mean/std aggregation.
- [x] **Step 2: Run `python -m pytest -q tests/test_benchmark.py`**; observed import failure before the module existed.
- [x] **Step 3: Implement parsing and aggregation helpers** with explicit metadata columns and numeric-only metric selection.
- [x] **Step 4: Run the focused test**; passed.

### Task 2: Orchestration and CLI

**Files:**
- Modify: `marl_ev_mcs/benchmark.py`
- Modify: `marl_ev_mcs/__main__.py`
- Modify: `tests/test_benchmark.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `run_experiment`, `checkpoint_names`, parsed policies/seeds/budgets.
- Produces: `run_benchmark(...) -> dict[str, pathlib.Path]` and the `benchmark` CLI mode.

- [x] **Step 1: Write failing tests** mocking `run_experiment` to verify training/evaluation call matrix, checkpoint isolation, CSV outputs, and CLI output creation.
- [x] **Step 2: Run focused tests**; observed failure before the orchestrator/mode existed.
- [x] **Step 3: Implement `run_benchmark`** and integrate CLI validation/argument routing.
- [x] **Step 4: Run focused tests**; passed.

### Task 3: Documentation and verification

**Files:**
- Modify: `README.md`
- Modify: `docs/application-project-profile.md`
- Test: `tests/`

- [x] **Step 1: Document the exact benchmark command and explain that its summaries are required before comparative claims.**
- [x] **Step 2: Run `python -m pytest -q`;** 45 passed.
- [x] **Step 3: Run a two-policy, one-seed smoke benchmark** and inspect all three CSVs plus `protocol.json`.
