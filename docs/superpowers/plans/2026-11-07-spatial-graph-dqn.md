# Spatial Graph DQN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional spatial graph-attention DQN policy for MCS/IEV candidate-action selection.

**Architecture:** A pure-PyTorch graph Q network treats each candidate road intersection as a node, connects spatial k-nearest neighbors, and produces one Q value per node through masked attention. Existing MLP behavior and checkpoints remain the default path.

**Tech Stack:** Python 3.12, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-spatial-graph-dqn-design.md`

## Global Constraints

- Preserve the existing MLP policy as the default and retain compatibility with bundled MLP checkpoints.
- Do not require torch-geometric or another external GNN framework.
- Keep graph checkpoint names separate from MLP names.

## Review Focus

- One candidate node and an empty candidate set must not cause attention-mask shape errors; Task 1 tests this.
- GNN checkpoint loading must never accept MLP weights silently; Task 2 tests distinct names.
- Existing `--policy mlp` invocation must remain unchanged; Task 3 tests the default CLI path.
- GNN training must persist, then evaluation must reload, a compatible checkpoint pair; Task 3 tests this.

---

### Task 1: Spatial graph Q network

**Files:**
- Modify: `DQN/net.py`
- Test: `tests/test_graph_net.py`

**Interfaces:**
- Produces: `SpatialGraphQNet(feature_dim=2, hidden_dim=32, neighbors=4)` with `forward(features, positions) -> Tensor[N, 1]`.

- [ ] Write tests for output shape, one-node input, and Q-value sensitivity to a connected neighbor.
- [ ] Run the tests and confirm `SpatialGraphQNet` import fails.
- [ ] Implement deterministic k-neighbor masking and masked self-attention in `DQN/net.py`.
- [ ] Run the graph tests and confirm pass.

### Task 2: Policy-selectable DQN integration

**Files:**
- Modify: `DQN/dqn.py`
- Test: `tests/test_graph_runner.py`

**Interfaces:**
- Produces: `DQN(policy="gnn", ...)`; `run_experiment(..., policy="gnn")`; `checkpoint_names(conf, policy="gnn")`.

- [ ] Write tests that graph checkpoint names differ from MLP names and a graph smoke train/save/reload cycle works.
- [ ] Run the tests and confirm graph-policy parameters are rejected.
- [ ] Add policy factory, position-aware selection/learning, and policy-specific checkpoint names.
- [ ] Run the graph runner tests and confirm pass.

### Task 3: CLI and documentation

**Files:**
- Modify: `marl_ev_mcs/__main__.py`
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Produces: `python -m marl_ev_mcs train --policy gnn ...` and equivalent evaluation command.

- [ ] Write a failing CLI round-trip test for GNN smoke train then evaluate.
- [ ] Run the CLI test and confirm `--policy` is unrecognized.
- [ ] Add `--policy {mlp,gnn}` and document GNN-specific checkpoint names and limits.
- [ ] Run the full suite and a manual GNN smoke round trip.
