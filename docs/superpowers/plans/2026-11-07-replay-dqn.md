# Replay DQN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace sparse on-policy updates with replay sampling and epsilon decay for MLP and GNN training.

**Architecture:** Store immutable per-agent transitions in bounded buffers. After warmup, sample variable-candidate transitions and update one at a time; inference remains unchanged.

**Tech Stack:** Python, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-replay-dqn-design.md`

### Task 1: Replay primitives and scheduling

- [ ] Add failing tests for bounded storage, immutable sampling, and epsilon decay.
- [ ] Implement `ReplayBuffer` and epsilon schedule in `DQN/dqn.py`.
- [ ] Verify primitive tests pass.

### Task 2: Runner integration

- [ ] Add a failing smoke training test that records replay-driven learning calls.
- [ ] Store valid MCS/IEV transitions and train sampled batches after warmup.
- [ ] Run full suite and short MLP/GNN training verification.
