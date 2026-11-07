# MCS Service Credit Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assign charging-completion revenue to the MCS action that initiated its task period, using an SMDP-duration DQN transition.

**Architecture:** `World` exposes one-step completion events produced by `MCS.finish_charge()`. The runner defers a dispatch action's replay entry until the MCS is actionable again, accumulates its completion rewards, and trains with a duration-aware discount.

**Tech Stack:** Python 3.12, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-service-credit-assignment-design.md`

## Global Constraints

- Keep EV transitions and environment matching/movement rules unchanged.
- Use `SERVICE_REVENUE_REWARD_WEIGHT` for service-completion bonus and `GAMMA ** duration_steps` for deferred MCS bootstrapping.
- Preserve terminal handling for empty next-observation candidate sets.
- Do not present results from this change as a validated performance improvement.

## Review Focus

- A completed service event must be emitted once and have the revenue of the MCS that completed it; Task 1 test owns this.
- The service bonus must not be consumed by a subsequent unrelated MCS action; Task 1 test owns this.
- A dispatched MCS that returns to the decision set must create exactly one deferred replay transition; Task 2 test owns this.
- Multiple completion events before an MCS becomes actionable must accumulate on its one pending transition; Task 2 test owns this.
- A pending service transition at the episode limit must be stored as terminal; Task 2 test owns this.

---

### Task 1: Completion-event contract

**Files:**
- Modify: `env/multiagent/core.py`
- Modify: `env/multiagent/world.py`
- Modify: `tests/test_reward_shaping.py`

**Interfaces:**
- Produces: `MCS.finish_charge() -> float` and `World.completed_service_events: list[tuple[int, float]]`.

- [x] **Step 1: Write failing tests** for returned service revenue, event ownership, one-step event clearing, and absence of next-action revenue consumption.
- [x] **Step 2: Run the focused test** with `python -m pytest -q tests/test_reward_shaping.py`; observed failure because the event contract did not exist.
- [x] **Step 3: Implement the completion-event contract** in `MCS` and `World`; remove service-revenue consumption from current-step MCS rewards.
- [x] **Step 4: Run the focused test**; passed.

### Task 2: Deferred MCS SMDP replay transitions

**Files:**
- Modify: `DQN/dqn.py`
- Create/Modify: `tests/test_service_credit_assignment.py`

**Interfaces:**
- Consumes: `World.completed_service_events`.
- Produces: duration-aware `discount` on MCS replay transitions and a runner-owned pending-transition registry keyed by MCS ID.

- [x] **Step 1: Write failing unit tests** for a deferred transition's accumulated completion bonus, `GAMMA ** duration_steps` discount, and terminal flushing.
- [x] **Step 2: Run the focused test** with `python -m pytest -q tests/test_service_credit_assignment.py`; observed failure because deferred-transition helpers did not exist.
- [x] **Step 3: Add minimal pending-transition helpers and let `mcs_learn` accept a per-transition discount**, defaulting to `GAMMA` for ordinary DQN samples.
- [x] **Step 4: Integrate the helpers into `run_experiment`**, deferring only MCS actions that entered a task state, finalizing on re-entry to decision agents, and flushing at episode end.
- [x] **Step 5: Run focused tests**; passed.

### Task 3: Regression and documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/application-project-profile.md`
- Test: `tests/`

- [x] **Step 1: Document the SMDP delayed-credit semantics and remaining reward-normalization limitation.**
- [x] **Step 2: Run `python -m pytest -q`**; 39 passed.
- [x] **Step 3: Run a deterministic smoke training and evaluation** and inspect that `metrics.csv` is written with finite values.
