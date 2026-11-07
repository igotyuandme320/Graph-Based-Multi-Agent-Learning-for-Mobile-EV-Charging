# SDD ledger — plan: docs/superpowers/plans/2026-11-07-service-credit-assignment.md

Pre-flight: Task 1 produces `World.completed_service_events`; Task 2 consumes it. The event carries `(mcs_id, revenue)` and is reset per world update.

Ruling: This workspace is not a Git repository, so worktree creation, task scripts, commits, and commit-based review packages are unavailable. Execute inline with TDD and record command evidence here. Cost if wrong: changes cannot be isolated or reviewed by commit range.

Task 1: complete (tests: `/Users/igotuandme/anaconda3/envs/marl-ev-mcs/bin/python -m pytest -q tests/test_reward_shaping.py tests/test_charge_outcomes.py` → 5 passed). `MCS.finish_charge()` returns revenue; `World` emits one-step completion events and no longer applies completion reward to a later normal action.

Task 2: complete (tests: `/Users/igotuandme/anaconda3/envs/marl-ev-mcs/bin/python -m pytest -q tests/test_service_credit_assignment.py tests/test_replay.py tests/test_runner.py` → 16 passed). Pending MCS transitions accumulate owned completion revenue, use `GAMMA ** duration_steps`, and are terminally flushed at an episode boundary.

Task 3: complete (tests: `/Users/igotuandme/anaconda3/envs/marl-ev-mcs/bin/python -m pytest -q` → 39 passed; smoke train/evaluate → finite metrics). README and application copy document the SMDP attribution and remaining reward-normalization limitation.

Review fix pass: an independent review found final-step re-entry could bootstrap across the episode boundary and completion bonuses were not time-discounted. Added a scripted runner integration test, made re-entry terminal on the final step, and discount each event by `GAMMA ** elapsed_steps`; focused test suite passes (4 passed).

Final review: independent re-review confirmed both fixes. No remaining Critical/Important findings in the service-credit scope. Verification after the fix: full suite 40 passed; deterministic smoke train/save/reload/evaluate produced finite metrics.
