# SDD ledger — plan: docs/superpowers/plans/2026-11-07-benchmark-protocol.md

Pre-flight: Task 1 produces parsed seed lists and two-stage DataFrame summaries; Task 2 consumes both while orchestrating `run_experiment` and checkpoint names.

Ruling: This workspace has no Git metadata, so worktree/commit-based plan tooling is unavailable. Execute inline with TDD and retain command evidence here. Cost if wrong: no commit-range isolation is available.

Task 1: complete (tests: `/Users/igotuandme/anaconda3/envs/marl-ev-mcs/bin/python -m pytest -q tests/test_benchmark.py` → 2 passed). Seed parsing rejects blank/duplicate tokens; summaries first average evaluation episodes within each seed-run, then calculate policy mean/std.

Task 2: complete (tests: `/Users/igotuandme/anaconda3/envs/marl-ev-mcs/bin/python -m pytest -q tests/test_benchmark.py tests/test_cli.py::test_benchmark_cli_writes_auditable_result_files` → 4 passed). Training/evaluation call matrix, checkpoint isolation, CSV and protocol outputs are covered.

Review fix pass: independent review found policy action RNG could contaminate later evaluation episode scenarios and invalid NumPy seeds were not rejected up front. Evaluation now resets scenario RNGs per `(eval_seed, episode)` and seed parsing enforces the NumPy range; focused benchmark tests pass (4 passed).

Task 3: complete (tests: full suite → 45 passed; two-policy smoke benchmark → 4 episode rows, 2 seed-run rows, all CSVs and protocol written). Independent re-review reproduced matched MLP/GNN scenarios across three evaluation episodes and found no remaining Critical/Important issue in scope.
