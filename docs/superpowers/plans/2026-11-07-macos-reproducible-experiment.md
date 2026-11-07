# macOS 可复现实验 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing MCS/IEV DQN simulation installable and runnable on macOS through documented, reproducible commands.

**Architecture:** Add a small `marl_ev_mcs` package that owns paths, presets and CLI orchestration while preserving the existing environment/DQN implementation. Refactor only the existing points that rely on Windows paths, removed timing APIs and implicit working directories; write metrics into a caller-selected output directory.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pandas, python-geohash, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-macos-reproducible-experiment-design.md`

## Global Constraints

- Preserve the simulation/DQN research logic and the included data and pretrained models.
- Use `pathlib` and paths relative to the repository root; never use Windows drive paths.
- Default macOS execution must work without CUDA.
- Smoke evaluation must write new artifacts only and must not overwrite bundled models or historical data.

## Review Focus

- A command launched outside the repository root still locates data, models and output correctly; Task 1 test owns this.
- A CPU-only PyTorch installation loads bundled checkpoints; Task 2 test owns this.
- Empty agent/action sets do not turn smoke metrics into division-by-zero crashes; Task 2 test owns this.
- Training mode must not load evaluation checkpoints; Task 2 test owns this.
- Output parent directories absent at startup are created before metrics are written; Task 3 test owns this.

---

### Task 1: Project paths and repeatable configuration

**Files:**
- Create: `marl_ev_mcs/__init__.py`
- Create: `marl_ev_mcs/paths.py`
- Create: `marl_ev_mcs/config.py`
- Modify: `env/multiagent/config.py`
- Modify: `env/multiagent/world.py`
- Test: `tests/test_paths.py`

**Interfaces:**
- Produces: `project_root() -> Path`, `data_path(*parts: str) -> Path`, `model_path(name: str) -> Path`, `default_config(preset: str) -> dict`.
- Consumes: the repository's `data/hash_point_remove.csv` and `data/track/20140805.csv`.

- [ ] **Step 1: Write failing path/config tests**

```python
def test_project_paths_resolve_bundled_data():
    assert data_path("hash_point_remove.csv").is_file()
    assert data_path("track", "20140805.csv").is_file()

def test_smoke_preset_is_smaller_than_default():
    assert default_config("smoke")["NUM_EV"] < default_config("default")["NUM_EV"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paths.py -v`
Expected: FAIL because `marl_ev_mcs` does not exist.

- [ ] **Step 3: Implement paths and presets, then replace environment Windows paths**

`paths.py` exposes the named helpers using the directory above its package as root. `config.py` returns a new config dictionary for `default` and `smoke`. `env/multiagent/config.py` uses `data_path` values, and `World.set_data` accepts its selected track day from config (default `5`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_paths.py -v`
Expected: PASS.

### Task 2: Parameterized DQN runner and safe evaluation

**Files:**
- Modify: `DQN/dqn.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `run_experiment(conf: dict, *, episodes: int, steps: int, evaluate: bool, model_dir: Path | None, seed: int | None) -> list[dict]`.
- Produces: a list of per-episode metrics with finite numeric values and an injectable checkpoint directory.

- [ ] **Step 1: Write failing runner tests**

```python
def test_evaluation_uses_cpu_safe_checkpoint_loading(tmp_path):
    # monkeypatch torch.load and run an injected one-step evaluator
    # assert map_location was passed and a metrics mapping was returned

def test_metric_helper_handles_no_charged_vehicles():
    assert metrics_for_empty_charge_population()["charged_proportion"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runner.py -v`
Expected: FAIL because the runner/metric helper does not exist.

- [ ] **Step 3: Implement a parameterized runner without changing reward/action logic**

Replace `time.clock()` with `time.perf_counter()`, add `map_location=device` when loading checkpoints, derive model/results locations from arguments, avoid division by zero in summary metrics, and return metrics. Preserve the legacy `train(conf)` API as a wrapper.

- [ ] **Step 4: Run runner tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -v`
Expected: PASS.

### Task 3: CLI, dependencies, documentation and end-to-end smoke evaluation

**Files:**
- Create: `marl_ev_mcs/__main__.py`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `.gitignore`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `python -m marl_ev_mcs evaluate --smoke --episodes 1 --steps 2 --output <dir>`.
- Produces: `<dir>/metrics.csv` and a zero exit code.

- [ ] **Step 1: Write a failing CLI test**

```python
def test_smoke_cli_creates_metrics_csv(tmp_path):
    result = subprocess.run([...], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "metrics.csv").is_file()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL because the package has no module CLI.

- [ ] **Step 3: Implement the CLI and installation material**

`__main__.py` parses `evaluate`/`train`, preset, episode/step, seed, model directory and output directory; it writes a CSV from runner results. Pin only minimum compatible dependencies in `requirements.txt`; README gives macOS virtual-environment install, smoke, normal evaluation and training commands. Ignore virtual environments, caches and generated result directories.

- [ ] **Step 4: Run CLI test and the documented smoke command**

Run: `python -m pytest tests/test_cli.py -v && python -m marl_ev_mcs evaluate --smoke --episodes 1 --steps 2 --output /tmp/marl-ev-mcs-smoke`
Expected: test PASS and `/tmp/marl-ev-mcs-smoke/metrics.csv` exists.

- [ ] **Step 5: Run full verification**

Run: `python -m pytest -q`
Expected: all tests pass.
