import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_cli_creates_metrics_csv(tmp_path):
    output = tmp_path / "nested" / "metrics"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marl_ev_mcs",
            "evaluate",
            "--smoke",
            "--episodes",
            "1",
            "--steps",
            "1",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "metrics.csv").is_file()


def test_train_cli_saves_checkpoints(tmp_path):
    output = tmp_path / "metrics"
    models = tmp_path / "trained-models"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marl_ev_mcs",
            "train",
            "--smoke",
            "--episodes",
            "1",
            "--steps",
            "1",
            "--output",
            str(output),
            "--model-output",
            str(models),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (models / "params1_20_2_mcs_net.pkl").is_file()
    assert (models / "params1_20_2_iev_net.pkl").is_file()


def test_benchmark_cli_writes_auditable_result_files(tmp_path):
    output = tmp_path / "benchmark"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marl_ev_mcs",
            "benchmark",
            "--smoke",
            "--policies",
            "mlp",
            "--train-seeds",
            "7",
            "--eval-seeds",
            "8",
            "--train-episodes",
            "1",
            "--train-steps",
            "1",
            "--eval-episodes",
            "1",
            "--eval-steps",
            "1",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "episode_metrics.csv").is_file()
    assert (output / "per_seed_metrics.csv").is_file()
    assert (output / "summary_metrics.csv").is_file()
    assert (output / "protocol.json").is_file()


def test_cli_is_available_outside_project_root(tmp_path):
    output = tmp_path / "external-metrics"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "marl_ev_mcs",
            "evaluate",
            "--smoke",
            "--episodes",
            "1",
            "--steps",
            "1",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output / "metrics.csv").is_file()


def test_gnn_cli_train_then_evaluate(tmp_path):
    models = tmp_path / "models"
    train = subprocess.run(
        [
            sys.executable,
            "-m",
            "marl_ev_mcs",
            "train",
            "--policy",
            "gnn",
            "--smoke",
            "--episodes",
            "1",
            "--steps",
            "1",
            "--model-output",
            str(models),
            "--output",
            str(tmp_path / "train-metrics"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert train.returncode == 0, train.stderr

    evaluate = subprocess.run(
        [
            sys.executable,
            "-m",
            "marl_ev_mcs",
            "evaluate",
            "--policy",
            "gnn",
            "--smoke",
            "--episodes",
            "1",
            "--steps",
            "1",
            "--model-dir",
            str(models),
            "--mcs-checkpoint",
            "params1_20_2_gnn_mcs_net.pkl",
            "--iev-checkpoint",
            "params1_20_2_gnn_iev_net.pkl",
            "--output",
            str(tmp_path / "eval-metrics"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert evaluate.returncode == 0, evaluate.stderr
    assert (tmp_path / "eval-metrics" / "metrics.csv").is_file()
