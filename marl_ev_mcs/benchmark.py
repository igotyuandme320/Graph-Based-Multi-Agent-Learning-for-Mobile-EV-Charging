"""Fair, reproducible multi-seed experiment orchestration."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import pandas as pd

from DQN.dqn import SUPPORTED_POLICIES, checkpoint_names, run_experiment


_RUN_COLUMNS = ["policy", "train_seed", "eval_seed"]
_NON_METRIC_COLUMNS = {"episode", *_RUN_COLUMNS}
_MAX_NUMPY_SEED = 2**32 - 1


def parse_seed_list(value: str) -> list[int]:
    """Parse a comma-separated, non-empty, unique sequence of integer seeds."""
    if not value or not value.strip():
        raise ValueError("Seed list must not be blank")
    tokens = value.split(",")
    if any(not token.strip() for token in tokens):
        raise ValueError("Seed list contains a blank token")
    try:
        seeds = [int(token.strip()) for token in tokens]
    except ValueError as error:
        raise ValueError("Seed list must contain integers") from error
    if any(seed < 0 or seed > _MAX_NUMPY_SEED for seed in seeds):
        raise ValueError(f"Seed values must be between 0 and {_MAX_NUMPY_SEED}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed values must be unique")
    return seeds


def summarize_episode_metrics(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate episodes to seed runs, then seed runs to policy mean/std rows."""
    missing = set(_RUN_COLUMNS) - set(episodes.columns)
    if missing:
        raise ValueError(f"Episode metrics missing run columns: {sorted(missing)}")
    metric_columns = [
        column for column in episodes.select_dtypes(include="number").columns
        if column not in _NON_METRIC_COLUMNS
    ]
    if not metric_columns:
        raise ValueError("Episode metrics contain no numeric metric columns")
    per_seed = (
        episodes.groupby(_RUN_COLUMNS, as_index=False, sort=True)[metric_columns]
        .mean()
    )
    records = []
    for policy, rows in per_seed.groupby("policy", sort=True):
        record: dict[str, float | int | str] = {"policy": policy, "n_runs": len(rows)}
        for metric in metric_columns:
            record[f"{metric}_n"] = int(rows[metric].count())
            record[f"{metric}_mean"] = float(rows[metric].mean())
            record[f"{metric}_std"] = float(rows[metric].std(ddof=1)) if len(rows) > 1 else 0.0
        records.append(record)
    return per_seed, pd.DataFrame(records)


def _validate_seed_sequence(seeds: Iterable[int], label: str) -> list[int]:
    values = list(seeds)
    if not values:
        raise ValueError(f"{label} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} values must be unique")
    if any(not isinstance(seed, int) or seed < 0 or seed > _MAX_NUMPY_SEED for seed in values):
        raise ValueError(f"{label} values must be integers between 0 and {_MAX_NUMPY_SEED}")
    return values


def run_benchmark(
    config: dict,
    *,
    policies: Iterable[str],
    train_seeds: Iterable[int],
    eval_seeds: Iterable[int],
    train_episodes: int,
    train_steps: int,
    eval_episodes: int,
    eval_steps: int,
    output_dir: Path,
) -> dict[str, Path]:
    """Train each policy/seed pair and evaluate it over a shared seed grid."""
    policy_list = list(policies)
    if not policy_list:
        raise ValueError("At least one policy is required")
    if len(set(policy_list)) != len(policy_list):
        raise ValueError("Policy values must be unique")
    unsupported = set(policy_list) - SUPPORTED_POLICIES
    if unsupported:
        raise ValueError(f"Unsupported policies: {sorted(unsupported)}")
    train_seed_list = _validate_seed_sequence(train_seeds, "train_seeds")
    eval_seed_list = _validate_seed_sequence(eval_seeds, "eval_seeds")
    for name, value in {
        "train_episodes": train_episodes,
        "train_steps": train_steps,
        "eval_episodes": eval_episodes,
        "eval_steps": eval_steps,
    }.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows: list[dict] = []
    for policy in policy_list:
        mcs_name, iev_name = checkpoint_names(config, policy=policy)
        for train_seed in train_seed_list:
            model_dir = output_dir / policy / f"train-seed-{train_seed}" / "models"
            run_experiment(
                deepcopy(config),
                episodes=train_episodes,
                steps=train_steps,
                evaluate=False,
                model_output_dir=model_dir,
                seed=train_seed,
                policy=policy,
            )
            for eval_seed in eval_seed_list:
                metrics = run_experiment(
                    deepcopy(config),
                    episodes=eval_episodes,
                    steps=eval_steps,
                    evaluate=True,
                    model_dir=model_dir,
                    mcs_model_name=mcs_name,
                    iev_model_name=iev_name,
                    seed=eval_seed,
                    policy=policy,
                )
                for metric in metrics:
                    episode_rows.append({
                        "policy": policy,
                        "train_seed": train_seed,
                        "eval_seed": eval_seed,
                        **metric,
                    })

    episode_metrics = pd.DataFrame(episode_rows)
    per_seed_metrics, summary_metrics = summarize_episode_metrics(episode_metrics)
    output_paths = {
        "episode_metrics": output_dir / "episode_metrics.csv",
        "per_seed_metrics": output_dir / "per_seed_metrics.csv",
        "summary_metrics": output_dir / "summary_metrics.csv",
        "protocol": output_dir / "protocol.json",
    }
    episode_metrics.to_csv(output_paths["episode_metrics"], index=False)
    per_seed_metrics.to_csv(output_paths["per_seed_metrics"], index=False)
    summary_metrics.to_csv(output_paths["summary_metrics"], index=False)
    output_paths["protocol"].write_text(json.dumps({
        "policies": policy_list,
        "train_seeds": train_seed_list,
        "eval_seeds": eval_seed_list,
        "train_episodes": train_episodes,
        "train_steps": train_steps,
        "eval_episodes": eval_episodes,
        "eval_steps": eval_steps,
        "config": config,
    }, indent=2, sort_keys=True), encoding="utf-8")
    return output_paths
