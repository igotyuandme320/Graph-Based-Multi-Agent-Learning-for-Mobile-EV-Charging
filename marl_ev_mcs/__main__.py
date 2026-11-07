"""Command-line entry point for repeatable experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from DQN.dqn import SUPPORTED_POLICIES, checkpoint_names, run_experiment
from marl_ev_mcs.benchmark import parse_seed_list, run_benchmark
from marl_ev_mcs.config import default_config
from marl_ev_mcs.paths import project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MCS/IEV DQN experiment.")
    parser.add_argument("mode", choices=("evaluate", "train", "benchmark"), help="Run one experiment or a multi-seed benchmark.")
    parser.add_argument("--smoke", action="store_true", help="Use the small, fast validation preset.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of simulation episodes.")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps per episode.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for repeatable runs.")
    parser.add_argument("--policy", choices=sorted(SUPPORTED_POLICIES), default="mlp", help="Q-network architecture.")
    parser.add_argument("--model-dir", type=Path, help="Directory containing evaluation checkpoints.")
    parser.add_argument(
        "--mcs-checkpoint", help="MCS checkpoint filename for evaluation."
    )
    parser.add_argument(
        "--iev-checkpoint", help="IEV checkpoint filename for evaluation."
    )
    parser.add_argument(
        "--model-output", type=Path, help="Directory where training writes its paired checkpoints."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "results",
        help="Directory for metrics.csv (created if needed).",
    )
    parser.add_argument("--policies", nargs="+", choices=sorted(SUPPORTED_POLICIES), default=["mlp", "gnn"], help="Policies to compare in benchmark mode.")
    parser.add_argument("--train-seeds", default="7,17,27", help="Comma-separated training seeds for benchmark mode.")
    parser.add_argument("--eval-seeds", default="101,202,303", help="Comma-separated evaluation seeds for benchmark mode.")
    parser.add_argument("--train-episodes", type=int, default=10, help="Training episodes per benchmark seed.")
    parser.add_argument("--train-steps", type=int, default=100, help="Training steps per benchmark episode.")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Evaluation episodes per benchmark seed.")
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation steps per benchmark episode.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1 or args.steps < 1:
        raise SystemExit("--episodes and --steps must both be at least 1")
    preset = "smoke" if args.smoke else "default"
    config = default_config(preset)
    if args.mode == "benchmark":
        try:
            outputs = run_benchmark(
                config,
                policies=args.policies,
                train_seeds=parse_seed_list(args.train_seeds),
                eval_seeds=parse_seed_list(args.eval_seeds),
                train_episodes=args.train_episodes,
                train_steps=args.train_steps,
                eval_episodes=args.eval_episodes,
                eval_steps=args.eval_steps,
                output_dir=args.output,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"Wrote benchmark results to {args.output}")
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0
    model_output = args.model_output or args.output / "models"
    results = run_experiment(
        config,
        episodes=args.episodes,
        steps=args.steps,
        evaluate=args.mode == "evaluate",
        model_dir=args.model_dir,
        mcs_model_name=args.mcs_checkpoint,
        iev_model_name=args.iev_checkpoint,
        model_output_dir=model_output if args.mode == "train" else None,
        seed=args.seed,
        policy=args.policy,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / "metrics.csv"
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"Wrote {len(results)} episode(s) to {output_file}")
    if args.mode == "train":
        mcs_name, iev_name = checkpoint_names(config, policy=args.policy)
        print(f"Wrote trained checkpoints to {model_output / mcs_name} and {model_output / iev_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
