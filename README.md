# MARL EV-MCS

Multi-agent reinforcement-learning experiments for coordinating electric vehicles (EVs) and mobile charging stations (MCSs). The project includes an MLP DQN baseline, a spatial graph-attention DQN variant, replay-based training, and reproducible multi-seed benchmarking.

## Reference

The original mobile-charging scheduling formulation is described in:

> Linfeng Liu, Zhuo Huang, and Jia Xu. “Multi-Agent Deep Reinforcement Learning Based Scheduling Approach for Mobile Charging in Internet of Electric Vehicles.” *IEEE Transactions on Mobile Computing*, 23(10), 10130–10145, 2024. [DOI: 10.1109/TMC.2024.3373410](https://doi.org/10.1109/TMC.2024.3373410)

## Installation

The project is tested with Conda on macOS.

```bash
conda env create -f environment.yml
conda activate marl-ev-mcs
python -m pip install -e .
```

Run the test suite after installation:

```bash
python -m pytest -q
```

## Quick start

Run a small evaluation using the bundled MLP checkpoint:

```bash
python -m marl_ev_mcs evaluate --smoke --episodes 1 --steps 20 --output results/smoke-eval
```

Train a new MLP policy:

```bash
python -m marl_ev_mcs train --smoke --episodes 10 --steps 100 --output results/mlp-train
```

Train and evaluate the graph policy. GNN checkpoints are separate from MLP checkpoints.

```bash
python -m marl_ev_mcs train --policy gnn --smoke --episodes 10 --steps 100 --output results/gnn-train

python -m marl_ev_mcs evaluate --policy gnn --smoke --episodes 5 --steps 100 \
  --model-dir results/gnn-train/models \
  --mcs-checkpoint params1_20_2_gnn_mcs_net.pkl \
  --iev-checkpoint params1_20_2_gnn_iev_net.pkl \
  --output results/gnn-eval
```

## Benchmarking

Use the benchmark command to train each policy with the same training seeds and evaluate every trained model with the same evaluation seeds.

```bash
python -m marl_ev_mcs benchmark --smoke --policies mlp gnn \
  --train-seeds 11,22,33 --eval-seeds 101,202,303 \
  --train-episodes 10 --train-steps 100 \
  --eval-episodes 3 --eval-steps 100 \
  --output results/benchmark
```

The command writes:

- `episode_metrics.csv`: one row per evaluation episode.
- `per_seed_metrics.csv`: episode averages for each `(policy, train_seed, eval_seed)` run.
- `summary_metrics.csv`: policy-level mean, standard deviation, and valid sample count for each metric.
- `protocol.json`: seeds, budgets, and configuration used for the run.

## Metrics

`completed_proportion` is the verified charging-completion rate. `assigned_proportion` only measures assignment to an MCS and should not be used as a service-success metric. Conditional metrics such as `mean_wait_time_completed` are empty when no request completes in a run; the benchmark summary records a metric-specific sample count.

## Layout

```text
DQN/             policy networks and training loop
env/             EV/MCS simulation environment
marl_ev_mcs/     CLI, configuration, paths, and benchmark runner
data/            traffic network and trajectory inputs
tests/           automated tests
```
