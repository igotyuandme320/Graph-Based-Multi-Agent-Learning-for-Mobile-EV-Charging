# 可复现实验基准协议设计

## 背景

现有 CLI 只能运行一组训练或评估。它不能保证 MLP 与 GNN 使用相同训练/评估种子，也不会自动保留逐次结果及均值、标准差。因此此前的探索性比较不能作为可重复的模型结论。

## 目标

新增 `benchmark` CLI 子命令，以相同训练预算和 seed 集合分别训练每种策略，并对每个训练模型使用相同评估 seed 集合评估。它应输出可审计的逐回合、按训练/评估 seed 汇总、按策略汇总 CSV。

## 非目标

- 不在本改动中训练长实验或宣称模型优劣。
- 不改变 DQN、环境、奖励或模型结构。
- 不把多个 seed 的结果错误视为独立的逐回合样本；策略汇总的样本单位为一个 `(policy, train_seed, eval_seed)` 运行。

## 命令与输出

命令形式：

```bash
python -m marl_ev_mcs benchmark --smoke \
  --policies mlp gnn --train-seeds 11,22,33 --eval-seeds 101,202,303 \
  --train-episodes 10 --train-steps 100 --eval-episodes 1 --eval-steps 100 \
  --output results/benchmark
```

每个策略/训练 seed 的 checkpoints 写入 `<output>/<policy>/train-seed-<seed>/models/`。输出目录还包含：

- `episode_metrics.csv`：每个评估回合一行，含 `policy`、`train_seed`、`eval_seed` 和环境指标。
- `per_seed_metrics.csv`：每个 `(policy, train_seed, eval_seed)` 一行，先在评估回合内取均值。
- `summary_metrics.csv`：每种策略一行；每个指标有 `_mean`、`_std` 和共享的 `n_runs`。标准差在单运行时写为 0。

## 约束与公平性

- `--policies`、训练 seed 和评估 seed 都必须非空；策略必须是项目支持的 `mlp` 或 `gnn`。
- 所有策略必须使用完全相同的训练/评估回合数、步数、配置 preset、训练 seed 与评估 seed。
- 训练不会读取旧模型；评估只读取同一 `(policy, train_seed)` 刚生成的 checkpoint 对。
- 输出记录命令参数到 `protocol.json`，使 CSV 可追溯。JSON 中不写机器相关的绝对路径以外的隐式状态。
