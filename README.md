# MARL EV MCS experiment

这是一个电动汽车（IEV）与移动充电站（MCS）协同调度的多智能体 DQN 仿真项目。已随项目提供路网、轨迹数据和默认评估模型。

## macOS（Conda）安装

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate marl-ev-mcs
python -m pip install -e .
```

如果环境已经由本项目创建，只需后两行。项目默认以 CPU 运行，因此不要求 CUDA；Apple Silicon 也可直接使用该环境。可编辑安装后，可以从任意目录运行 `python -m marl_ev_mcs`。

## 先验证安装

下面的命令使用很小的车辆/MCS 数量、1 个回合和 2 个仿真步，不会覆盖内置模型或数据：

```bash
python -m marl_ev_mcs evaluate --smoke --episodes 1 --steps 2 --output results/smoke
```

成功后会生成 `results/smoke/metrics.csv`。可用以下命令检查项目测试：

```bash
python -m pytest -q
```

## 正式评估与训练

默认评估使用附带的 `params1_500_50` 模型和原始的 500 辆 EV / 50 个 MCS 配置：

```bash
python -m marl_ev_mcs evaluate --episodes 1 --steps 100 --output results/evaluation
```

训练不会读取预训练模型；初次运行建议先缩短回合和步数确认流程：

```bash
python -m marl_ev_mcs train --smoke --episodes 2 --steps 10 --output results/train-smoke
```

可通过 `--seed` 固定随机数种子，通过 `--model-dir`、`--mcs-checkpoint` 和 `--iev-checkpoint` 选择一对评估模型。训练时，`--model-output` 指定新模型保存位置；未指定则保存到本次 `--output` 目录下的 `models/`。数据、模型和结果路径不会依赖当前工作目录以外的隐式相对路径。

## 空间图 DQN（GNN）

默认策略为 `mlp`，可加载项目附带的旧模型。`gnn` 策略把候选路口作为图节点，以空间 k 近邻关系做注意力消息传递；它不依赖 `torch-geometric`，但需要先训练自己的 GNN checkpoint，不能直接加载旧 MLP 模型。

先进行一个小规模训练：

```bash
python -m marl_ev_mcs train --policy gnn --smoke --episodes 10 --steps 100 --output results/gnn-train
```

再加载刚训练的模型评估：

```bash
python -m marl_ev_mcs evaluate --policy gnn --smoke --episodes 5 --steps 100 \
  --model-dir results/gnn-train/models \
  --mcs-checkpoint params1_20_2_gnn_mcs_net.pkl \
  --iev-checkpoint params1_20_2_gnn_iev_net.pkl \
  --output results/gnn-evaluation
```

`gnn` 是新的网络结构，不代表未经充分训练就一定优于原 MLP；应在相同随机种子和相同回合数下与 `mlp`、随机策略和启发式基线对比。

## 训练机制

训练模式现在使用经验回放（Replay Buffer）：每个有效交互都会保存，缓存积累后随机抽取小批经验更新网络；探索率会从 0.9 线性衰减到 0.05。默认值为容量 5000、预热 32 条经验、每类智能体每次采样 16 条。评估模式不探索、不训练，只执行当前策略。

这些参数位于 `marl_ev_mcs/config.py`，适合先保持默认值；做正式对比时应固定训练 seed，并使用多个评估 seed 汇总均值和标准差。

对于 MCS，一次调度后可能经历自动匹配、移动和充电的多步任务期。训练会将该动作暂存到 MCS 再次可决策或回合结束，再把服务完成收益加回原动作；每笔收益按其发生时刻以 `0.9 ^ 已过步数` 折扣，bootstrap 项使用 `0.9 ^ 任务持续步数`。这避免了把完成充电的收益错误奖励给服务结束后的下一次巡航动作，也不会在回合边界 bootstrap 到下一回合。现有局部奖励仍包含跨 agent 归一化；奖励标度的影响需要通过消融实验验证。

## 公平比较 MLP 与 GNN

不要用单次训练结果判断策略优劣。`benchmark` 会让每种策略使用相同训练 budget、训练 seed 和评估 seed，并把每个训练 seed 的模型保存在独立目录。先用小型验证命令检查流程：

```bash
python -m marl_ev_mcs benchmark --smoke --policies mlp gnn \
  --train-seeds 7 --eval-seeds 101 \
  --train-episodes 1 --train-steps 5 --eval-episodes 1 --eval-steps 5 \
  --output results/benchmark-smoke
```

正式对比可提高预算并使用多个种子：

```bash
python -m marl_ev_mcs benchmark --policies mlp gnn \
  --train-seeds 11,22,33 --eval-seeds 101,202,303 \
  --train-episodes 60 --train-steps 100 --eval-episodes 5 --eval-steps 100 \
  --output results/benchmark
```

结果目录中的 `episode_metrics.csv` 是逐评估回合原始结果；`per_seed_metrics.csv` 先对每个 `(policy, train_seed, eval_seed)` 平均；`summary_metrics.csv` 再以 seed-run 为样本报告每种策略的均值、标准差和 `n_runs`。`protocol.json` 保存完整的配置、预算和种子，因此只有基于这一协议的结果才适合用于方法对比。

## 如何解读评估指标

`metrics.csv` 会把一次请求的生命周期分开报告：

- `assigned_proportion`：发起充电请求的车辆中，已分配到 MCS 的比例。
- `completed_proportion`：发起请求后，实际完成充电的比例；`charged_proportion` 为这一字段保留的兼容别名。
- `failed_proportion`：等待超时或低电量触发失败处理的比例。
- `unserved_request_count`：尚未分配、也未失败的请求数；`in_progress_request_count` 是已分配但回合结束时尚未完成的请求数。
- `mean_wait_time_completed`、`mean_wait_time_failed`：分别只在已完成和失败请求上计算。`mean_wait_time` 是已完成请求的兼容别名。

因此不能把 `assigned_proportion` 当作服务成功率；进行模型对比时，应优先报告完成率、失败率、未完成请求数、等待时间和成本/利润，并使用相同训练预算与多随机种子。
