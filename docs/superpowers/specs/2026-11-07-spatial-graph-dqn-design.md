# 空间图 DQN 设计

## 意图

为现有 MCS/IEV DQN 增加图学习策略，让候选路口不仅依据自身两维局部特征评分，还能聚合空间邻近候选路口的信息。

## 目标

- 新增纯 PyTorch 的空间图注意力 Q 网络，无需 `torch-geometric`。
- 候选路口为图节点；节点特征为现有局部观测，边由候选路口 k 近邻关系构成。
- MCS 和 IEV 都支持 `gnn` 策略；训练、保存、重载和评估可从 CLI 使用。
- 保持 `mlp` 为默认策略，兼容现有预训练 MLP checkpoint。

## 非目标

- 本次不改变环境、奖励函数或训练算法为 MAPPO/QMIX。
- 不声称新的 GNN 未经完整训练就优于已有策略。
- 不引入异构图框架或 GPU 专属依赖。

## 方案比较

1. 使用 PyTorch Geometric 异构图：表达力最强，但 macOS 安装和 PyTorch 版本耦合较高。
2. 纯 PyTorch 空间注意力图网络（采用）：用候选路口 k 近邻边构造稀疏注意力，直接复用已有候选特征和位置。
3. 仅向 MLP 加经纬度：改动最小，但不进行节点间消息传递，不是图学习。

采用方案 2：它提供真实的空间消息传递，且能在现有 Conda 环境稳定运行。

## 设计

- `SpatialGraphQNet` 接收 `features: [N, 2]` 和 `positions: [N, 2]`，标准化输入后构造每个节点的 k 近邻掩码（含自环）。
- 单层多头空间注意力在掩码边上传递消息，每个候选节点输出一个 Q 值。
- `DQN` 依据 `policy="mlp" | "gnn"` 创建网络；图策略的动作选择与学习同时传入候选位置。
- 图策略 checkpoint 名含 `_gnn_`，避免覆盖或误加载旧 MLP checkpoint。
- CLI 新增 `--policy`；GNN 初始只支持训练或加载对应 GNN checkpoint。

## 验收

- 图网络输出与候选节点数一致，并且改变相邻节点信息会影响节点 Q 值。
- `train --policy gnn --smoke` 写出 GNN checkpoint；随后 `evaluate --policy gnn` 可重载并完成 smoke 仿真。
- 所有现有 MLP 测试和运行命令继续通过。
