# MCS 服务收益归因设计

## 背景

当前环境在 MCS 完成充电时把收益保存在该 MCS 的 `pending_service_revenue` 中；该收益会在 MCS 恢复空闲后的下一次普通动作获得。收益不会串给其他 MCS，也只消费一次，但它不属于最初让 MCS 接单的动作，因此不是严格的调度信用分配。

## 目标

把一个 MCS 从一次动作后自动接单、到它再次可作决策（或回合结束）的过程建模为一条半马尔可夫（SMDP）transition。服务完成收益应累加到发起该服务周期的原始动作上，目标网络的 bootstrap 项使用 `GAMMA ** duration_steps`。

## 非目标

- 不改变 EV 的一步 transition。
- 不改变自动匹配、移动和充电的环境规则。
- 不声称该改动已经完成正式消融或带来性能提升。
- 不在此改动中替换现有全局 reward normalization。

## 设计决定

- `MCS.finish_charge()` 返回本次服务收入；`World` 在每个环境步收集 `(mcs_id, revenue)` 完成事件。完成事件是一次性的，并在下一环境步开始时清空。
- 训练循环对“动作后进入任务状态”的 MCS 暂存原始 DQN transition，而不立即放入 replay buffer。暂存项记录初始归一化局部 reward、累计服务奖金、持续环境步数以及原始 observation/action。
- 对仍处于任务期的暂存项，每个环境步增加持续时间；同一 MCS 的每一笔完成收入按 `GAMMA ** elapsed_steps * SERVICE_REVENUE_REWARD_WEIGHT` 累加到该暂存项。
- 当该 MCS 再次出现在可决策 agents 中时，使用该时刻观察作为 next state，写入一条 transition，`discount = GAMMA ** duration_steps`。如果任务跨回合仍未回到可决策状态，以终止 transition 写入，避免经验丢失。
- `World.get_reward_n()` 不再领取 `pending_service_revenue`，所以服务奖金不会错误加到完成后的新动作。原有移动成本和局部奖励仍属于当前一步。

## 约束与解释

- 由于既有 MCS 局部 reward 在动作时已做跨 agent 归一化，暂存的基础 reward 与后续完成奖金的标度并非完全同质；这是现有 reward 设计遗留限制，正式结论必须通过 reward-normalization 消融验证。
- 当 MCS 完成后被环境自动连续派给另一 EV、尚未重新成为可决策 agent 时，所有完成收益继续归入同一暂存服务周期；这是对当前自动匹配语义最保守的归因。
- 已有 `pending_service_revenue` API 将移除/停止用于学习，测试改为验证 world completion event 与原始 transition 的绑定。
