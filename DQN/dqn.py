"""DQN policies and a reproducible runner for the MCS/IEV simulation."""

from __future__ import annotations

import random
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from DQN.net import IEVNet, MCSNet, SpatialGraphQNet
from env.multiagent.core import MCS, Vehicle
from env.multiagent.environment import MultiAgentEnv
from env.multiagent.world import World
from marl_ev_mcs.paths import model_path, project_root


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.005
GAMMA = 0.9
TARGET_REPLACE_ITER = 10
DEFAULT_MCS_MODEL = "params1_500_50_mcs_net.pkl"
DEFAULT_IEV_MODEL = "params1_500_50_iev_net.pkl"
DEFAULT_GNN_MCS_MODEL = "params1_500_50_gnn_mcs_net.pkl"
DEFAULT_GNN_IEV_MODEL = "params1_500_50_gnn_iev_net.pkl"
SUPPORTED_POLICIES = {"mlp", "gnn"}
_NUMPY_SEED_MODULUS = 2**32


class ReplayBuffer:
    """A bounded store of immutable, variable-candidate DQN transitions."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self._items = deque(maxlen=capacity)

    def append(self, transition: dict) -> None:
        self._items.append(deepcopy(transition))

    def sample(self, batch_size: int) -> list[dict]:
        if batch_size < 1:
            raise ValueError("Replay batch size must be positive")
        return deepcopy(random.sample(list(self._items), k=min(batch_size, len(self._items))))

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class PendingMCSServiceTransition:
    """An MCS decision held until its automatically executed service period ends."""

    observation: list
    positions: list
    action: int
    base_reward: float
    duration_steps: int = 1
    completion_bonus: float = 0.0

    def advance(self) -> None:
        self.duration_steps += 1

    def add_completion_revenue(self, revenue: float, reward_weight: float) -> None:
        elapsed_steps = self.duration_steps - 1
        self.completion_bonus += (GAMMA ** elapsed_steps) * revenue * reward_weight

    def finalize(self, *, next_observation: list, next_positions: list, done: bool) -> dict:
        return {
            "observation": deepcopy(self.observation),
            "positions": deepcopy(self.positions),
            "action": self.action,
            "next_observation": deepcopy(next_observation),
            "next_positions": deepcopy(next_positions),
            "reward": self.base_reward + self.completion_bonus,
            "discount": GAMMA ** self.duration_steps,
            "done": done,
        }


def epsilon_for_step(step: int, *, start: float, end: float, decay_steps: int) -> float:
    """Linearly decay exploration probability while respecting its final floor."""
    if decay_steps < 1:
        raise ValueError("Epsilon decay steps must be positive")
    progress = min(max(step, 0) / decay_steps, 1.0)
    if progress == 1.0:
        return end
    return start + (end - start) * progress


def _seed_evaluation_episode(base_seed: int, episode: int) -> None:
    """Reset scenario RNGs so each evaluation episode is policy-comparable."""
    episode_seed = (base_seed + episode) % _NUMPY_SEED_MODULUS
    random.seed(episode_seed)
    np.random.seed(episode_seed)
    torch.manual_seed(episode_seed)


def checkpoint_names(conf: dict, policy: str = "mlp") -> tuple[str, str]:
    """Return the paired checkpoint names for an experiment configuration."""
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(f"Unsupported policy: {policy}")
    prefix = f"params{conf['PARAMS_NUM']}_{conf['PARAMS1']}_{conf['PARAMS2']}"
    marker = "" if policy == "mlp" else "_gnn"
    return f"{prefix}{marker}_mcs_net.pkl", f"{prefix}{marker}_iev_net.pkl"


def _default_checkpoint_names(policy: str) -> tuple[str, str]:
    if policy == "mlp":
        return DEFAULT_MCS_MODEL, DEFAULT_IEV_MODEL
    if policy == "gnn":
        return DEFAULT_GNN_MCS_MODEL, DEFAULT_GNN_IEV_MODEL
    raise ValueError(f"Unsupported policy: {policy}")


def _build_network_pair(policy: str):
    if policy == "mlp":
        return MCSNet(), MCSNet(), IEVNet(), IEVNet()
    if policy == "gnn":
        return SpatialGraphQNet(), SpatialGraphQNet(), SpatialGraphQNet(), SpatialGraphQNet()
    raise ValueError(f"Unsupported policy: {policy}")


class DQN:
    """The original pair of policy/value networks with configurable checkpoints."""

    def __init__(
        self,
        *,
        evaluate: bool = True,
        model_dir: Path | None = None,
        mcs_model_name: str | None = None,
        iev_model_name: str | None = None,
        policy: str = "mlp",
    ):
        self.policy = policy
        mcs_eval, mcs_target, iev_eval, iev_target = _build_network_pair(policy)
        self.mcs_eval_net, self.mcs_target_net = mcs_eval.to(device), mcs_target.to(device)
        self.iev_eval_net, self.iev_target_net = iev_eval.to(device), iev_target.to(device)
        self.mcs_learn_step_counter = 0
        self.iev_learn_step_counter = 0
        self.mcs_optimizer = torch.optim.Adam(self.mcs_eval_net.parameters(), lr=LR)
        self.iev_optimizer = torch.optim.Adam(self.iev_eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        if evaluate:
            default_mcs, default_iev = _default_checkpoint_names(policy)
            directory = Path(model_dir) if model_dir is not None else model_path(DEFAULT_MCS_MODEL).parent
            self._load_checkpoint_pair(
                directory / (mcs_model_name or default_mcs), directory / (iev_model_name or default_iev)
            )

    def _load_checkpoint_pair(self, mcs_checkpoint: Path, iev_checkpoint: Path) -> None:
        """Load bundled checkpoints on the selected local device."""
        mcs_state = torch.load(mcs_checkpoint, map_location=device, weights_only=True)
        iev_state = torch.load(iev_checkpoint, map_location=device, weights_only=True)
        self.mcs_eval_net.load_state_dict(mcs_state)
        self.mcs_target_net.load_state_dict(mcs_state)
        self.iev_eval_net.load_state_dict(iev_state)
        self.iev_target_net.load_state_dict(iev_state)

    def _values(self, network, features, positions=None):
        if self.policy == "gnn":
            if positions is None:
                raise ValueError("Graph policy requires candidate positions")
            return network(features, positions)
        return network(features)

    def mcs_choose_action(self, features, epsilon: float, positions=None):
        if not features:
            return None
        if np.random.uniform() >= epsilon:
            with torch.no_grad():
                values = self._values(self.mcs_eval_net, features, positions).reshape(1, -1)
            return int(torch.max(values, 1)[1].item())
        return int(np.random.randint(0, len(features)))

    def iev_choose_action(self, features, epsilon: float, positions=None):
        if not features:
            return None
        if np.random.uniform() >= epsilon:
            with torch.no_grad():
                values = self._values(self.iev_eval_net, features, positions).reshape(1, -1)
            return int(torch.max(values, 1)[1].item())
        return int(np.random.randint(0, len(features)))

    def mcs_learn(
        self, observation, action, next_observation, reward, positions=None,
        next_positions=None, done=False, discount=GAMMA,
    ) -> None:
        if self.mcs_learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.mcs_target_net.load_state_dict(self.mcs_eval_net.state_dict())
        self.mcs_learn_step_counter += 1
        q_eval = self._values(self.mcs_eval_net, observation, positions).reshape(-1)[action]
        if done:
            q_target = torch.tensor(float(reward), dtype=torch.float32, device=device)
        else:
            q_next = self._values(self.mcs_target_net, next_observation, next_positions).reshape(1, -1).detach()
            q_target = reward + discount * q_next.max(1)[0]
            q_target = q_target[0]
        loss = self.loss_func(q_eval, q_target)
        self.mcs_optimizer.zero_grad()
        loss.backward()
        self.mcs_optimizer.step()

    def iev_learn(self, observation, action, next_observation, reward, positions=None, next_positions=None, done=False) -> None:
        if self.iev_learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.iev_target_net.load_state_dict(self.iev_eval_net.state_dict())
        self.iev_learn_step_counter += 1
        q_eval = self._values(self.iev_eval_net, observation, positions).reshape(-1)[action]
        if done:
            q_target = torch.tensor(float(reward), dtype=torch.float32, device=device)
        else:
            q_next = self._values(self.iev_target_net, next_observation, next_positions).reshape(1, -1).detach()
            q_target = reward + GAMMA * q_next.max(1)[0]
            q_target = q_target[0]
        loss = self.loss_func(q_eval, q_target)
        self.iev_optimizer.zero_grad()
        loss.backward()
        self.iev_optimizer.step()

    def save_model(self, conf: dict, model_dir: Path) -> tuple[Path, Path]:
        """Save both networks under the experiment's existing naming convention."""
        model_dir.mkdir(parents=True, exist_ok=True)
        mcs_name, iev_name = checkpoint_names(conf, policy=self.policy)
        mcs_path, iev_path = model_dir / mcs_name, model_dir / iev_name
        torch.save(self.mcs_eval_net.state_dict(), mcs_path)
        torch.save(self.iev_eval_net.state_dict(), iev_path)
        return mcs_path, iev_path


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _conditional_mean(numerator: float, denominator: float) -> float | None:
    """Return no value when a conditional population is empty."""
    return numerator / denominator if denominator else None


def _episode_metrics(world: World, episode: int, decision_times: list[float]) -> dict[str, float | int | None]:
    profit = sum(mcs.total_profit - mcs.cost for mcs in world.MCSs)
    cost = sum(mcs.cost for mcs in world.MCSs)
    profit += cost
    cost = ((profit + cost) / 1.6) * 0.5 + cost
    profit -= cost
    idle_time = sum(mcs.total_idle_time for mcs in world.MCSs)
    mcs_reward = sum(mcs.total_reward for mcs in world.MCSs)
    requested = [ev for ev in world.EVs if getattr(ev, "requested_charge", False)]
    # `is_charged` is a legacy assignment flag. Use the explicit outcome
    # fields for all reported service-quality metrics, with a fallback for
    # legacy simulation objects created before the split was introduced.
    assigned = [
        ev for ev in requested
        if getattr(ev, "assigned_charge", getattr(ev, "is_charged", False))
    ]
    completed = [ev for ev in requested if getattr(ev, "charge_completed", False)]
    failed = [ev for ev in requested if getattr(ev, "fail_charge", False)]
    unserved = [
        ev for ev in requested
        if ev not in assigned and ev not in completed and ev not in failed
    ]
    in_progress = [
        ev for ev in assigned
        if ev not in completed and ev not in failed
    ]
    completed_wait_time = sum(ev.total_wait_time for ev in completed)
    assigned_wait_time = sum(ev.total_wait_time for ev in assigned)
    failed_wait_time = sum(ev.total_wait_time for ev in failed)
    completed_extra_distance = sum(ev.total_extra_dist for ev in completed)
    assigned_extra_distance = sum(ev.total_extra_dist for ev in assigned)
    iev_reward = sum(ev.total_reward for ev in completed)
    expense = sum(ev.expense for ev in completed)
    return {
        "episode": episode,
        "charge_request_count": len(requested),
        "assigned_charge_count": len(assigned),
        "completed_charge_count": len(completed),
        "failed_charge_count": len(failed),
        "unserved_request_count": len(unserved),
        "in_progress_request_count": len(in_progress),
        "assigned_proportion": _safe_divide(len(assigned), len(requested)),
        "completed_proportion": _safe_divide(len(completed), len(requested)),
        # Retained for backwards-compatible CSV consumers; it now has the
        # correct meaning: a request whose charge was actually completed.
        "charged_proportion": _safe_divide(len(completed), len(requested)),
        "failed_proportion": _safe_divide(len(failed), len(requested)),
        "mean_wait_time": _conditional_mean(completed_wait_time, len(completed)),
        "mean_wait_time_completed": _conditional_mean(completed_wait_time, len(completed)),
        "mean_wait_time_assigned": _conditional_mean(assigned_wait_time, len(assigned)),
        "mean_wait_time_failed": _conditional_mean(failed_wait_time, len(failed)),
        "mean_extra_distance": _conditional_mean(completed_extra_distance, len(completed)),
        "mean_extra_distance_completed": _conditional_mean(completed_extra_distance, len(completed)),
        "mean_extra_distance_assigned": _conditional_mean(assigned_extra_distance, len(assigned)),
        "mean_expense": _conditional_mean(expense, len(completed)),
        "mcs_idle_time": _safe_divide(idle_time, len(world.MCSs)),
        "mcs_cost": _safe_divide(cost, len(world.MCSs)),
        "mcs_profit": _safe_divide(profit, len(world.MCSs)),
        "mcs_reward": _safe_divide(mcs_reward, idle_time / 5),
        "iev_reward": _conditional_mean(iev_reward, completed_wait_time / 5),
        "mean_decision_ms": float(np.mean(decision_times)) if decision_times else 0.0,
    }


def _fallback_position(agent: MCS | Vehicle):
    """Keep an agent stationary or advance its recorded route without candidates."""
    if isinstance(agent, Vehicle) and agent.track and agent.track_index + 1 < len(agent.track):
        agent.track_index += 1
        return agent.track[agent.track_index]
    return agent.pos


def run_experiment(
    conf: dict,
    *,
    episodes: int = 60,
    steps: int = 100,
    evaluate: bool = True,
    model_dir: Path | None = None,
    mcs_model_name: str | None = None,
    iev_model_name: str | None = None,
    model_output_dir: Path | None = None,
    seed: int | None = None,
    policy: str = "mlp",
) -> list[dict[str, float | int]]:
    """Run a bounded simulation and return one stable metric mapping per episode."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    world = World(conf)
    env = MultiAgentEnv(world)
    dqn = DQN(
        evaluate=evaluate,
        model_dir=model_dir,
        mcs_model_name=mcs_model_name,
        iev_model_name=iev_model_name,
        policy=policy,
    )
    mcs_replay = ReplayBuffer(conf.get("REPLAY_CAPACITY", 5000))
    iev_replay = ReplayBuffer(conf.get("REPLAY_CAPACITY", 5000))
    environment_steps = 0
    results = []
    for episode in range(episodes):
        if evaluate and seed is not None:
            _seed_evaluation_episode(seed, episode)
        observations = env.reset()
        decision_times = []
        pending_mcs_transitions: dict[int, PendingMCSServiceTransition] = {}
        for step_index in range(steps):
            exploration = 0.0 if evaluate else epsilon_for_step(
                environment_steps,
                start=conf.get("EPSILON_START", 0.9),
                end=conf.get("EPSILON_END", 0.05),
                decay_steps=conf.get("EPSILON_DECAY_STEPS", 5000),
            )
            mcs_epsilon = 1.0 if conf["MCS_RANDOM"] else exploration
            iev_epsilon = 1.0 if conf["IEV_RANDOM"] else exploration
            actions, selected_actions = [], []
            for index, agent in enumerate(env.world.agents):
                started = time.perf_counter()
                observation = observations[index]
                if isinstance(agent, MCS):
                    action = dqn.mcs_choose_action(observation["feat"], mcs_epsilon, observation["pos"])
                else:
                    action = dqn.iev_choose_action(observation["feat"], iev_epsilon, observation["pos"])
                position = observation["pos"][action] if action is not None else _fallback_position(agent)
                actions.append(position)
                selected_actions.append(action)
                decision_times.append((time.perf_counter() - started) * 1000)
            new_observations, old_observations, rewards = env.step(actions)
            if not evaluate:
                for pending in pending_mcs_transitions.values():
                    pending.advance()
                for mcs_id, revenue in env.world.completed_service_events:
                    pending = pending_mcs_transitions.get(mcs_id)
                    if pending is not None:
                        pending.add_completion_revenue(
                            revenue, conf.get("SERVICE_REVENUE_REWARD_WEIGHT", 0)
                        )
                ready_mcs_observations = {
                    agent.ID: new_observations[index]
                    for index, agent in enumerate(env.world.agents)
                    if isinstance(agent, MCS)
                }
                for mcs_id, pending in list(pending_mcs_transitions.items()):
                    ready_observation = ready_mcs_observations.get(mcs_id)
                    if ready_observation is not None:
                        mcs_replay.append(pending.finalize(
                            next_observation=ready_observation["feat"],
                            next_positions=ready_observation["pos"],
                            done=step_index == steps - 1 or not ready_observation["feat"],
                        ))
                        del pending_mcs_transitions[mcs_id]
                for index, agent in enumerate(env.world.last_agents):
                    previous = observations[index]["feat"]
                    following = old_observations[index]["feat"]
                    action = selected_actions[index]
                    if not previous or action is None:
                        continue
                    if isinstance(agent, MCS) and not conf["MCS_RANDOM"]:
                        if not agent.is_idle and agent.charge_ev is not None:
                            pending_mcs_transitions[agent.ID] = PendingMCSServiceTransition(
                                observation=previous,
                                positions=observations[index]["pos"],
                                action=action,
                                base_reward=rewards[index],
                            )
                        else:
                            mcs_replay.append({
                                "observation": previous,
                                "positions": observations[index]["pos"],
                                "action": action,
                                "next_observation": following,
                                "next_positions": old_observations[index]["pos"],
                                "reward": rewards[index],
                                "discount": GAMMA,
                                "done": step_index == steps - 1 or not following,
                            })
                    elif isinstance(agent, Vehicle) and not conf["IEV_RANDOM"]:
                        iev_replay.append({
                            "observation": previous,
                            "positions": observations[index]["pos"],
                            "action": action,
                            "next_observation": following,
                            "next_positions": old_observations[index]["pos"],
                            "reward": rewards[index],
                            "done": step_index == steps - 1 or not following,
                        })
                if step_index == steps - 1:
                    for pending in pending_mcs_transitions.values():
                        mcs_replay.append(pending.finalize(
                            next_observation=[], next_positions=[], done=True,
                        ))
                    pending_mcs_transitions.clear()
                if len(mcs_replay) >= conf.get("REPLAY_WARMUP", 32):
                    for transition in mcs_replay.sample(conf.get("REPLAY_BATCH_SIZE", 16)):
                        dqn.mcs_learn(
                            transition["observation"], transition["action"], transition["next_observation"],
                            transition["reward"], transition["positions"], transition["next_positions"],
                            transition["done"], transition.get("discount", GAMMA),
                        )
                if len(iev_replay) >= conf.get("REPLAY_WARMUP", 32):
                    for transition in iev_replay.sample(conf.get("REPLAY_BATCH_SIZE", 16)):
                        dqn.iev_learn(
                            transition["observation"], transition["action"], transition["next_observation"],
                            transition["reward"], transition["positions"], transition["next_positions"], transition["done"],
                        )
            observations = new_observations
            environment_steps += 1
        results.append(_episode_metrics(world, episode, decision_times))
    if not evaluate:
        output_dir = Path(model_output_dir) if model_output_dir is not None else project_root() / "results" / "models"
        dqn.save_model(conf, output_dir)
    return results


def train(conf: dict, policy: str = "mlp"):
    """Backward-compatible legacy entry point for the original experiment scripts."""
    return run_experiment(conf, evaluate=False, policy=policy)
