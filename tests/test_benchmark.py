import pandas as pd
import pytest
import numpy as np

from DQN import dqn as runner
from env.multiagent.core import MCS
from marl_ev_mcs import benchmark
from marl_ev_mcs.config import default_config


def test_parse_seed_list_rejects_blank_and_duplicate_tokens():
    assert benchmark.parse_seed_list("11, 22,33") == [11, 22, 33]

    with pytest.raises(ValueError, match="blank"):
        benchmark.parse_seed_list("11,,22")
    with pytest.raises(ValueError, match="unique"):
        benchmark.parse_seed_list("11,11")
    with pytest.raises(ValueError, match="between"):
        benchmark.parse_seed_list("-1")
    with pytest.raises(ValueError, match="between"):
        benchmark.parse_seed_list(str(2**32))


def test_summary_aggregates_episodes_before_seed_runs_and_fills_single_std():
    episodes = pd.DataFrame([
        {"policy": "mlp", "train_seed": 1, "eval_seed": 10, "episode": 0, "completed_proportion": 0.2},
        {"policy": "mlp", "train_seed": 1, "eval_seed": 10, "episode": 1, "completed_proportion": 0.4},
        {"policy": "mlp", "train_seed": 2, "eval_seed": 10, "episode": 0, "completed_proportion": 0.6},
        {"policy": "mlp", "train_seed": 2, "eval_seed": 10, "episode": 1, "completed_proportion": 0.8},
        {"policy": "gnn", "train_seed": 1, "eval_seed": 10, "episode": 0, "completed_proportion": 0.5},
    ])

    per_seed, summary = benchmark.summarize_episode_metrics(episodes)

    mlp_runs = per_seed.loc[per_seed["policy"] == "mlp", "completed_proportion"].tolist()
    assert mlp_runs == pytest.approx([0.3, 0.7])
    mlp_summary = summary.loc[summary["policy"] == "mlp"].iloc[0]
    gnn_summary = summary.loc[summary["policy"] == "gnn"].iloc[0]
    assert mlp_summary["completed_proportion_mean"] == pytest.approx(0.5)
    assert mlp_summary["completed_proportion_std"] == pytest.approx(0.282842712474619)
    assert mlp_summary["completed_proportion_n"] == 2
    assert mlp_summary["n_runs"] == 2
    assert gnn_summary["completed_proportion_std"] == 0.0
    assert gnn_summary["completed_proportion_n"] == 1


def test_benchmark_trains_each_policy_seed_then_evaluates_its_own_checkpoints(monkeypatch, tmp_path):
    calls = []

    def fake_run_experiment(config, **kwargs):
        calls.append(kwargs)
        if kwargs["evaluate"]:
            return [{"episode": 0, "completed_proportion": kwargs["seed"] / 1000}]
        return [{"episode": 0, "completed_proportion": 0.0}]

    monkeypatch.setattr(benchmark, "run_experiment", fake_run_experiment)
    outputs = benchmark.run_benchmark(
        default_config("smoke"),
        policies=["mlp", "gnn"],
        train_seeds=[11, 22],
        eval_seeds=[101, 202],
        train_episodes=2,
        train_steps=3,
        eval_episodes=1,
        eval_steps=4,
        output_dir=tmp_path,
    )

    train_calls = [call for call in calls if not call["evaluate"]]
    eval_calls = [call for call in calls if call["evaluate"]]
    assert len(train_calls) == 4
    assert len(eval_calls) == 8
    assert all(call["episodes"] == 2 and call["steps"] == 3 for call in train_calls)
    assert all(call["episodes"] == 1 and call["steps"] == 4 for call in eval_calls)
    assert {
        call["model_output_dir"].relative_to(tmp_path).as_posix()
        for call in train_calls
    } == {
        "mlp/train-seed-11/models", "mlp/train-seed-22/models",
        "gnn/train-seed-11/models", "gnn/train-seed-22/models",
    }
    assert all("train-seed-" in call["model_dir"].as_posix() for call in eval_calls)
    assert all(call["mcs_model_name"].endswith("_mcs_net.pkl") for call in eval_calls)
    assert {
        (call["policy"], call["seed"], call["model_dir"].relative_to(tmp_path).as_posix())
        for call in eval_calls
    } == {
        (policy, eval_seed, f"{policy}/train-seed-{train_seed}/models")
        for policy in ("mlp", "gnn")
        for train_seed in (11, 22)
        for eval_seed in (101, 202)
    }
    assert outputs["episode_metrics"].is_file()
    assert outputs["per_seed_metrics"].is_file()
    assert outputs["summary_metrics"].is_file()
    assert outputs["protocol"].is_file()
    assert len(pd.read_csv(outputs["per_seed_metrics"])) == 8


def test_same_seed_recreates_each_episode_scenario_despite_policy_rng_consumption(monkeypatch):
    scenarios = {"mlp": [], "gnn": []}
    config = default_config("smoke")

    class FakeWorld:
        def __init__(self, conf):
            self.mcs = MCS(1, [104.0, 30.6], conf)
            self.agents = []
            self.last_agents = []
            self.completed_service_events = []

    class FakeEnv:
        def __init__(self, world):
            self.world = world

        def reset(self):
            scenarios[current_policy[0]].append(int(np.random.randint(0, 2**32 - 1)))
            self.world.agents = [self.world.mcs]
            return [{"feat": [[1.0, 2.0]], "pos": [[104.0, 30.6]]}]

        def step(self, actions):
            self.world.last_agents = self.world.agents
            self.world.agents = [self.world.mcs]
            observation = {"feat": [[1.0, 2.0]], "pos": [[104.0, 30.6]]}
            return [observation], [observation], [0.0]

    current_policy = ["mlp"]

    class FakeDQN:
        def __init__(self, *, policy, **kwargs):
            self.policy = policy

        def mcs_choose_action(self, features, epsilon, positions):
            np.random.uniform(size=1 if self.policy == "mlp" else 7)
            return 0

        def iev_choose_action(self, features, epsilon, positions):
            return 0

        def save_model(self, conf, model_dir):
            return None

    monkeypatch.setattr(runner, "World", FakeWorld)
    monkeypatch.setattr(runner, "MultiAgentEnv", FakeEnv)
    monkeypatch.setattr(runner, "DQN", FakeDQN)
    monkeypatch.setattr(runner, "_episode_metrics", lambda *args: {"episode": 0})

    for policy in ("mlp", "gnn"):
        current_policy[0] = policy
        runner.run_experiment(config, episodes=2, steps=1, evaluate=True, seed=101, policy=policy)

    assert scenarios["mlp"] == scenarios["gnn"]
