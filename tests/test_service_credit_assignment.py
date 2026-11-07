import pytest
import torch

from DQN import dqn as runner
from env.multiagent.core import MCS
from marl_ev_mcs.config import default_config


def test_pending_mcs_transition_accumulates_service_bonus_and_uses_smdp_discount():
    pending = runner.PendingMCSServiceTransition(
        observation=[[1.0, 2.0]],
        positions=[[104.0, 30.6]],
        action=0,
        base_reward=0.25,
    )

    pending.advance()
    pending.add_completion_revenue(revenue=10.0, reward_weight=0.05)
    transition = pending.finalize(
        next_observation=[[3.0, 4.0]],
        next_positions=[[104.1, 30.7]],
        done=False,
    )

    assert transition["reward"] == pytest.approx(0.70)
    assert transition["discount"] == pytest.approx(runner.GAMMA ** 2)
    assert transition["done"] is False


def test_pending_mcs_transition_flushes_as_terminal_without_next_candidates():
    pending = runner.PendingMCSServiceTransition(
        observation=[[1.0, 2.0]],
        positions=[[104.0, 30.6]],
        action=0,
        base_reward=0.25,
    )

    terminal = pending.finalize(next_observation=[], next_positions=[], done=True)

    assert terminal["done"] is True
    assert terminal["next_observation"] == []


def test_mcs_learning_uses_transition_specific_discount(monkeypatch):
    dqn = runner.DQN(evaluate=False)
    captured = {}

    def capture_target(q_eval, q_target):
        captured["target"] = float(q_target.detach().cpu())
        return q_eval * 0

    monkeypatch.setattr(dqn, "loss_func", capture_target)
    monkeypatch.setattr(
        dqn.mcs_target_net,
        "forward",
        lambda features: torch.tensor([[2.0]], dtype=torch.float32),
    )

    dqn.mcs_learn(
        [[1.0, 2.0]], 0, [[3.0, 4.0]], reward=1.0, discount=0.25
    )

    assert captured["target"] == pytest.approx(1.5)


def test_runner_marks_reentered_pending_mcs_transition_terminal_on_final_step(monkeypatch):
    config = default_config("smoke")
    config.update({"REPLAY_WARMUP": 1, "REPLAY_BATCH_SIZE": 1, "REPLAY_CAPACITY": 4})
    mcs = MCS(7, [104.0, 30.6], config)
    calls = []

    class FakeWorld:
        def __init__(self, conf):
            self.mcs = mcs
            self.agents = [mcs]
            self.last_agents = []
            self.completed_service_events = []

    class FakeEnv:
        def __init__(self, world):
            self.world = world
            self.step_count = 0

        def reset(self):
            return [{"feat": [[1.0, 2.0]], "pos": [[104.0, 30.6]]}]

        def step(self, actions):
            self.step_count += 1
            if self.step_count == 1:
                self.world.last_agents = [mcs]
                self.world.agents = []
                mcs.is_idle = False
                mcs.charge_ev = object()
                self.world.completed_service_events = []
                return [], [{"feat": [[1.5, 2.5]], "pos": [[104.0, 30.6]]}], [0.25]
            self.world.last_agents = []
            self.world.agents = [mcs]
            mcs.is_idle = True
            mcs.charge_ev = None
            self.world.completed_service_events = [(mcs.ID, 10.0), (mcs.ID, 5.0)]
            return [{"feat": [[3.0, 4.0]], "pos": [[104.1, 30.7]]}], [], []

    class FakeDQN:
        def __init__(self, **kwargs):
            pass

        def mcs_choose_action(self, features, epsilon, positions):
            return 0

        def iev_choose_action(self, features, epsilon, positions):
            return 0

        def mcs_learn(self, *args):
            calls.append(args)

        def iev_learn(self, *args):
            raise AssertionError("This scripted run has no IEV decisions")

        def save_model(self, conf, model_dir):
            return None

    monkeypatch.setattr(runner, "World", FakeWorld)
    monkeypatch.setattr(runner, "MultiAgentEnv", FakeEnv)
    monkeypatch.setattr(runner, "DQN", FakeDQN)
    monkeypatch.setattr(runner, "_episode_metrics", lambda *args: {"episode": 0})

    runner.run_experiment(config, episodes=1, steps=2, evaluate=False)

    assert len(calls) == 1
    assert calls[0][3] == pytest.approx(0.25 + runner.GAMMA * 0.75)
    assert calls[0][6] is True
    assert calls[0][7] == pytest.approx(runner.GAMMA ** 2)
