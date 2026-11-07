import torch

from DQN import dqn as runner
from marl_ev_mcs.config import default_config


def test_replay_buffer_is_bounded_and_copies_transitions():
    buffer = runner.ReplayBuffer(capacity=2)
    buffer.append({"features": [[1.0, 2.0]]})
    second = {"features": [[3.0, 4.0]]}
    buffer.append(second)
    second["features"][0][0] = 99.0
    buffer.append({"features": [[5.0, 6.0]]})

    samples = buffer.sample(10)

    assert len(buffer) == 2
    assert len(samples) == 2
    assert any(sample["features"][0][0] == 3.0 for sample in samples)


def test_epsilon_schedule_decays_and_stops_at_floor():
    assert runner.epsilon_for_step(0, start=0.9, end=0.1, decay_steps=100) == 0.9
    assert runner.epsilon_for_step(50, start=0.9, end=0.1, decay_steps=100) == 0.5
    assert runner.epsilon_for_step(500, start=0.9, end=0.1, decay_steps=100) == 0.1


def test_epsilon_is_exploration_probability(monkeypatch):
    dqn = runner.DQN(evaluate=False)
    monkeypatch.setattr(dqn.mcs_eval_net, "forward", lambda features: torch.tensor([[1.0], [9.0]]))
    monkeypatch.setattr(runner.np.random, "randint", lambda low, high: 0)

    greedy = dqn.mcs_choose_action([[1.0, 1.0], [2.0, 2.0]], epsilon=0.0)
    exploratory = dqn.mcs_choose_action([[1.0, 1.0], [2.0, 2.0]], epsilon=1.0)

    assert greedy == 1
    assert exploratory == 0


def test_terminal_transition_does_not_bootstrap_target_network(monkeypatch):
    dqn = runner.DQN(evaluate=False)

    def forbidden_target(*args, **kwargs):
        raise AssertionError("terminal transitions must not bootstrap")

    monkeypatch.setattr(dqn.mcs_target_net, "forward", forbidden_target)
    dqn.mcs_learn([[1.0, 1.0]], 0, [], 1.0, done=True)


def test_mlp_iev_learning_updates_finite_parameters():
    dqn = runner.DQN(evaluate=False, policy="mlp")
    before = [parameter.detach().clone() for parameter in dqn.iev_eval_net.parameters()]

    dqn.iev_learn([[1.0, 2.0], [2.0, 3.0]], 0, [[2.0, 1.0], [3.0, 2.0]], 1.0)

    after = list(dqn.iev_eval_net.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in after)
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_gnn_mcs_learning_updates_finite_parameters():
    dqn = runner.DQN(evaluate=False, policy="gnn")
    before = [parameter.detach().clone() for parameter in dqn.mcs_eval_net.parameters()]
    positions = [[103.9, 30.6], [104.0, 30.65]]

    dqn.mcs_learn(
        [[1.0, 2.0], [2.0, 3.0]], 0, [[2.0, 1.0], [3.0, 2.0]], 1.0, positions, positions
    )

    after = list(dqn.mcs_eval_net.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in after)
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


def test_training_updates_from_replayed_mcs_transition(monkeypatch, tmp_path):
    calls = []

    def record_learn(self, *args, **kwargs):
        calls.append(args)

    monkeypatch.setattr(runner.DQN, "mcs_learn", record_learn)
    config = default_config("smoke")
    config.update({"REPLAY_WARMUP": 1, "REPLAY_BATCH_SIZE": 1, "REPLAY_CAPACITY": 8})

    runner.run_experiment(
        config, episodes=1, steps=2, evaluate=False, model_output_dir=tmp_path, seed=7
    )

    assert calls


def test_gnn_training_updates_from_replayed_transition(monkeypatch, tmp_path):
    calls = []

    def record_learn(self, *args, **kwargs):
        calls.append(args)

    monkeypatch.setattr(runner.DQN, "mcs_learn", record_learn)
    config = default_config("smoke")
    config.update({"REPLAY_WARMUP": 1, "REPLAY_BATCH_SIZE": 1, "REPLAY_CAPACITY": 8})

    runner.run_experiment(
        config, episodes=1, steps=2, evaluate=False, policy="gnn", model_output_dir=tmp_path, seed=7
    )

    assert calls
    assert calls[0][4]
