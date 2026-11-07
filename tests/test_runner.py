import math

from DQN import dqn as runner
from DQN.net import IEVNet, MCSNet
from marl_ev_mcs.config import default_config


def test_evaluation_loads_checkpoints_with_explicit_map_location(monkeypatch, tmp_path):
    loads = []

    def fake_load(path, *, map_location, weights_only=True):
        loads.append((path, map_location, weights_only))
        return MCSNet().state_dict() if "mcs" in str(path) else IEVNet().state_dict()

    monkeypatch.setattr(runner.torch, "load", fake_load)

    runner.DQN(evaluate=True, model_dir=tmp_path)

    assert len(loads) == 2
    assert all(map_location == runner.device for _, map_location, _ in loads)
    assert all(weights_only is True for _, _, weights_only in loads)


def test_smoke_evaluation_returns_finite_metrics():
    metrics = runner.run_experiment(
        default_config("smoke"), episodes=1, steps=1, evaluate=True, seed=7
    )

    assert len(metrics) == 1
    assert metrics[0]["episode"] == 0
    assert metrics[0]["charged_proportion"] == 0.0
    assert all(math.isfinite(value) for value in metrics[0].values() if isinstance(value, float))


def test_training_saves_reusable_checkpoint_pair_without_loading_evaluation_model(monkeypatch, tmp_path):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("Training must not load an evaluation checkpoint")

    monkeypatch.setattr(runner.torch, "load", forbidden_load)
    config = default_config("smoke")
    runner.run_experiment(
        config, episodes=1, steps=1, evaluate=False, model_output_dir=tmp_path, seed=7
    )

    mcs_name, iev_name = runner.checkpoint_names(config)
    assert (tmp_path / mcs_name).is_file()
    assert (tmp_path / iev_name).is_file()


def test_evaluation_can_select_explicit_checkpoint_names(monkeypatch, tmp_path):
    loaded_names = []

    def fake_load(path, *, map_location, weights_only=True):
        loaded_names.append(path.name)
        return MCSNet().state_dict() if "mcs" in path.name else IEVNet().state_dict()

    monkeypatch.setattr(runner.torch, "load", fake_load)
    runner.DQN(
        evaluate=True,
        model_dir=tmp_path,
        mcs_model_name="custom_mcs_net.pkl",
        iev_model_name="custom_iev_net.pkl",
    )

    assert loaded_names == ["custom_mcs_net.pkl", "custom_iev_net.pkl"]


def test_saved_training_checkpoints_can_be_reloaded(tmp_path):
    config = default_config("smoke")
    runner.run_experiment(
        config, episodes=1, steps=1, evaluate=False, model_output_dir=tmp_path, seed=7
    )

    mcs_name, iev_name = runner.checkpoint_names(config)
    reloaded = runner.DQN(
        evaluate=True,
        model_dir=tmp_path,
        mcs_model_name=mcs_name,
        iev_model_name=iev_name,
    )

    assert reloaded.mcs_eval_net is not None
    assert reloaded.iev_eval_net is not None
