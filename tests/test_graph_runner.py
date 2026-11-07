import pytest

from DQN import dqn as runner
from marl_ev_mcs.config import default_config
from marl_ev_mcs.paths import model_path


def test_graph_checkpoint_names_do_not_collide_with_mlp_names():
    config = default_config("smoke")

    assert runner.checkpoint_names(config, policy="gnn") != runner.checkpoint_names(config, policy="mlp")
    assert "_gnn_" in runner.checkpoint_names(config, policy="gnn")[0]


def test_graph_policy_smoke_training_saves_and_reloads(tmp_path):
    config = default_config("smoke")
    runner.run_experiment(
        config,
        episodes=1,
        steps=1,
        evaluate=False,
        policy="gnn",
        model_output_dir=tmp_path,
        seed=7,
    )

    mcs_name, iev_name = runner.checkpoint_names(config, policy="gnn")
    assert (tmp_path / mcs_name).is_file()
    assert (tmp_path / iev_name).is_file()

    metrics = runner.run_experiment(
        config,
        episodes=1,
        steps=1,
        evaluate=True,
        policy="gnn",
        model_dir=tmp_path,
        mcs_model_name=mcs_name,
        iev_model_name=iev_name,
        seed=7,
    )

    assert len(metrics) == 1


def test_graph_policy_rejects_mlp_checkpoint_pair():
    config = default_config("default")
    mcs_name, iev_name = runner.checkpoint_names(config, policy="mlp")

    with pytest.raises(RuntimeError):
        runner.DQN(
            evaluate=True,
            policy="gnn",
            model_dir=model_path(mcs_name).parent,
            mcs_model_name=mcs_name,
            iev_model_name=iev_name,
        )
