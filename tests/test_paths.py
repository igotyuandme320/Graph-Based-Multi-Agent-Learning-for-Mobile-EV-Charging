from marl_ev_mcs.config import default_config
from marl_ev_mcs.paths import data_path, model_path, project_root


def test_project_paths_resolve_bundled_data():
    assert project_root().is_dir()
    assert data_path("hash_point_remove.csv").is_file()
    assert data_path("track", "20140805.csv").is_file()


def test_model_path_resolves_bundled_checkpoint():
    assert model_path("params1_500_50_mcs_net.pkl").is_file()


def test_smoke_preset_is_smaller_than_default():
    assert default_config("smoke")["NUM_EV"] < default_config("default")["NUM_EV"]
    assert default_config("smoke")["NUM_MCS"] < default_config("default")["NUM_MCS"]
