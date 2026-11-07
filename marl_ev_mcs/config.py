"""Named configurations for the original experiment and a fast smoke run."""

from copy import deepcopy


_DEFAULT_CONFIG = {
    "LOW_POWER": 8,
    "MAX_WAIT_TIME": 2,
    "CHARGE_SPEED": 120,
    "NUM_MCS": 50,
    "NUM_EV": 500,
    "POWER_MEAN": 60,
    "PARAMS_NUM": 1,
    "PARAMS1": 500,
    "PARAMS2": 50,
    "MEAN": 40,
    "STAND": 16,
    "MCS_RANDOM": False,
    "IEV_RANDOM": False,
    "TRACK_DAY": 5,
    "REPLAY_CAPACITY": 5000,
    "REPLAY_WARMUP": 32,
    "REPLAY_BATCH_SIZE": 16,
    "EPSILON_START": 0.9,
    "EPSILON_END": 0.05,
    "EPSILON_DECAY_STEPS": 5000,
    "SERVICE_REVENUE_REWARD_WEIGHT": 0.05,
    "MCS_MOVE_COST_PENALTY": 1.0,
    "IEV_WAIT_PENALTY": 0.05,
}


def default_config(preset: str = "default") -> dict:
    """Return an independent configuration for a named experiment preset."""
    config = deepcopy(_DEFAULT_CONFIG)
    if preset == "default":
        return config
    if preset == "smoke":
        config.update({"NUM_MCS": 2, "NUM_EV": 20, "PARAMS1": 20, "PARAMS2": 2})
        return config
    raise ValueError(f"Unknown preset: {preset}. Expected 'default' or 'smoke'.")
