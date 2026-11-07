from env.multiagent.config import CHARGE_PRICE
from env.multiagent.core import MCS
from env.multiagent.world import World
from marl_ev_mcs.config import default_config


def test_finished_charge_returns_revenue_and_world_records_the_owning_mcs():
    mcs = MCS(1, [104.0, 30.6], default_config("smoke"))
    mcs.charge_power = 10.0
    world = World.__new__(World)
    world.completed_service_events = []

    revenue = World.record_completed_service(world, mcs)

    assert revenue == 10.0 * CHARGE_PRICE
    assert world.completed_service_events == [(mcs.ID, revenue)]


def test_new_world_step_clears_previous_completion_events():
    world = World.__new__(World)
    world.completed_service_events = [(1, 18.0)]

    World.begin_update(world)

    assert world.completed_service_events == []


def test_reward_shaping_weights_are_configurable():
    config = default_config("smoke")

    assert config["SERVICE_REVENUE_REWARD_WEIGHT"] > 0
    assert config["MCS_MOVE_COST_PENALTY"] > 0
    assert config["IEV_WAIT_PENALTY"] > 0
