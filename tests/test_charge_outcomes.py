from types import SimpleNamespace

import pytest

from DQN.dqn import _episode_metrics
from env.multiagent.config import CHARGE_PRICE
from env.multiagent.core import MCS, Vehicle
from marl_ev_mcs.config import default_config


def test_finishing_charge_marks_the_assigned_vehicle_as_completed():
    config = default_config("smoke")
    ev = Vehicle(1, remain=5.0, distance=20.0, pos=[104.0, 30.6], conf=config)
    mcs = MCS(1, [104.0, 30.6], config)
    mcs.is_idle = False
    mcs.charge_ev = ev
    mcs.charge_power = 10.0

    revenue = mcs.finish_charge()

    assert ev.charge_completed is True
    assert ev.need_charge is False
    assert mcs.completed_service_count == 1
    assert revenue == pytest.approx(10.0 * CHARGE_PRICE)


def test_episode_metrics_distinguish_assignment_completion_failure_and_unserved():
    def ev(**overrides):
        values = {
            "requested_charge": True,
            "assigned_charge": False,
            "charge_completed": False,
            "fail_charge": False,
            "total_wait_time": 0.0,
            "total_extra_dist": 0.0,
            "total_reward": 0.0,
            "expense": 0.0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    world = SimpleNamespace(
        MCSs=[],
        EVs=[
            ev(assigned_charge=True, charge_completed=True, total_wait_time=10.0, total_extra_dist=2.0),
            ev(fail_charge=True, total_wait_time=25.0),
            ev(assigned_charge=True, total_wait_time=7.0),
            ev(),
        ],
    )

    metrics = _episode_metrics(world, episode=0, decision_times=[])

    assert metrics["charge_request_count"] == 4
    assert metrics["assigned_proportion"] == pytest.approx(0.5)
    assert metrics["completed_proportion"] == pytest.approx(0.25)
    assert metrics["charged_proportion"] == pytest.approx(0.25)
    assert metrics["failed_proportion"] == pytest.approx(0.25)
    assert metrics["unserved_request_count"] == 1
    assert metrics["in_progress_request_count"] == 1
    assert metrics["mean_wait_time_completed"] == pytest.approx(10.0)
    assert metrics["mean_wait_time_failed"] == pytest.approx(25.0)


def test_completed_only_metrics_are_undefined_when_no_request_completes():
    ev = SimpleNamespace(
        requested_charge=True,
        assigned_charge=False,
        charge_completed=False,
        fail_charge=True,
        total_wait_time=25.0,
        total_extra_dist=3.0,
        total_reward=0.0,
        expense=0.0,
    )
    world = SimpleNamespace(MCSs=[], EVs=[ev])

    metrics = _episode_metrics(world, episode=0, decision_times=[])

    assert metrics["mean_wait_time_completed"] is None
    assert metrics["mean_extra_distance_completed"] is None
    assert metrics["mean_expense"] is None
