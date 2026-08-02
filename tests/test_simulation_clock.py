
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from modules.utils import simulation_time as simclock

LONDON = datetime(2024, 3, 11, 9, 30, tzinfo=timezone.utc)
TOKYO = datetime(2024, 3, 11, 2, 0, tzinfo=timezone.utc)
SATURDAY = datetime(2024, 3, 16, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _restore_clock():
    previous = simclock.get_mode()
    yield
    simclock.set_mode(previous)
    simclock.reset()


def test_live_mode_returns_wall_clock():
    simclock.set_mode(simclock.TimeMode.LIVE)
    simclock.set_bar_time(LONDON, step=0)
    delta = abs((simclock.now(timezone.utc) - datetime.now(timezone.utc)).total_seconds())
    assert delta < 5, "live mode must ignore simulated bar time"


def test_simulation_mode_returns_bar_time():
    with simclock.simulation_mode(LONDON):
        assert simclock.now(timezone.utc) == LONDON


def test_simulation_time_is_independent_of_wall_clock():
    with simclock.simulation_mode(LONDON):
        first = simclock.now(timezone.utc)
        first_session = simclock.get_session_info()


    with simclock.simulation_mode(LONDON):
        second = simclock.now(timezone.utc)
        second_session = simclock.get_session_info()

    assert first == second
    assert first_session["hour"] == second_session["hour"] == 9
    assert first_session["weekday"] == second_session["weekday"] == 0


def test_session_info_tracks_the_replayed_bar():
    with simclock.simulation_mode(TOKYO):
        assert simclock.get_session_info()["hour"] == 2
        simclock.set_bar_time(LONDON, step=1)
        assert simclock.get_session_info()["hour"] == 9


def test_weekend_detection_uses_bar_time():
    with simclock.simulation_mode(SATURDAY):
        info = simclock.get_session_info()
        assert info["weekday"] == 5
        assert info["is_weekend"] is True


def test_naive_timestamps_are_accepted():
    with simclock.simulation_mode():
        simclock.set_bar_time(datetime(2024, 3, 11, 9, 30), step=0)  # noqa: DTZ001 - naive on purpose
        assert simclock.now(timezone.utc).hour == 9


def test_cooldown_counts_steps_not_seconds():
    with simclock.simulation_mode(LONDON):
        assert simclock.cooldown_elapsed("entry", min_steps=5)
        simclock.record_cooldown_action("entry")
        assert not simclock.cooldown_elapsed("entry", min_steps=5)

        for step in range(1, 5):
            simclock.set_bar_time(LONDON + timedelta(minutes=15 * step), step=step)
            assert not simclock.cooldown_elapsed("entry", min_steps=5)

        simclock.set_bar_time(LONDON + timedelta(minutes=75), step=5)
        assert simclock.cooldown_elapsed("entry", min_steps=5)


def test_cooldowns_reset_between_episodes():
    with simclock.simulation_mode(LONDON):
        simclock.record_cooldown_action("entry")
        assert not simclock.cooldown_elapsed("entry", min_steps=10)
        simclock.reset()
        assert simclock.cooldown_elapsed("entry", min_steps=10)


def test_install_clock_redirects_datetime_now():
    import types

    victim = types.ModuleType("victim")
    exec("from datetime import datetime\ndef when(): return datetime.now()", victim.__dict__)

    patched = simclock.install_clock(victim)
    assert "datetime-class" in patched

    with simclock.simulation_mode(LONDON):
        assert victim.when().hour == 9  # type: ignore[attr-defined]


def test_install_clock_handles_module_style_import():
    import types

    victim = types.ModuleType("victim2")
    exec("import datetime\ndef when(): return datetime.datetime.now()", victim.__dict__)

    patched = simclock.install_clock(victim)
    assert "datetime-module" in patched

    with simclock.simulation_mode(TOKYO):
        assert victim.when().hour == 2  # type: ignore[attr-defined]


def test_install_clock_makes_sleep_a_noop_under_simulation():
    import types

    victim = types.ModuleType("victim3")
    exec("import time\ndef nap(): time.sleep(30); return 'done'", victim.__dict__)
    simclock.install_clock(victim)

    with simclock.simulation_mode(LONDON):
        started = datetime.now(timezone.utc)
        assert victim.nap() == "done"  # type: ignore[attr-defined]
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    assert elapsed < 1.0, "sleep() was not suppressed under simulation"


def test_env_publishes_bar_time_on_step(env):
    simclock.set_mode(simclock.TimeMode.SIMULATION)
    env.reset(seed=0)

    first = simclock.get_bar_time()
    assert first is not None, "reset() must anchor simulated time"

    import numpy as np

    for _ in range(5):
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        env.step(int(valid[0]) if valid.size else 0)

    later = simclock.get_bar_time()
    assert later is not None
    assert later > first, "stepping the env must advance simulated time"
    assert simclock.get_current_step() == env.current_step
