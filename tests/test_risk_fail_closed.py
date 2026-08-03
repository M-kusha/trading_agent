from __future__ import annotations

from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from envs.core.env_types import CloseReason, PropFirmConfig, PropPosition
from envs.prop_firm_env import PropFirmTradingEnv
from modules.meta.live_action_mask import LiveActionMaskBuilder, LiveMaskConfig
from modules.meta.ppo_core import PPOCore, PPOCoreConfig
from modules.meta.ppo_observation_builder import PPO_OBS_SIZE
from trading.state import LiveStateHost


def _live_frames(market_data: dict) -> dict:
    frames = {
        timeframe: frame.tail({"M15": 150, "H1": 80, "H4": 60, "D1": 60}[timeframe]).copy()
        for timeframe, frame in market_data["XAUUSD"].items()
    }
    frames["M15"]["spread"] = 35.0
    return frames


@pytest.fixture
def risk_env(market_data: dict) -> Generator[PropFirmTradingEnv, None, None]:
    config = PropFirmConfig()

    # Keep these tests focused on the non-negotiable risk boundary.  The
    # curriculum's softer teaching gates, session timing, and randomized
    # execution must not decide whether the hard veto is exercised.
    config.domain_randomization_enabled = False
    config.mirror_augmentation_prob = 0.0
    config.high_vol_oversample_prob = 0.0
    config.enforce_weekend_block = False
    config.enforce_no_new_trades_window = False
    config.enforce_hard_close = False
    config.entry_quality_gate_enabled = False
    config.min_setup_quality_for_entry = 0.0
    config.observation_period_required = False
    config.max_trades_per_day = 999
    config.max_trades_per_session = 999
    config.max_trades_per_episode = 999
    config.max_consecutive_losses = 999
    config.daily_drawdown_limit = 0.50
    config.max_drawdown_limit = 0.50
    config.daily_dd_safety_buffer = 0.0
    config.max_dd_safety_buffer = 0.0
    config.firm_daily_drawdown_limit = 0.05
    config.firm_max_drawdown_limit = 0.10
    config.dd_entry_veto_fraction = 0.75
    config.max_steps_per_episode = 128

    env = PropFirmTradingEnv(market_data, config)
    env.reset(seed=17)
    try:
        yield env
    finally:
        env.close()


def _set_total_drawdown_without_daily_breach(
    env: PropFirmTradingEnv, *, equity: float
) -> None:
    env.balance = float(equity)
    env.equity = float(equity)
    # Isolate the total-account veto.  The account is already at this day's
    # starting balance, so an episode termination cannot obscure entry denial.
    env.day_start_balance = float(equity)


def test_direct_raw_entry_is_rejected_at_firm_drawdown_veto(
    risk_env: PropFirmTradingEnv,
) -> None:
    _set_total_drawdown_without_daily_breach(risk_env, equity=92_000.0)

    mask = risk_env.action_masks()
    assert mask[0]
    assert not mask[1:9].any()

    _, _, terminated, truncated, info = risk_env.step(1)

    assert not terminated
    assert not truncated
    assert info["entry_allowed"] is False
    assert info["block_reason"] == "drawdown_entry_veto"
    assert risk_env.pending_entry is None
    assert risk_env.position is None


def test_pending_entry_is_cancelled_if_drawdown_worsens_before_fill(
    risk_env: PropFirmTradingEnv,
) -> None:
    _, _, terminated, truncated, first_info = risk_env.step(1)
    assert not terminated
    assert not truncated
    assert first_info["entry_allowed"] is True
    assert risk_env.pending_entry is not None
    assert risk_env.position is None

    _set_total_drawdown_without_daily_breach(risk_env, equity=92_000.0)
    _, _, terminated, truncated, second_info = risk_env.step(0)

    assert not terminated
    assert not truncated
    assert second_info["block_reason"] == "drawdown_entry_veto"
    assert risk_env.pending_entry is None
    assert risk_env.position is None
    assert risk_env.daily_trades == 0


@pytest.mark.parametrize(
    ("current_dd", "daily_dd", "risk_timestamp"),
    [
        pytest.param(None, None, "fresh", id="missing-drawdown"),
        pytest.param(float("nan"), 0.0, "fresh", id="nan-drawdown"),
        pytest.param(0.0, 0.0, None, id="missing-timestamp"),
        pytest.param(0.0, 0.0, "stale", id="stale-timestamp"),
        pytest.param(0.08, 0.0, "fresh", id="high-drawdown"),
    ],
)
def test_live_mask_fails_closed_for_invalid_or_unsafe_risk_state(
    current_dd: float | None,
    daily_dd: float | None,
    risk_timestamp: str | None,
) -> None:
    now = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    timestamp = {
        "fresh": now,
        "stale": now - timedelta(seconds=61),
        None: None,
    }[risk_timestamp]
    builder = LiveActionMaskBuilder(LiveMaskConfig())

    flat_mask = builder.get_action_mask(
        has_position=False,
        trade_open_allowed=True,
        current_dd=current_dd,
        daily_dd=daily_dd,
        current_time=now,
        risk_timestamp=timestamp,
    )
    assert flat_mask[0]
    assert not flat_mask[1:9].any()
    assert not flat_mask[9]

    positioned_mask = builder.get_action_mask(
        has_position=True,
        trade_open_allowed=True,
        current_dd=current_dd,
        daily_dd=daily_dd,
        current_time=now,
        risk_timestamp=timestamp,
    )
    assert positioned_mask[0]
    assert not positioned_mask[1:9].any()
    assert positioned_mask[9]


def test_old_checkpoint_and_wrong_runtime_observation_width_fail_closed() -> None:
    runtime_core = PPOCore(PPOCoreConfig(obs_size=PPO_OBS_SIZE))
    runtime_core._sb3_model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(PPO_OBS_SIZE - 1,))
    )

    with pytest.raises(RuntimeError, match="loaded checkpoint expects"):
        runtime_core.select_action(
            np.zeros(PPO_OBS_SIZE, dtype=np.float32), deterministic=True
        )

    wrong_config_core = PPOCore(PPOCoreConfig(obs_size=PPO_OBS_SIZE + 1))
    with pytest.raises(ValueError, match="observation width"):
        wrong_config_core.select_action(
            np.zeros(PPO_OBS_SIZE, dtype=np.float32), deterministic=True
        )


@pytest.mark.parametrize(
    ("direction", "bar_open", "bar_high", "bar_low", "expected_mid"),
    [
        pytest.param("long", 101.0, 103.0, 99.0, 100.0, id="long-touch"),
        pytest.param("long", 98.0, 99.0, 97.0, 98.0, id="long-gap-through"),
        pytest.param("short", 99.0, 101.0, 98.0, 100.0, id="short-touch"),
        pytest.param("short", 102.0, 103.0, 101.0, 102.0, id="short-gap-through"),
    ],
)
def test_intrabar_stop_triggers_at_stop_or_worse_gap_open_and_executes(
    risk_env: PropFirmTradingEnv,
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    expected_mid: float,
) -> None:
    entry_price = 101.0 if direction == "long" else 99.0
    position = PropPosition(
        instrument="XAUUSD",
        direction=direction,
        entry_price=entry_price,
        entry_dt=None,
        entry_bar=risk_env.episode_bars,
        lot_size=1.0,
        initial_risk_eur=100.0,
        stop_price=100.0,
        stop_distance_price=1.0,
    )
    risk_env.position = position

    monkeypatch.setattr(
        risk_env,
        "_get_ohlcv",
        lambda *_args, **_kwargs: {
            "open": np.asarray([bar_open]),
            "high": np.asarray([bar_high]),
            "low": np.asarray([bar_low]),
            "close": np.asarray([bar_open]),
        },
    )

    stop_mid = risk_env._intrabar_stop_mid(position)
    assert stop_mid is not None
    assert stop_mid == pytest.approx(expected_mid)

    result = risk_env._close_position_now(
        reason=CloseReason.HARD_STOP.value,
        dt=None,
        mid=float(stop_mid),
        vol_proxy=0.0,
    )
    assert result.close_reason is CloseReason.HARD_STOP
    assert result.net_pnl < 0.0
    assert risk_env.position is None
    assert risk_env.total_trades == 1


def test_terminal_breach_forfeits_all_prior_positive_episode_reward(
    risk_env: PropFirmTradingEnv,
) -> None:
    risk_env.config.firm_daily_drawdown_limit = 0.50
    risk_env.balance = 89_000.0
    risk_env.equity = 89_000.0
    risk_env._episode_reward_total = 100.0
    risk_env._episode_return = 100.0

    _, reward, terminated, truncated, info = risk_env.step(0)

    assert terminated
    assert not truncated
    assert info["termination_reason"] == "max_drawdown_breach"
    assert reward <= -125.0
    assert risk_env._episode_reward_total <= -25.0
    assert risk_env._episode_return <= -25.0


def test_live_state_host_preserves_open_position_and_stop_risk(market_data: dict) -> None:
    host = LiveStateHost("XAUUSD")
    host.update(_live_frames(market_data))
    host.sync_account(
        balance=100_500.0,
        equity=100_600.0,
        initial_balance=100_000.0,
        day_start_balance=100_200.0,
        peak_balance=101_000.0,
        daily_trades=2,
        consecutive_losses=1,
        position={
            "instrument": "XAUUSD",
            "side": 1,
            "entry_price": 4_000.0,
            "lots": 0.50,
            "sl": 3_990.0,
            "open_time": "2026-08-03T06:00:00Z",
            "contract_size": 100.0,
        },
    )

    assert host.env.position is not None
    assert host.env.position.direction == "long"
    assert host.env.position.initial_risk_eur == pytest.approx(500.0)
    inputs = host.observation_inputs()
    assert inputs["account_state"]["has_position"] is True
    assert inputs["risk_state"]["portfolio_risk"]["total_exposure"] > 0.0


def test_live_state_host_rejects_stopless_open_position(market_data: dict) -> None:
    host = LiveStateHost("XAUUSD")
    host.update(_live_frames(market_data))
    with pytest.raises(ValueError, match="entry/size/stop"):
        host.sync_account(
            balance=100_000.0,
            equity=100_000.0,
            initial_balance=100_000.0,
            day_start_balance=100_000.0,
            peak_balance=100_000.0,
            position={"instrument": "XAUUSD", "side": 1, "entry_price": 4_000.0, "lots": 0.5, "sl": 0.0},
        )
