"""Prop-firm limits belong to one account over calendar time.

Drawdown was previously inferred from an equity curve concatenated across
independently reset episodes, so the jump from one episode's closing balance
back to the next episode's opening €100k registered as a loss that never
happened - which is how a published 13.77% drawdown was produced.

A limit cannot be recovered from a pile of resets, and a mean over reset windows
cannot pass or fail a challenge, because no single account ever lived through
that mean. The ledger walks one account; the reset windows answer a different
question and no longer carry a prop-firm label at all.
"""

from __future__ import annotations

import pandas as pd
import pytest

from evaluate.holdout_report import AccountLedger, Result


def _ledger(**over):
    kw = {"initial_balance": 100_000.0, "max_dd_limit": 0.10, "daily_dd_limit": 0.05}
    kw.update(over)
    return AccountLedger(**kw)


T0 = pd.Timestamp("2026-06-15 08:00", tz="UTC")


# ── the limits ──────────────────────────────────────────────────────────────

def test_a_healthy_account_survives():
    led = _ledger()
    for i in range(20):
        assert led.mark(T0 + pd.Timedelta(minutes=15 * i), 100_500.0) is None
    assert led.summary()["survived"] is True


def test_max_drawdown_kills_the_account():
    """Bleed across sessions so the daily rule cannot fire first.

    Losing 10% inside one session breaches the 5% daily limit on the way down,
    so isolating the maximum-loss rule means letting the daily anchor roll.
    """
    led = _ledger(rollover_hour_utc=22)
    equity = 100_000.0
    breach = None
    for day in range(6):
        session = pd.Timestamp("2026-06-15 08:00", tz="UTC") + pd.Timedelta(days=day)
        equity -= 2_000.0                      # 2% of the initial balance per day
        breach = led.mark(session, equity)
        if breach is not None:
            break

    assert breach is not None
    assert breach.kind == "max_drawdown", f"expected max_drawdown, got {breach.kind}"
    assert led.summary()["survived"] is False


def test_the_limit_is_inclusive_matching_the_environment():
    """The env breaches on >=; the ledger must not disagree by a cent."""
    led = _ledger()
    # Exactly 5% down on the day is also exactly the daily limit.
    breach = led.mark(T0, 95_000.0)
    assert breach is not None, "exactly at the limit must count as a breach"
    assert breach.drawdown_pct == pytest.approx(breach.limit_pct)


def test_daily_drawdown_kills_the_account():
    led = _ledger()
    breach = led.mark(T0, 95_000.0)
    assert breach is not None
    assert breach.kind == "daily_drawdown"


def test_only_the_first_breach_is_recorded():
    led = _ledger()
    led.mark(T0, 90_000.0)
    led.mark(T0 + pd.Timedelta(minutes=15), 80_000.0)
    assert len(led.summary()["breaches"]) == 1


# ── reset rules ─────────────────────────────────────────────────────────────

def test_the_daily_anchor_rolls_at_the_broker_hour():
    """A new session must reset the daily anchor, or yesterday's loss carries."""
    led = _ledger(rollover_hour_utc=22)
    led.mark(pd.Timestamp("2026-06-15 20:00", tz="UTC"), 100_000.0)
    led.mark(pd.Timestamp("2026-06-15 21:00", tz="UTC"), 97_000.0)

    # Past 22:00 UTC is a new trading day: the 3% is behind us.
    assert led.mark(pd.Timestamp("2026-06-15 23:00", tz="UTC"), 96_000.0) is None
    assert led.day_start == pytest.approx(96_000.0)


def test_a_loss_inside_one_session_accumulates():
    led = _ledger(rollover_hour_utc=22)
    led.mark(pd.Timestamp("2026-06-15 08:00", tz="UTC"), 100_000.0)
    breach = led.mark(pd.Timestamp("2026-06-15 15:00", tz="UTC"), 94_900.0)
    assert breach is not None and breach.kind == "daily_drawdown"


def test_static_and_trailing_anchors_differ():
    static = _ledger(trailing=False)
    trailing = _ledger(trailing=True)
    for led in (static, trailing):
        led.mark(T0, 120_000.0)

    # Back to 110k: nothing lost against the initial balance, 8.3% off the peak.
    static.mark(T0 + pd.Timedelta(minutes=15), 110_000.0)
    trailing.mark(T0 + pd.Timedelta(minutes=15), 110_000.0)

    assert static.summary()["worst_total_drawdown_pct"] == pytest.approx(0.0)
    assert trailing.summary()["worst_total_drawdown_pct"] > 8.0
    assert static.summary()["anchor"] == "initial_balance"
    assert trailing.summary()["anchor"] == "peak"


def test_the_summary_states_its_rules():
    s = _ledger().summary()
    for key in ("anchor", "rollover_hour_utc", "max_dd_limit_pct", "daily_dd_limit_pct"):
        assert key in s, f"{key} must be stated, not assumed"


# ── scope separation ────────────────────────────────────────────────────────

def test_reset_window_statistics_refuse_to_judge_a_challenge():
    r = Result(name="trained-model", return_pct=25.7, max_drawdown_pct=3.0)
    with pytest.raises(NotImplementedError, match="continuous_replay"):
        _ = r.challenge_thresholds_met


def test_reset_windows_describe_themselves_honestly():
    r = Result(name="trained-model", return_pct=2.57, episodes=10,
               max_drawdown_pct=3.0, trades=100, profit_factor=1.4)
    text = r.window_statistics_summary
    assert "per window" in text or "/window" in text
    assert "10 windows" in text


def test_profit_factor_is_not_reported_without_trades():
    r = Result(name="always-flat", episodes=10, trades=0)
    assert "n/a" in r.window_statistics_summary
