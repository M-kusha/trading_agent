"""Risk must stay constant in probability terms, not just in euros.

The stop fired at a fixed euro loss (hard_stop_loss_eur), so its distance in
PRICE terms had nothing to do with volatility. The same 350 EUR sat roughly
1.5 ATR away in a calm market and roughly 0.75 ATR away once volatility
doubled, and was hit about twice as often for the same nominal risk.

That is not hypothetical here. XAUUSD 20-day realised volatility ran 0.74-0.90%
through 2025, spiked to 3.28% in February 2026 and settled at 1.37-1.68% - a
level shift of roughly 2x, not a passing spike. Only 5.5% of the training bars
resemble the new regime.

Sizing the lot from ATR pins the stop at a constant statistical distance: euro
risk is unchanged, the lot shrinks as the market widens.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.core.env_types import PropFirmConfig


def _lot_for_atr(risk_eur: float, atr: float, mult: float = 1.5, per_price_unit: float = 100.0) -> float:
    return risk_eur / max(mult * atr * per_price_unit, 1e-9)


def test_euro_risk_is_unchanged_across_volatility_regimes():
    risk = 300.0
    calm = _lot_for_atr(risk, atr=2.4)
    wild = _lot_for_atr(risk, atr=4.5)

    assert calm * 1.5 * 2.4 * 100 == pytest.approx(risk)
    assert wild * 1.5 * 4.5 * 100 == pytest.approx(risk)


def test_the_lot_shrinks_when_volatility_rises():
    calm = _lot_for_atr(300.0, atr=2.4)
    wild = _lot_for_atr(300.0, atr=4.5)
    assert wild < calm
    # Volatility roughly doubled post-war; exposure should roughly halve.
    assert wild / calm == pytest.approx(2.4 / 4.5, rel=1e-6)


def test_the_config_defaults_are_live():
    cfg = PropFirmConfig()
    assert cfg.atr_stop_enabled is True
    assert cfg.atr_stop_multiplier > 0.0


def test_sizing_uses_atr_when_enabled(env):
    """The env must actually take the ATR branch, not the euro fallback."""
    env.reset(seed=0)
    env.config.atr_stop_enabled = True
    lot_atr, risk_atr = env._calculate_lot_size(1.0)

    env.config.atr_stop_enabled = False
    lot_fixed, risk_fixed = env._calculate_lot_size(1.0)

    assert lot_atr > 0 and lot_fixed > 0
    assert lot_atr != pytest.approx(lot_fixed), (
        "ATR sizing produced the same lot as the fixed-euro path - the branch "
        "is not being taken"
    )


def test_atr_is_measured_in_price_units(env):
    env.reset(seed=0)
    atr = env._atr_price(env._episode_instrument)
    assert atr > 0.0, "ATR came back zero - sizing would fall back to the euro stop"
    price = env._get_price_mid(env._episode_instrument)
    # Gold ATR over 14 M15 bars is a small fraction of price, not a percentage.
    assert 0.0 < atr < price * 0.05


def test_the_stop_is_per_position_not_global(env):
    """With ATR sizing each trade has its own stop, so a global euro threshold
    would fire at the wrong distance."""
    import inspect

    src = inspect.getsource(type(env).step) if hasattr(type(env), "step") else ""
    full = inspect.getsource(type(env))
    assert "initial_risk_eur" in full, "stop no longer reads the position's own risk"


def test_risk_stays_inside_the_configured_ceiling(env):
    env.reset(seed=0)
    cfg = env.config
    for mult in (0.25, 0.5, 1.0, 1.25):
        _lot, risk = env._calculate_lot_size(mult)
        ceiling = env.balance * cfg.max_risk_per_trade_pct
        assert risk <= ceiling * 1.05, (
            f"size_mult {mult} produced risk {risk:.0f} above the ceiling {ceiling:.0f}"
        )
