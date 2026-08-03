from __future__ import annotations

import numpy as np
import pytest

from envs.core.execution_model import ExecutionConfig, ExecutionModel


def test_deterministic_cost_mode_keeps_expected_costs_but_removes_rng_noise() -> None:
    config = ExecutionConfig(
        deterministic_costs=True,
        base_spread_points=0.20,
        spread_mult_range=(0.8, 1.4),
        slippage_points_sigma=0.05,
        slippage_mult_range=(0.6, 1.8),
        spread_shock_enabled=False,
        rejection_enabled=False,
    )
    first = ExecutionModel(config, np.random.default_rng(1))
    second = ExecutionModel(config, np.random.default_rng(999))

    first.set_episode_randomization(spread_mult=1.5, slippage_mult=2.0)
    second.set_episode_randomization(spread_mult=1.5, slippage_mult=2.0)
    spread_a = first._compute_spread_points(vol_proxy=0.4, lot_size=2.0)
    spread_b = second._compute_spread_points(vol_proxy=0.4, lot_size=2.0)
    slip_a = first._compute_slippage_points(vol_proxy=0.4, lot_size=2.0)
    slip_b = second._compute_slippage_points(vol_proxy=0.4, lot_size=2.0)

    assert spread_a == pytest.approx(spread_b)
    assert slip_a == pytest.approx(slip_b)
    assert spread_a > 0.0
    assert slip_a > 0.0


def test_recorded_spread_and_stress_multiplier_remain_visible_in_deterministic_mode() -> None:
    config = ExecutionConfig(deterministic_costs=True)
    model = ExecutionModel(config, np.random.default_rng(3))
    model.set_episode_randomization(spread_mult=2.0, slippage_mult=1.0)

    bid, ask, spread = model.quote(
        mid=2_000.0,
        vol_proxy=0.3,
        lot_size=1.0,
        data_spread=0.35,
    )

    assert spread == pytest.approx(0.70)
    assert ask - bid == pytest.approx(0.70)
