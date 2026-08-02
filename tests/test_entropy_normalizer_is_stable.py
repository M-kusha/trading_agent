"""Normalised entropy must not swing on a single mask sample.

get_ent_coef receives current_entropy - SB3's rollout MEAN policy entropy,
averaged over thousands of steps - and n_valid_actions, a single instantaneous
action-mask read taken at callback time. The valid action count genuinely
differs by state: flat allows hold plus every size bucket long and short, while
holding a position allows only hold and close.

Normalising a batch mean by a point sample therefore made the same raw entropy
report as 0.42 or 0.67 depending only on whether a position happened to be open
at the instant the callback ran, and the PID reacted to the swing. A live run
raised ent_coef 0.100 -> 0.130 -> 0.169 across three consecutive updates while
raw entropy sat between 0.925 and 0.939.

The EMA over valid-action counts is the denominator that matches the numerator.
"""

from __future__ import annotations

import numpy as np
import pytest

from train.controllers.entropy_controller import SmartEntropyController


def _controller(n_actions: int = 9) -> SmartEntropyController:
    return SmartEntropyController(n_actions=n_actions)


def test_normalisation_does_not_track_the_instantaneous_sample():
    """Alternating flat/in-position samples must not swing the normaliser."""
    ctrl = _controller()
    raw_entropy = 0.93

    norms = []
    for i in range(40):
        n_valid = 9 if i % 2 == 0 else 2
        ctrl.get_ent_coef(
            current_entropy=raw_entropy,
            current_ent_coef=0.10,
            timesteps_elapsed=(i + 1) * 2048,
            n_valid_actions=n_valid,
        )
        norms.append(ctrl._normalize_entropy(raw_entropy, None))

    # Once the EMA settles the reading must be steady, not oscillating with the
    # parity of the sample.
    tail = np.asarray(norms[-10:])
    assert tail.std() < 0.02, f"normalised entropy still swinging: {tail}"


def test_a_single_sample_no_longer_flips_the_reading():
    ctrl = _controller()
    for _ in range(30):
        ctrl.update_valid_actions_estimate(9)

    before = ctrl._effective_valid_actions(None)
    ctrl.update_valid_actions_estimate(2)
    after = ctrl._effective_valid_actions(None)

    assert abs(after - before) <= 1, (
        f"one in-position sample moved the effective action count {before} -> {after}"
    )


def test_the_estimate_still_converges_to_a_persistent_change():
    """Smoothing must not mean ignoring a real shift in the action space."""
    ctrl = _controller()
    for _ in range(30):
        ctrl.update_valid_actions_estimate(9)
    assert ctrl._effective_valid_actions(None) >= 8

    for _ in range(80):
        ctrl.update_valid_actions_estimate(4)
    assert ctrl._effective_valid_actions(None) <= 5


def test_a_stationary_policy_does_not_drive_ent_coef_up():
    """The observed failure: constant entropy, rising ent_coef."""
    ctrl = _controller()
    ent_coef = 0.10
    seen = [ent_coef]

    for i in range(30):
        n_valid = 9 if i % 3 else 2
        ent_coef, _reason, applied = ctrl.get_ent_coef(
            current_entropy=1.87 * np.log(9) / np.log(9) * 0.85 * np.log(9) / np.log(9),
            current_ent_coef=ent_coef,
            timesteps_elapsed=(i + 1) * 2048,
            n_valid_actions=n_valid,
        )
        if applied:
            seen.append(ent_coef)

    # At the setpoint the controller may still nudge, but it must not ratchet
    # monotonically upward off a normaliser artefact.
    assert seen[-1] <= seen[0] * 2.0, f"ent_coef ratcheted: {seen}"


def test_normalised_entropy_is_bounded():
    ctrl = _controller()
    for _ in range(20):
        ctrl.update_valid_actions_estimate(9)
    for raw in (0.0, 0.5, 1.0, 2.0, 5.0):
        v = ctrl._normalize_entropy(raw, None)
        assert 0.0 <= v <= 1.5, f"raw {raw} -> {v}"
