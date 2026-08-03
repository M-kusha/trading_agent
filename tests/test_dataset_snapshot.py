from __future__ import annotations

import copy
from typing import cast

import numpy as np
import pandas as pd
import pytest

from train.train_prop_firm import (
    OPTUNA_OBJECTIVE_VERSION,
    _normalize_market_frame,
    build_dataset_manifest,
    validate_bound_optuna_params,
)


def _frame(rows: int = 32) -> pd.DataFrame:
    time = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    close = 2_000.0 + np.arange(rows, dtype=np.float64)
    return pd.DataFrame(
        {
            "time": time,
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": np.full(rows, 100.0),
            "spread": np.full(rows, 35.0),
        }
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda f: f.__setitem__("open", np.nan), "non-finite"),
        (lambda f: f.__setitem__("spread", -1.0), "negative"),
        (lambda f: f.__setitem__("high", f["low"] - 1.0), "OHLC envelope"),
        (lambda f: f.__setitem__("time", list(f["time"][:-1]) + [f["time"].iloc[-2]]), "duplicate"),
    ],
)
def test_market_substrate_fails_closed_instead_of_being_zero_filled(mutate, match):
    frame = _frame()
    mutate(frame)
    with pytest.raises(ValueError, match=match):
        _normalize_market_frame(frame, source="fixture", require_spread=True)


def test_dataset_fingerprint_is_stable_and_content_sensitive():
    data = {"XAUUSD": {"M15": _frame()}}
    first = build_dataset_manifest(data, role="test")
    second = build_dataset_manifest(data, role="test")
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert first["training_code_sha256"] == second["training_code_sha256"]

    changed = {"XAUUSD": {"M15": _frame()}}
    changed_frame = changed["XAUUSD"]["M15"]
    prior_close = cast(float, changed_frame.loc[3, "close"])
    changed_frame.loc[3, "close"] = prior_close + 0.01
    third = build_dataset_manifest(changed, role="test")
    assert third["dataset_fingerprint"] != first["dataset_fingerprint"]


def test_optuna_params_require_matching_dataset_code_and_objective():
    manifest = build_dataset_manifest({"XAUUSD": {"M15": _frame()}}, role="optuna_development")
    metadata = {
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "training_code_sha256": manifest["training_code_sha256"],
        "objective_version": OPTUNA_OBJECTIVE_VERSION,
    }
    payload = {"params": {"learning_rate": 1e-4}, "metadata": metadata}
    assert validate_bound_optuna_params(payload, manifest) == payload["params"]

    for key in metadata:
        bad = copy.deepcopy(payload)
        bad["metadata"][key] = "wrong"
        with pytest.raises(ValueError, match=key):
            validate_bound_optuna_params(bad, manifest)

    with pytest.raises(ValueError, match="unbound"):
        validate_bound_optuna_params({"learning_rate": 1e-4}, manifest)
