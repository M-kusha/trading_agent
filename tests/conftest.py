from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "processed"
TIMEFRAMES = ("M15", "H1", "H4", "D1")


def _load_instrument(instrument: str) -> dict:
    out: dict = {}
    if not DATA_DIR.is_dir():
        return out
    for path in sorted(DATA_DIR.glob("*.csv")):
        parts = path.stem.replace("_features", "").split("_")
        if len(parts) < 2:
            continue
        tf, inst = parts[-1].upper(), "_".join(parts[:-1])
        if inst != instrument or tf not in TIMEFRAMES:
            continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            continue
        if "volume" not in df.columns:
            df["volume"] = 1.0
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.sort_values("time").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)
        out[tf] = df
    return out


@pytest.fixture(scope="session")
def market_data() -> dict:
    data = _load_instrument("XAUUSD")
    if not data:
        pytest.skip("data/processed CSVs not available")
    missing = [tf for tf in TIMEFRAMES if tf not in data]
    if missing:
        pytest.skip(f"missing timeframes: {missing}")
    return {"XAUUSD": data}


@pytest.fixture(scope="session")
def env(market_data):
    from envs.core.env_types import PropFirmConfig
    from envs.prop_firm_env import PropFirmTradingEnv

    return PropFirmTradingEnv(market_data, PropFirmConfig())


@pytest.fixture(scope="session")
def rollout(env) -> np.ndarray:
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    collected = [obs.copy()]
    for _ in range(400):
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        action = int(rng.choice(valid)) if valid.size else 0
        obs, _reward, terminated, truncated, _info = env.step(action)
        collected.append(obs.copy())
        if terminated or truncated:
            obs, _ = env.reset()
            collected.append(obs.copy())
    return np.asarray(collected, dtype=np.float64)
