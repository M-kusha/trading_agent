import os
from datetime import datetime, timezone

import numpy as np


def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


_LAYER_W = dict(
    liquidityheatmaplayer=2.0,
    lhl=2.0,
    fractalregimeconfirmation=1.5,
    frc=1.5,
    mtd=1.0,
    markerregimeswitcher=1.0,
    switcher=1.0,

    positionmanager=1.5,
    themeexpert=1.2,
    regimebiasexpert=1.3,
    seasonalityriskexpert=1.1,
    metarlexpert=1.4,
    trademonitorvetoexpert=0.8,
    dynamicriskcontroller=1.0,
)


_TRADING_MODE: str = "TRAINING"


_LIVE_PARAMS = {
    "SIG_K": 6.0,
    "SIG_KNEE": 0.25,
    "BASE_GATE": 0.30,
    "VOL_REF": 0.015,
    "VOL_MULT_EXTREME": 1.8,
    "VOL_MULT_HIGH": 1.5,
    "VOL_MULT_ELEVATED": 1.2,
    "CONSENSUS_DISCOUNT": 0.90,
}


_TRAINING_PARAMS = {
    "SIG_K": 4.0,
    "SIG_KNEE": 0.15,
    "BASE_GATE": 0.15,
    "VOL_REF": 0.02,
    "VOL_MULT_EXTREME": 1.4,
    "VOL_MULT_HIGH": 1.2,
    "VOL_MULT_ELEVATED": 1.1,
    "CONSENSUS_DISCOUNT": 0.80,
}


def set_trading_mode(mode: str) -> None:
    global _TRADING_MODE
    mode = mode.upper().strip()
    if mode not in ("LIVE", "TRAINING"):
        mode = "TRAINING"
    _TRADING_MODE = mode


def get_trading_mode() -> str:
    return _TRADING_MODE


def get_gate_params() -> dict:
    if _TRADING_MODE == "LIVE":
        return _LIVE_PARAMS.copy()
    return _TRAINING_PARAMS.copy()


def _get_sig_k() -> float:
    return get_gate_params()["SIG_K"]

def _get_sig_knee() -> float:
    return get_gate_params()["SIG_KNEE"]

def _get_base_gate() -> float:
    return get_gate_params()["BASE_GATE"]

def _get_vol_ref() -> float:
    return get_gate_params()["VOL_REF"]

def _squash(c: float) -> float:
    params = get_gate_params()
    return 1.0 / (1.0 + np.exp(-params["SIG_K"] * (c - params["SIG_KNEE"])))


def _smart_gate(volatility: float, maj: int) -> float:
    params = get_gate_params()
    gate = params["BASE_GATE"]
    vol_ref = params["VOL_REF"]


    if volatility > vol_ref * 3:
        gate *= params["VOL_MULT_EXTREME"]
    elif volatility > vol_ref * 2:
        gate *= params["VOL_MULT_HIGH"]
    elif volatility > vol_ref * 1.5:
        gate *= params["VOL_MULT_ELEVATED"]


    if abs(maj) > 0:
        gate *= params["CONSENSUS_DISCOUNT"]

    return gate


def get_gate_info() -> dict:
    return {
        "mode": _TRADING_MODE,
        "params": get_gate_params(),
        "description": "CONSERVATIVE" if _TRADING_MODE == "LIVE" else "EXPLORATORY",
    }
