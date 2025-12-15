# training/contracts/obs_contract.py
"""
Observation Contract for Training (TRAINING-ONLY)
=================================================

Defines the canonical 64-column observation schema used by prebaked signals.
Provides schema validation and extraction utilities.

Design goals:
- Exact feature naming + canonical ordering (OBS_COLUMNS)
- Fast index lookup (OBS_IDX)
- Safe extraction that cannot silently reorder incorrectly
- Optional schema/order validation for prebaked CSV integrity
- Stable schema fingerprinting for cache/version checks
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# SCHEMA VERSION - Increment when OBS_COLUMNS changes
# ═══════════════════════════════════════════════════════════════════════
OBS_SCHEMA_VERSION = "v1"


# ═══════════════════════════════════════════════════════════════════════
# CANONICAL 64-COLUMN OBSERVATION SCHEMA
# Must match scripts/prebake_signals.py and envs/exploration_env.py exactly
# ═══════════════════════════════════════════════════════════════════════
OBS_COLUMNS: List[str] = [
    # M15 Price Features [0-9]
    "m15_price_norm",
    "m15_return_1",
    "m15_return_5",
    "m15_return_20",
    "m15_volatility",
    "m15_rsi",
    "m15_atr_norm",
    "m15_trend_strength",
    "m15_momentum",
    "m15_volume_ratio",
    # HTF Context [10-15]
    "htf_h1_trend",
    "htf_h4_trend",
    "htf_d1_trend",
    "htf_alignment",
    "htf_volatility_ratio",
    "htf_context_strength",
    # Voting/Expert Signals [16-23]
    "expert_theme_signal",
    "expert_trend_signal",
    "expert_momentum_signal",
    "expert_seasonality_signal",
    "expert_5",
    "expert_6",
    "expert_7",
    "expert_8",
    # Committee Consensus [24-31]
    "consensus_direction",
    "consensus_confidence",
    "consensus_agreement",
    "committee_bullish_count",
    "committee_bearish_count",
    "committee_neutral_count",
    "committee_strength",
    "committee_clarity",
    # Risk/Memory [32-39]
    "risk_level",
    "risk_gate",
    "memory_gate",
    "danger_zone_count",
    "recent_loss_streak",
    "drawdown_risk",
    "exposure_risk",
    "risk_adjusted_size",
    # Account State [40-47]
    "balance_norm",
    "equity_norm",
    "margin_used",
    "drawdown_pct",
    "position_count",
    "position_exposure",
    "daily_pnl_norm",
    "win_rate",
    # World Model Predictions [48-55]
    "wm_price_pred",
    "wm_volatility_pred",
    "wm_trend_pred",
    "wm_confidence",
    "wm_scenario_bull",
    "wm_scenario_bear",
    "wm_scenario_range",
    "wm_uncertainty",
    # Trading Mode [56-63]
    "mode_intensity",
    "mode_entry_allowed",
    "mode_entry_quality",
    "mode_risk_mult",
    "timing_score",
    "session_quality",
    "liquidity_score",
    "mode_effectiveness",
]

# Derived constants
OBS_SIZE = len(OBS_COLUMNS)
if OBS_SIZE != 64:
    raise ValueError(f"OBS_SIZE must be 64, got {OBS_SIZE}")

# Defensive: no duplicates
_dups = [c for c in set(OBS_COLUMNS) if OBS_COLUMNS.count(c) > 1]
if _dups:
    raise ValueError(f"OBS_COLUMNS contains duplicates: {_dups}")

# Index mapping for fast access
OBS_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(OBS_COLUMNS)}
if len(OBS_IDX) != OBS_SIZE:
    raise ValueError("OBS_IDX mapping size mismatch; check OBS_COLUMNS uniqueness")


# ═══════════════════════════════════════════════════════════════════════
# FEATURE GROUP INDICES (for targeted access)
# ═══════════════════════════════════════════════════════════════════════
FEATURE_GROUPS: Dict[str, Tuple[int, int]] = {
    "m15_price": (0, 10),        # M15 price features
    "htf_context": (10, 16),     # Higher timeframe context
    "expert_signals": (16, 24),  # Voting/expert signals
    "committee": (24, 32),       # Committee consensus
    "risk_memory": (32, 40),     # Risk and memory gates
    "account": (40, 48),         # Account state (overwritten live)
    "world_model": (48, 56),     # World model predictions
    "trading_mode": (56, 64),    # Trading mode features
}

# Account slot indices (these are overwritten with live values during training)
ACCOUNT_SLOTS: Dict[str, int] = {
    "balance_norm": OBS_IDX["balance_norm"],  # 40
    "equity_norm": OBS_IDX["equity_norm"],  # 41
    "margin_used": OBS_IDX["margin_used"],  # 42
    "drawdown_pct": OBS_IDX["drawdown_pct"],  # 43
    "position_count": OBS_IDX["position_count"],  # 44
    "position_exposure": OBS_IDX["position_exposure"],  # 45
    "daily_pnl_norm": OBS_IDX["daily_pnl_norm"],  # 46
    "win_rate": OBS_IDX["win_rate"],  # 47
}

# Gate indices (used by safety filter)
GATE_SLOTS: Dict[str, int] = {
    "risk_gate": OBS_IDX["risk_gate"],  # 33
    "memory_gate": OBS_IDX["memory_gate"],  # 34
    "mode_entry_allowed": OBS_IDX["mode_entry_allowed"],  # 57
}


def schema_hash() -> str:
    """
    Generate SHA256 hash of schema version + column names.

    Returns:
        32-char hex string (truncated SHA256)
    """
    payload = f"{OBS_SCHEMA_VERSION}:" + ",".join(OBS_COLUMNS)
    full_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return full_hash[:32]


def validate_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame has all required observation columns.

    Returns:
        (is_valid, missing_columns)
    """
    missing = [col for col in OBS_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def validate_df_order(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """
    Optional integrity check: verify that the OBS_COLUMNS appear in the DataFrame
    in the same relative order as the canonical schema.

    Note:
        Extraction via df[OBS_COLUMNS] will always reorder correctly, so this check
        is mainly for detecting prebaker drift / inconsistent file formats.
    """
    present = [c for c in df.columns if c in OBS_IDX]
    if not present:
        return False, "No observation columns found in DataFrame"

    expected = [c for c in OBS_COLUMNS if c in set(present)]
    if present != expected:
        return False, "Observation columns are not in canonical order in the source DataFrame"
    return True, None


def extract_obs_matrix(df: pd.DataFrame, validate: bool = True) -> np.ndarray:
    """
    Extract observation matrix from DataFrame in canonical column order.

    Behavior:
      - validate=True: requires all columns and raises on missing.
      - validate=False: missing columns are filled with zeros (explicitly),
        and present columns are extracted in canonical order.

    Returns:
        (N, 64) float32 matrix with NaN/inf sanitized.
    """
    if validate:
        ok, missing = validate_df(df)
        if not ok:
            raise ValueError(
                f"DataFrame missing {len(missing)} observation columns. "
                f"First 5: {missing[:5]}"
            )
        matrix = df[OBS_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    else:
        # Fill missing columns with zeros rather than raising KeyError.
        n = len(df)
        matrix = np.zeros((n, OBS_SIZE), dtype=np.float32)
        for j, col in enumerate(OBS_COLUMNS):
            if col in df.columns:
                # Convert per-column to float32; non-numeric becomes NaN -> sanitized below.
                matrix[:, j] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32, copy=False)

    # Sanitize numeric issues (keep bounded, deterministic values)
    # Use in-place conversion when possible to reduce allocations.
    np.nan_to_num(matrix, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    return matrix


def validate_obs_matrix(arr: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Validate observation matrix for correctness.

    Requirements:
      - 2D array
      - shape (N, 64)
      - dtype float32
      - all finite
    """
    if not isinstance(arr, np.ndarray):
        return False, f"Expected np.ndarray, got {type(arr)}"

    if arr.ndim != 2:
        return False, f"Expected 2D array, got {arr.ndim}D"

    if arr.shape[1] != OBS_SIZE:
        return False, f"Expected {OBS_SIZE} columns, got {arr.shape[1]}"

    if arr.dtype != np.float32:
        return False, f"Expected float32, got {arr.dtype}"

    if not np.all(np.isfinite(arr)):
        non_finite = int((~np.isfinite(arr)).sum())
        return False, f"Found {non_finite} non-finite values"

    return True, None


def get_obs_index(name: str) -> int:
    """Get index for observation column by name."""
    try:
        return OBS_IDX[name]
    except KeyError as e:
        raise KeyError(f"Unknown observation column: {name}") from e


def describe_schema() -> str:
    """Return human-readable schema description."""
    lines = [
        f"Observation Schema {OBS_SCHEMA_VERSION}",
        f"Hash: {schema_hash()}",
        f"Size: {OBS_SIZE} columns",
        "",
        "Feature Groups:",
    ]

    for group_name, (start, end) in FEATURE_GROUPS.items():
        cols = OBS_COLUMNS[start:end]
        preview = ", ".join(cols[:3]) + ("..." if len(cols) > 3 else "")
        lines.append(f"  [{start:2d}-{end-1:2d}] {group_name}: {preview}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test
    print(describe_schema())
    print(f"\nSchema hash: {schema_hash()}")
    print(f"\nAccount slots: {ACCOUNT_SLOTS}")
    print(f"Gate slots: {GATE_SLOTS}")
