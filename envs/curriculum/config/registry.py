# envs/curriculum/config/registry.py
"""
Canonical metric registry with split namespaces.

Separates OBSERVED METRICS (what evaluators produce) from THRESHOLD FIELDS
(what CompetenceThresholds contains). This prevents silent failures from
metric name mismatches.

Upgrades (Jan 2026):
- Fixed composite component key mismatch: composite scoring uses "drawdown"
  (component), not "max_drawdown" (observed metric). Added canonicalization
  for composite component keys to preserve backward compatibility.
"""

from __future__ import annotations

from typing import Dict, Optional, Set


# ---- Namespace 1: Observed Metrics ----
OBSERVED_METRICS: Set[str] = {
    "win_rate",
    "profit_factor",
    "avg_pnl",
    "r_multiple",
    "max_drawdown",
    "dd_breach_rate",
    "consecutive_loss_rate",
    "win_rate_std",
    "pnl_std",
    "trade_count_avg",
    "entropy",
    "consistency",
    "trade_activity",
}

OBSERVED_ALIASES: Dict[str, str] = {
    "drawdown": "max_drawdown",
    "max_avg_drawdown": "max_drawdown",
    "dd": "max_drawdown",
    "pf": "profit_factor",
    "winrate": "win_rate",
    "r_mult": "r_multiple",
}

# ---- Namespace 2: Threshold Fields ----
THRESHOLD_FIELDS: Set[str] = {
    "min_win_rate",
    "min_profit_factor",
    "min_avg_pnl",
    "min_avg_r_multiple",
    "max_avg_drawdown",
    "max_dd_breach_rate",
    "max_consecutive_loss_rate",
    "max_win_rate_std",
    "max_pnl_std",
    "min_trade_count_avg",
    "min_entropy",
}

THRESHOLD_ALIASES: Dict[str, str] = {
    "max_drawdown": "max_avg_drawdown",
    "min_r_multiple": "min_avg_r_multiple",
}

# ---- Bridge: Observed Metric -> Threshold Field ----
METRIC_TO_THRESHOLD: Dict[str, str] = {
    "win_rate": "min_win_rate",
    "profit_factor": "min_profit_factor",
    "avg_pnl": "min_avg_pnl",
    "r_multiple": "min_avg_r_multiple",
    "max_drawdown": "max_avg_drawdown",
    "dd_breach_rate": "max_dd_breach_rate",
    "consecutive_loss_rate": "max_consecutive_loss_rate",
    "win_rate_std": "max_win_rate_std",
    "pnl_std": "max_pnl_std",
    "trade_count_avg": "min_trade_count_avg",
    "entropy": "min_entropy",
}

# ---- Composite Scoring Component Keys ----
# These must match compute_composite_score() component names.
# NOTE: Composite uses "drawdown" (a normalized component), while observed metric is "max_drawdown".
COMPOSITE_KEY_ALIASES: Dict[str, str] = {
    "max_drawdown": "drawdown",
}

COMPOSITE_WEIGHT_KEYS: Set[str] = {
    "win_rate",
    "profit_factor",
    "drawdown",
    "consistency",
    "r_multiple",
    "dd_breach_rate",
    "trade_activity",
    "consecutive_loss_rate",
}

COMPOSITE_HARD_FLOOR_KEYS: Set[str] = {
    "win_rate",
    "drawdown",
    "dd_breach_rate",
    "profit_factor",
    "r_multiple",
}

# Legacy: Combined set for backward compatibility
METRIC_CANONICAL_NAMES: Set[str] = OBSERVED_METRICS | THRESHOLD_FIELDS


def canonicalize_observed_metric(name: str) -> str:
    lower = name.lower()
    return OBSERVED_ALIASES.get(lower, lower)


def canonicalize_threshold_field(name: str) -> str:
    lower = name.lower()
    return THRESHOLD_ALIASES.get(lower, lower)


def canonicalize_metric(name: str) -> str:
    """
    DEPRECATED: Use canonicalize_observed_metric or canonicalize_threshold_field.
    """
    lower = name.lower()
    if lower in OBSERVED_METRICS or lower in OBSERVED_ALIASES:
        return canonicalize_observed_metric(lower)
    if lower in THRESHOLD_FIELDS or lower in THRESHOLD_ALIASES:
        return canonicalize_threshold_field(lower)
    return lower


def canonicalize_composite_key(name: str) -> str:
    """Canonicalize composite component key (e.g., max_drawdown -> drawdown)."""
    lower = name.lower()
    return COMPOSITE_KEY_ALIASES.get(lower, lower)


def is_valid_observed_metric(name: str) -> bool:
    canonical = canonicalize_observed_metric(name)
    return canonical in OBSERVED_METRICS


def is_valid_threshold_field(name: str) -> bool:
    canonical = canonicalize_threshold_field(name)
    return canonical in THRESHOLD_FIELDS


def get_threshold_for_metric(metric: str) -> Optional[str]:
    canonical = canonicalize_observed_metric(metric)
    return METRIC_TO_THRESHOLD.get(canonical)


__all__ = [
    "OBSERVED_METRICS",
    "OBSERVED_ALIASES",
    "THRESHOLD_FIELDS",
    "THRESHOLD_ALIASES",
    "METRIC_TO_THRESHOLD",
    "COMPOSITE_KEY_ALIASES",
    "COMPOSITE_WEIGHT_KEYS",
    "COMPOSITE_HARD_FLOOR_KEYS",
    "METRIC_CANONICAL_NAMES",
    "canonicalize_observed_metric",
    "canonicalize_threshold_field",
    "canonicalize_metric",
    "canonicalize_composite_key",
    "is_valid_observed_metric",
    "is_valid_threshold_field",
    "get_threshold_for_metric",
]
