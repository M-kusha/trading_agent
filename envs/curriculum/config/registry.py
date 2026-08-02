

from __future__ import annotations

from typing import Dict, Optional, Set

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

    "avg_bars_between_trades",
    "setup_skipped_per_episode",
    "entry_certainty_avg",
    "avg_setup_quality",
    "fomo_trade_rate",
    "revenge_trade_rate",
}

OBSERVED_ALIASES: Dict[str, str] = {
    "drawdown": "max_drawdown",
    "max_avg_drawdown": "max_drawdown",
    "dd": "max_drawdown",
    "pf": "profit_factor",
    "winrate": "win_rate",
    "r_mult": "r_multiple",
    "entry_certainty": "entry_certainty_avg",
    "setup_skipped": "setup_skipped_per_episode",
}


THRESHOLD_FIELDS: Set[str] = {
    "min_win_rate",
    "min_profit_factor",
    "min_avg_pnl",
    "min_avg_r_multiple",
    "max_avg_drawdown",
    "max_dd_breach_rate",
    "max_consecutive_loss_rate",
    "max_win_rate_std",
    "min_trades_per_episode_for_win_rate_stability",
    "max_win_rate_wilson_width",
    "max_pnl_std",
    "min_trade_count_avg",
    "min_entropy",
    "max_mask_collapse_rate",
    "max_stop_mode_rate",

    "min_avg_bars_between_trades",
    "min_setup_skipped_per_episode",
    "min_entry_certainty_avg",
    "min_avg_setup_quality",
    "max_fomo_trade_rate",
    "max_revenge_trade_rate",
    "consistency_streak_required",
}

THRESHOLD_ALIASES: Dict[str, str] = {
    "max_drawdown": "max_avg_drawdown",
    "min_r_multiple": "min_avg_r_multiple",
}


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
    "avg_bars_between_trades": "min_avg_bars_between_trades",
    "setup_skipped_per_episode": "min_setup_skipped_per_episode",
    "entry_certainty_avg": "min_entry_certainty_avg",
    "avg_setup_quality": "min_avg_setup_quality",
    "fomo_trade_rate": "max_fomo_trade_rate",
    "revenge_trade_rate": "max_revenge_trade_rate",
}


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


METRIC_CANONICAL_NAMES: Set[str] = OBSERVED_METRICS | THRESHOLD_FIELDS


def canonicalize_observed_metric(name: str) -> str:
    lower = name.lower()
    return OBSERVED_ALIASES.get(lower, lower)


def canonicalize_threshold_field(name: str) -> str:
    lower = name.lower()
    return THRESHOLD_ALIASES.get(lower, lower)


def canonicalize_metric(name: str) -> str:
    lower = name.lower()
    if lower in OBSERVED_METRICS or lower in OBSERVED_ALIASES:
        return canonicalize_observed_metric(lower)
    if lower in THRESHOLD_FIELDS or lower in THRESHOLD_ALIASES:
        return canonicalize_threshold_field(lower)
    return lower


def canonicalize_composite_key(name: str) -> str:
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
    "COMPOSITE_HARD_FLOOR_KEYS",
    "COMPOSITE_KEY_ALIASES",
    "COMPOSITE_WEIGHT_KEYS",
    "METRIC_CANONICAL_NAMES",
    "METRIC_TO_THRESHOLD",
    "OBSERVED_ALIASES",
    "OBSERVED_METRICS",
    "THRESHOLD_ALIASES",
    "THRESHOLD_FIELDS",
    "canonicalize_composite_key",
    "canonicalize_metric",
    "canonicalize_observed_metric",
    "canonicalize_threshold_field",
    "get_threshold_for_metric",
    "is_valid_observed_metric",
    "is_valid_threshold_field",
]
