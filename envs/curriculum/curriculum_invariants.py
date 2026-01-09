# envs/curriculum_invariants.py
"""
Curriculum Invariant Checker & Anti-Gaming Guards
==================================================

Phase 1: Trade accounting invariants
Phase 2: Anti-gaming alignment checks
Phase 4: Runtime invariant validation

This module provides:
1. Trade accounting reconciliation (wins + losses == trade_count)
2. Anti-gaming companion checks for each promotion metric
3. Structured invariant violations with rate-limited logging
4. Regime-based skill decomposition helpers
"""

from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np

from envs.core.shared_utils import (
    safe_float as _sf,
    safe_int as _si,
    clamp as _clamp,
    get_envs_logger,
    iso_timestamp,
)


logger = get_envs_logger("curriculum_invariants")

def _is_finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False

def _finite_or(x: Any, default: float) -> float:
    v = _sf(x, default)
    return v if _is_finite(v) else default

def _finite_int(x: Any, default: int = 0) -> int:
    try:
        v = int(_si(x, default))
    except Exception:
        v = default
    return max(0, v)


# =============================================================================
# Invariant Violation Types
# =============================================================================

class InvariantSeverity(Enum):
    """Severity levels for invariant violations."""
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Unexpected but recoverable
    ERROR = "error"         # Needs attention, may affect results
    CRITICAL = "critical"   # Data integrity issue, stop and investigate


class InvariantType(Enum):
    """Categories of invariants."""
    TRADE_ACCOUNTING = "trade_accounting"
    METRIC_BOUNDS = "metric_bounds"
    STAGE_PROGRESSION = "stage_progression"
    COUNTER_INTEGRITY = "counter_integrity"
    NUMERIC_SANITY = "numeric_sanity"
    EXIT_DISTRIBUTION = "exit_distribution"
    ANTI_GAMING = "anti_gaming"


@dataclass
class InvariantViolation:
    """Structured representation of an invariant violation."""
    invariant_type: InvariantType
    severity: InvariantSeverity
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=iso_timestamp)
    stage_name: str = ""
    stage_epoch: int = 0
    episode_idx: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.invariant_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
            "stage_name": self.stage_name,
            "stage_epoch": self.stage_epoch,
            "episode_idx": self.episode_idx,
        }
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.invariant_type.value}: {self.message}"


# =============================================================================
# Rate-Limited Logger for Invariant Violations
# =============================================================================

class RateLimitedInvariantLogger:
    """
    Rate-limited logging for invariant violations.
    
    Prevents log spam during long training runs while ensuring violations
    are reported at least once per stage epoch.
    """
    
    # Default rate limits (seconds between same-type violations)
    DEFAULT_RATE_LIMITS: ClassVar[Dict[InvariantSeverity, float]] = {
        InvariantSeverity.INFO: 300.0,      # 5 minutes
        InvariantSeverity.WARNING: 60.0,    # 1 minute
        InvariantSeverity.ERROR: 10.0,      # 10 seconds
        InvariantSeverity.CRITICAL: 0.0,    # Always log
    }
    
    def __init__(
        self,
        rate_limits: Optional[Dict[InvariantSeverity, float]] = None,
        max_per_epoch: int = 100,
    ):
        self._rate_limits = rate_limits or self.DEFAULT_RATE_LIMITS
        self._max_per_epoch = max_per_epoch
        
        # Track last log time per (type, severity, stage_name, stage_epoch) tuple
        # Including stage_name prevents cross-stage suppression when epoch resets
        self._last_log_time: Dict[Tuple[str, str, str, int], float] = {}
        
        # Count per (stage_name, epoch) for overflow protection
        self._epoch_counts: Dict[Tuple[str, int], int] = defaultdict(int)
        
        # Accumulated violations (for batch export) - keep most recent
        self._max_stored = 1000
        self._violations: deque[InvariantViolation] = deque(maxlen=self._max_stored)
    
    def log(self, violation: InvariantViolation) -> bool:
        """
        Log a violation if rate limit allows.
        
        Returns True if logged, False if suppressed.
        """
        # Include stage_name to prevent cross-stage suppression
        key = (
            violation.invariant_type.value,
            violation.severity.value,
            violation.stage_name,
            violation.stage_epoch,
        )
        
        # Epoch key includes stage_name
        epoch_key = (violation.stage_name, violation.stage_epoch)
        
        now = time.time()
        rate_limit = self._rate_limits.get(violation.severity, 60.0)
        
        # Check rate limit
        last_time = self._last_log_time.get(key, 0.0)
        if now - last_time < rate_limit:
            return False
        
        # Check epoch overflow
        if self._epoch_counts[epoch_key] >= self._max_per_epoch:
            return False
        
        # Log it
        self._last_log_time[key] = now
        self._epoch_counts[epoch_key] += 1
        
        # Store for batch export
        self._violations.append(violation)
        
        # Actual logging
        log_msg = str(violation)
        if violation.context:
            log_msg += f" | context={violation.context}"
        
        if violation.severity == InvariantSeverity.CRITICAL:
            logger.error(log_msg)
        elif violation.severity == InvariantSeverity.ERROR:
            logger.error(log_msg)
        elif violation.severity == InvariantSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        return True
    
    def reset_epoch(self, epoch: int, stage_name: str = "") -> None:
        """Reset counters for a new epoch within a stage."""
        epoch_key = (stage_name, epoch)
        self._epoch_counts[epoch_key] = 0
        
        # Clean up old epoch data for this stage
        old_keys = [
            k for k in self._epoch_counts 
            if k[0] == stage_name and k[1] < epoch - 1
        ]
        for k in old_keys:
            del self._epoch_counts[k]
        
        # Bounded cleanup of _last_log_time to prevent unbounded growth
        # 1) remove entries for same stage older than last 2 epochs
        old_stage_epoch_keys = [
            k for k in self._last_log_time.keys()
            if k[2] == stage_name and k[3] < epoch - 1
        ]
        for k in old_stage_epoch_keys:
            del self._last_log_time[k]

        # 2) keep only recent entries for other stages (last 10 min)
        cutoff_time = time.time() - 600  # 10 minutes
        stale_keys = [
            k for k, t in self._last_log_time.items()
            if k[2] != stage_name and t < cutoff_time  # k[2] is stage_name
        ]
        for k in stale_keys:
            del self._last_log_time[k]
    
    def get_violations(self, clear: bool = False) -> List[Dict[str, Any]]:
        """Get accumulated violations as dicts."""
        result = [v.to_dict() for v in self._violations]
        if clear:
            self._violations.clear()
        return result
    
    def get_summary(self) -> Dict[str, int]:
        """Get violation counts by type and severity."""
        summary: Dict[str, int] = defaultdict(int)
        for v in self._violations:
            key = f"{v.invariant_type.value}_{v.severity.value}"
            summary[key] += 1
        return dict(summary)


# =============================================================================
# Trade Accounting Reconciler
# =============================================================================

@dataclass
class TradeAccountingResult:
    """Result of trade accounting reconciliation."""
    original_trade_count: int
    original_wins: int
    original_losses: int
    
    reconciled_trade_count: int
    reconciled_wins: int
    reconciled_losses: int
    
    was_reconciled: bool = False
    reconciliation_method: str = ""
    violation: Optional[InvariantViolation] = None
    
    @property
    def pooled_trades(self) -> int:
        """Canonical denominator for win stats: wins + losses."""
        return self.reconciled_wins + self.reconciled_losses


def reconcile_trade_accounting(
    trade_count: int,
    winning_trades: int,
    losing_trades: int,
    win_rate: float,
    stage_name: str = "",
    stage_epoch: int = 0,
    episode_idx: int = 0,
) -> TradeAccountingResult:
    """
    Reconcile trade accounting to ensure wins + losses == trade_count.
    
    Rules:
    1. If wins + losses > trade_count: trust wins + losses (more specific)
    2. If wins + losses < trade_count and both are 0: derive from win_rate
    3. If wins + losses < trade_count but non-zero: trust wins + losses
       (some trades may have no clear outcome, e.g., scratch trades)
    4. If wins + losses == trade_count: no reconciliation needed
    
    Returns:
        TradeAccountingResult with reconciled values and any violations
    """
    # Sanitize inputs (defensive: avoid negative and NaN/Inf)
    trade_count = _finite_int(trade_count, 0)
    winning_trades = _finite_int(winning_trades, 0)
    losing_trades = _finite_int(losing_trades, 0)
    win_rate = _finite_or(win_rate, 0.0)
    win_rate = float(_clamp(win_rate, 0.0, 1.0))

    original = TradeAccountingResult(
        original_trade_count=trade_count,
        original_wins=winning_trades,
        original_losses=losing_trades,
        reconciled_trade_count=trade_count,
        reconciled_wins=winning_trades,
        reconciled_losses=losing_trades,
    )
    
    computed_sum = winning_trades + losing_trades
    
    # Case 1: Perfect match
    if computed_sum == trade_count:
        return original
    
    # Case 2: wins + losses > trade_count (inconsistency)
    if computed_sum > trade_count:
        original.reconciled_trade_count = computed_sum
        original.reconciled_wins = winning_trades
        original.reconciled_losses = losing_trades
        original.was_reconciled = True
        original.reconciliation_method = "trust_wins_losses_sum"
        original.violation = InvariantViolation(
            invariant_type=InvariantType.TRADE_ACCOUNTING,
            severity=InvariantSeverity.WARNING,
            message=f"wins({winning_trades})+losses({losing_trades})={computed_sum} > trade_count({trade_count})",
            context={
                "original_trade_count": trade_count,
                "computed_sum": computed_sum,
                "method": "trust_wins_losses_sum",
            },
            stage_name=stage_name,
            stage_epoch=stage_epoch,
            episode_idx=episode_idx,
        )
        return original
    
    # Case 3: wins + losses == 0 but trade_count > 0 (derive from win_rate)
    if computed_sum == 0 and trade_count > 0:
        approx_wins = int(round(win_rate * trade_count))
        approx_losses = trade_count - approx_wins
        original.reconciled_wins = max(0, min(trade_count, approx_wins))
        original.reconciled_losses = max(0, approx_losses)
        original.was_reconciled = True
        original.reconciliation_method = "derive_from_win_rate"
        # Info level - this is expected when win/loss breakdown not available
        original.violation = InvariantViolation(
            invariant_type=InvariantType.TRADE_ACCOUNTING,
            severity=InvariantSeverity.INFO,
            message=f"Derived wins/losses from win_rate({win_rate:.2f}) for {trade_count} trades",
            context={
                "trade_count": trade_count,
                "win_rate": win_rate,
                "derived_wins": original.reconciled_wins,
                "derived_losses": original.reconciled_losses,
            },
            stage_name=stage_name,
            stage_epoch=stage_epoch,
            episode_idx=episode_idx,
        )
        return original
    
    # Case 4: wins + losses < trade_count with non-zero values
    # This can happen with scratch trades or incomplete data
    if computed_sum < trade_count:
        # Keep wins/losses as-is, but note the discrepancy
        # Don't inflate wins/losses to match trade_count
        original.reconciled_trade_count = computed_sum  # Use pooled as canonical
        original.was_reconciled = True
        original.reconciliation_method = "use_pooled_denominator"
        gap = trade_count - computed_sum
        if gap > trade_count * 0.2:  # More than 20% unaccounted
            original.violation = InvariantViolation(
                invariant_type=InvariantType.TRADE_ACCOUNTING,
                severity=InvariantSeverity.WARNING,
                message=f"{gap} trades unaccounted ({gap/trade_count*100:.1f}%)",
                context={
                    "trade_count": trade_count,
                    "wins": winning_trades,
                    "losses": losing_trades,
                    "unaccounted": gap,
                },
                stage_name=stage_name,
                stage_epoch=stage_epoch,
                episode_idx=episode_idx,
            )
        return original
    
    return original


# =============================================================================
# Anti-Gaming Checks
# =============================================================================

@dataclass
class AntiGamingCheckResult:
    """Result of an anti-gaming check."""
    metric_name: str
    gaming_detected: bool
    gaming_score: float  # 0.0 = no gaming, 1.0 = definite gaming
    explanation: str
    counter_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


class AntiGamingChecker:
    """
    Anti-gaming companion checks for promotion metrics.
    
    For each key metric that can be gamed, provides counter-metrics
    that detect degenerate strategies.
    
    Metric -> Gaming Strategy -> Counter-Metric:
    - High profit_factor -> trade rarely, avoid losses -> min_trade_activity + opportunity_cost
    - Low drawdown -> tiny positions -> expected_return_per_risk + min_meaningful_risk
    - High win_rate -> cut winners early -> r_multiple + time_in_trade_efficiency
    - High R-multiple -> let losses run -> loss_management + max_consecutive_losses
    - High consistency -> avoid difficult conditions -> regime_coverage
    """
    
    # Minimum thresholds for anti-gaming checks
    MIN_TRADES_PER_EPISODE = 0.5  # At least some trading activity
    MIN_MEANINGFUL_RISK_EXPOSURE = 0.001  # 0.1% of capital at risk
    MIN_TIME_IN_TRADE = 3  # Minimum bars held to count as real trade
    MAX_WINNER_CUT_RATIO = 0.3  # MFE vs actual profit ratio threshold
    
    def __init__(
        self,
        min_trade_activity: float = 0.5,
        min_risk_exposure: float = 0.001,
        min_bars_held: float = 3.0,
        max_early_exit_ratio: float = 0.3,
    ):
        self.min_trade_activity = min_trade_activity
        self.min_risk_exposure = min_risk_exposure
        self.min_bars_held = min_bars_held
        self.max_early_exit_ratio = max_early_exit_ratio
    
    def check_profit_factor_gaming(
        self,
        profit_factor: float,
        avg_trade_count: float,
        episode_length: float,
        opportunity_cost_estimate: float = 0.0,
    ) -> AntiGamingCheckResult:
        """
        Check if high profit factor is achieved through inactivity.
        
        Gaming: Trade very rarely to avoid losses.
        Counter: Require minimum trading activity + penalize opportunity cost.
        
        Note: min_trade_activity is defined as "minimum trades per 1000 bars".
        """
        # Compute actual trades per 1000 bars
        trades_per_1000_bars = avg_trade_count / max(episode_length / 1000.0, 0.001)
        
        # min_trade_activity is directly in "trades per 1000 bars" units
        min_threshold = self.min_trade_activity
        
        gaming_score = 0.0
        explanations = []
        
        # High PF with low activity is suspicious
        if profit_factor > 2.0 and trades_per_1000_bars < min_threshold:
            activity_ratio = trades_per_1000_bars / max(min_threshold, 0.01)
            gaming_score = max(gaming_score, (1.0 - activity_ratio) * 0.7)
            explanations.append(f"High PF({profit_factor:.1f}) with low activity({trades_per_1000_bars:.1f}/1k bars)")
        
        # Perfect PF (infinite) is always suspicious unless it's from very few trades
        if profit_factor > 5.0 and avg_trade_count < 3:
            gaming_score = max(gaming_score, 0.5)
            explanations.append(f"Very high PF({profit_factor:.1f}) from only {avg_trade_count:.1f} trades")
        
        return AntiGamingCheckResult(
            metric_name="profit_factor",
            gaming_detected=gaming_score > 0.5,
            gaming_score=gaming_score,
            explanation="; ".join(explanations) if explanations else "No gaming detected",
            counter_metrics={
                "trades_per_1000_bars": trades_per_1000_bars,
                "min_threshold": min_threshold,
                "opportunity_cost": opportunity_cost_estimate,
                "activity_ratio": trades_per_1000_bars / max(min_threshold, 0.01),
            },
            recommendation="Increase trading activity while maintaining quality" if gaming_score > 0.3 else "",
        )
    
    def check_drawdown_gaming(
        self,
        avg_drawdown: float,
        avg_position_size_pct: float,
        expected_return_per_risk: float,
        avg_pnl: float,
    ) -> AntiGamingCheckResult:
        """
        Check if low drawdown is achieved through tiny positions.
        
        Gaming: Use extremely small position sizes to minimize DD.
        Counter: Require meaningful risk exposure + return per unit risk.
        """
        gaming_score = 0.0
        explanations = []
        
        # Very low DD with very small positions is suspicious
        if avg_drawdown < 0.01 and avg_position_size_pct < self.min_risk_exposure:
            gaming_score = max(gaming_score, 0.6)
            explanations.append(f"Tiny positions({avg_position_size_pct*100:.3f}%) keeping DD artificially low")
        
        # Low DD but also low returns indicates overly conservative
        if avg_drawdown < 0.02 and avg_pnl < 0.001:
            gaming_score = max(gaming_score, 0.4)
            explanations.append("Low DD but negligible returns - may be avoiding risk entirely")
        
        # Good metric: return per unit risk taken
        if avg_drawdown > 0 and expected_return_per_risk < 0.1:
            gaming_score = max(gaming_score, 0.3)
            explanations.append(f"Poor return/risk ratio({expected_return_per_risk:.2f})")
        
        return AntiGamingCheckResult(
            metric_name="drawdown",
            gaming_detected=gaming_score > 0.5,
            gaming_score=gaming_score,
            explanation="; ".join(explanations) if explanations else "Legitimate low drawdown",
            counter_metrics={
                "position_size_pct": avg_position_size_pct,
                "return_per_risk": expected_return_per_risk,
                "is_meaningful_risk": avg_position_size_pct >= self.min_risk_exposure,
            },
            recommendation="Take more meaningful risk while maintaining discipline" if gaming_score > 0.3 else "",
        )
    
    def check_win_rate_gaming(
        self,
        win_rate: float,
        avg_r_multiple: float,
        avg_mfe: float,
        avg_profit_per_win: float,
        avg_bars_held_winners: float,
    ) -> AntiGamingCheckResult:
        """
        Check if high win rate is achieved by cutting winners early.
        
        Gaming: Close winning trades at tiny profits to boost win rate.
        Counter: Require adequate R-multiple + MFE capture ratio.
        """
        gaming_score = 0.0
        explanations = []
        
        # High win rate but poor R-multiple
        if win_rate > 0.6 and avg_r_multiple < 0.1:
            gaming_score = max(gaming_score, 0.5)
            explanations.append(f"High WR({win_rate:.0%}) but poor R({avg_r_multiple:.2f})")
        
        # High win rate with short hold times on winners
        if win_rate > 0.6 and avg_bars_held_winners < self.min_bars_held:
            gaming_score = max(gaming_score, 0.4)
            explanations.append(f"Winners closed too quickly ({avg_bars_held_winners:.1f} bars)")
        
        # MFE capture ratio: are we letting winners run?
        if avg_mfe > 0 and avg_profit_per_win > 0:
            mfe_capture = avg_profit_per_win / avg_mfe
            if mfe_capture < self.max_early_exit_ratio:
                gaming_score = max(gaming_score, 0.6)
                explanations.append(f"Only capturing {mfe_capture:.0%} of MFE")
        
        return AntiGamingCheckResult(
            metric_name="win_rate",
            gaming_detected=gaming_score > 0.5,
            gaming_score=gaming_score,
            explanation="; ".join(explanations) if explanations else "Legitimate win rate",
            counter_metrics={
                "r_multiple": avg_r_multiple,
                "mfe_capture_ratio": avg_profit_per_win / max(avg_mfe, 0.0001) if avg_mfe > 0 else 0.0,
                "avg_bars_held_winners": avg_bars_held_winners,
            },
            recommendation="Let winners run longer to capture more MFE" if gaming_score > 0.3 else "",
        )
    
    def check_r_multiple_gaming(
        self,
        avg_r_multiple: float,
        max_consecutive_losses: int,
        avg_mae: float,
        loss_recovery_rate: float,
    ) -> AntiGamingCheckResult:
        """
        Check if high R-multiple is achieved by letting losses run.
        
        Gaming: Hold losing positions hoping for recovery, creating artificially
                high R on winners while accumulating large losers.
        Counter: Monitor MAE, consecutive losses, and loss management.
        """
        gaming_score = 0.0
        explanations = []
        
        # High R but also high MAE suggests holding losers too long
        if avg_r_multiple > 0.5 and avg_mae > 0.02:
            gaming_score = max(gaming_score, 0.4)
            explanations.append(f"Good R({avg_r_multiple:.2f}) but high MAE({avg_mae:.1%}) - holding losers?")
        
        # High R with many consecutive losses
        if avg_r_multiple > 0.3 and max_consecutive_losses > 5:
            gaming_score = max(gaming_score, 0.5)
            explanations.append(f"High consecutive losses({max_consecutive_losses}) despite good R")
        
        # Low loss recovery suggests poor loss management
        if loss_recovery_rate < 0.3:
            gaming_score = max(gaming_score, 0.3)
            explanations.append(f"Poor loss recovery rate({loss_recovery_rate:.0%})")
        
        return AntiGamingCheckResult(
            metric_name="r_multiple",
            gaming_detected=gaming_score > 0.5,
            gaming_score=gaming_score,
            explanation="; ".join(explanations) if explanations else "Legitimate R-multiple",
            counter_metrics={
                "mae_ratio": avg_mae,
                "max_consecutive_losses": max_consecutive_losses,
                "loss_recovery_rate": loss_recovery_rate,
            },
            recommendation="Cut losses faster to reduce MAE" if gaming_score > 0.3 else "",
        )
    
    def check_consistency_gaming(
        self,
        win_rate_std: float,
        regime_coverage: Dict[str, float],
        performance_by_regime: Dict[str, float],
    ) -> AntiGamingCheckResult:
        """
        Check if high consistency is achieved by avoiding difficult conditions.
        
        Gaming: Only trade in easy/favorable conditions to appear consistent.
        Counter: Require trading across different regimes/conditions.
        """
        gaming_score = 0.0
        explanations = []
        
        # Check regime coverage
        if regime_coverage:
            total_coverage = sum(regime_coverage.values())
            num_regimes = len(regime_coverage)
            
            # If only trading in one or two regimes
            active_regimes = sum(1 for v in regime_coverage.values() if v > 0.1)
            if active_regimes < num_regimes * 0.5:
                gaming_score = max(gaming_score, 0.4)
                explanations.append(f"Only active in {active_regimes}/{num_regimes} regimes")
            
            # Check if performance varies wildly by regime
            if performance_by_regime:
                perfs = list(performance_by_regime.values())
                if len(perfs) > 1:
                    perf_std = float(np.std(perfs))
                    if perf_std > 0.3:
                        gaming_score = max(gaming_score, 0.3)
                        explanations.append(f"Performance varies by regime (std={perf_std:.2f})")
        
        return AntiGamingCheckResult(
            metric_name="consistency",
            gaming_detected=gaming_score > 0.5,
            gaming_score=gaming_score,
            explanation="; ".join(explanations) if explanations else "Legitimate consistency",
            counter_metrics={
                "regime_coverage": regime_coverage,
                "performance_by_regime": performance_by_regime,
            },
            recommendation="Trade across more market conditions" if gaming_score > 0.3 else "",
        )
    
    def run_all_checks(
        self,
        stats: Dict[str, Any],
        episode_metrics: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, AntiGamingCheckResult]:
        """
        Run all anti-gaming checks on provided statistics.
        
        Args:
            stats: Dictionary with keys like 'mean_profit_factor', 'mean_win_rate', etc.
            episode_metrics: Optional list of per-episode metrics for deeper analysis
            
        Returns:
            Dictionary mapping metric names to check results
        """
        results = {}
        
        # Extract values with safe defaults
        pf = _finite_or(stats.get("mean_profit_factor"), 1.0)
        wr = _finite_or(stats.get("mean_win_rate"), 0.5)
        dd = _finite_or(stats.get("mean_drawdown"), 0.05)
        r_mult = _finite_or(stats.get("mean_r_multiple"), 0.0)
        trade_count = _finite_or(stats.get("mean_trade_count"), 1.0)
        pnl = _finite_or(stats.get("mean_pnl"), 0.0)
        mae = _finite_or(stats.get("mean_mae", stats.get("avg_mae")), 0.01)
        mfe = _finite_or(stats.get("mean_mfe", stats.get("avg_mfe")), 0.01)
        bars_held = _finite_or(stats.get("avg_bars_held"), 10.0)
        wr_std = _finite_or(stats.get("std_win_rate"), 0.1)
        
        # Extract additional metrics that may be available
        episode_length = _finite_or(stats.get("mean_episode_length", stats.get("episode_length")), 2000.0)
        position_size_pct = _finite_or(stats.get("mean_position_size_pct", stats.get("avg_position_size_pct")), 0.01)
        loss_recovery = _finite_or(stats.get("loss_recovery_rate", stats.get("recovery_rate")), 0.5)
        regime_coverage = stats.get("regime_coverage", {})
        perf_by_regime = stats.get("performance_by_regime", {})
        
        # Track which fields are using fallback defaults for debugging
        missing_fields = []
        if "mean_episode_length" not in stats and "episode_length" not in stats:
            missing_fields.append("episode_length")
        if "mean_position_size_pct" not in stats and "avg_position_size_pct" not in stats:
            missing_fields.append("position_size_pct")
        if "loss_recovery_rate" not in stats and "recovery_rate" not in stats:
            missing_fields.append("loss_recovery_rate")
        if not regime_coverage:
            missing_fields.append("regime_coverage")
        if not perf_by_regime:
            missing_fields.append("performance_by_regime")
            
        if missing_fields:
            logger.debug(f"Anti-gaming checks using fallback defaults for: {missing_fields}")
        
        # Profit factor check
        results["profit_factor"] = self.check_profit_factor_gaming(
            profit_factor=pf,
            avg_trade_count=trade_count,
            episode_length=episode_length,
            opportunity_cost_estimate=0.0,
        )
        
        # Drawdown check
        results["drawdown"] = self.check_drawdown_gaming(
            avg_drawdown=dd,
            avg_position_size_pct=position_size_pct,
            expected_return_per_risk=pnl / max(dd, 0.001),
            avg_pnl=pnl,
        )
        
        # Win rate check
        results["win_rate"] = self.check_win_rate_gaming(
            win_rate=wr,
            avg_r_multiple=r_mult,
            avg_mfe=mfe,
            avg_profit_per_win=pnl * wr if pnl > 0 else 0.0,
            avg_bars_held_winners=bars_held,
        )
        
        # R-multiple check
        results["r_multiple"] = self.check_r_multiple_gaming(
            avg_r_multiple=r_mult,
            max_consecutive_losses=_si(stats.get("max_consecutive_losses"), 3),
            avg_mae=mae,
            loss_recovery_rate=loss_recovery,
        )
        
        # Consistency check - uses regime data if available
        results["consistency"] = self.check_consistency_gaming(
            win_rate_std=wr_std,
            regime_coverage=regime_coverage if isinstance(regime_coverage, dict) else {},
            performance_by_regime=perf_by_regime if isinstance(perf_by_regime, dict) else {},
        )
        
        return results
    
    def get_aggregate_gaming_score(
        self,
        check_results: Dict[str, AntiGamingCheckResult],
    ) -> Tuple[float, List[str]]:
        """
        Get aggregate gaming score and list of concerns.
        
        Returns:
            Tuple of (aggregate_score, list of concern strings)
        """
        scores = [r.gaming_score for r in check_results.values()]
        concerns = [
            f"{r.metric_name}: {r.explanation}"
            for r in check_results.values()
            if r.gaming_detected
        ]
        
        # Use max rather than mean - one gaming strategy is enough
        aggregate = max(scores) if scores else 0.0
        
        return aggregate, concerns


# =============================================================================
# Metric Bounds Checker
# =============================================================================

@dataclass
class MetricBounds:
    """Valid bounds for a metric."""
    name: str
    min_value: float
    max_value: float
    allow_nan: bool = False
    allow_inf: bool = False


# Helper functions for NaN/Inf that work with both Python float and numpy types
def _is_nan(x: Any) -> bool:
    """Check if value is NaN, handling both float and np.floating."""
    try:
        return bool(np.isnan(float(x)))
    except Exception:
        return False


def _is_inf(x: Any) -> bool:
    """Check if value is infinite, handling both float and np.floating."""
    try:
        return bool(np.isinf(float(x)))
    except Exception:
        return False


DEFAULT_METRIC_BOUNDS = [
    # Win rate variants
    MetricBounds("win_rate", 0.0, 1.0),
    MetricBounds("mean_win_rate", 0.0, 1.0),
    # Drawdown variants
    MetricBounds("max_drawdown", 0.0, 1.0),
    MetricBounds("mean_drawdown", 0.0, 1.0),
    MetricBounds("daily_drawdown", 0.0, 1.0),
    MetricBounds("dd_breach_rate", 0.0, 1.0),
    # Profit factor variants
    MetricBounds("profit_factor", 0.0, 100.0),  # Capped, not infinite
    MetricBounds("mean_profit_factor", 0.0, 100.0),
    # Entry quality variants
    MetricBounds("avg_entry_quality", 0.0, 1.0),
    MetricBounds("mean_entry_quality", 0.0, 1.0),
    # R-multiple variants
    MetricBounds("r_multiple", -10.0, 50.0),
    MetricBounds("mean_r_multiple", -10.0, 50.0),
    MetricBounds("avg_r_multiple", -10.0, 50.0),
    # Trade counts
    MetricBounds("trade_count", 0.0, 10000.0),
    MetricBounds("mean_trade_count", 0.0, 10000.0),
    MetricBounds("std_win_rate", 0.0, 1.0),
    MetricBounds("mean_pnl", -1e9, 1e9, allow_nan=False, allow_inf=False),
    MetricBounds("avg_mae", 0.0, 1.0),
    MetricBounds("avg_mfe", 0.0, 1.0),
    MetricBounds("episode_length", 0.0, 100000.0),
    # Other
    MetricBounds("policy_entropy", -1.0, 10.0),  # -1 = not available
    MetricBounds("sharpe_ratio", -10.0, 20.0),
    MetricBounds("mean_sharpe_ratio", -10.0, 20.0),
]


def check_metric_bounds(
    metrics: Dict[str, Any],
    bounds: Optional[List[MetricBounds]] = None,
    stage_name: str = "",
    stage_epoch: int = 0,
    episode_idx: int = 0,
) -> List[InvariantViolation]:
    """
    Check that all metrics are within valid bounds.
    
    Returns list of violations found.
    """
    bounds = bounds or DEFAULT_METRIC_BOUNDS
    violations = []
    
    for bound in bounds:
        if bound.name not in metrics:
            continue
        
        value = metrics[bound.name]
        
        # Check for NaN (handles both float and np.floating)
        if _is_nan(value):
            if not bound.allow_nan:
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.NUMERIC_SANITY,
                    severity=InvariantSeverity.ERROR,
                    message=f"NaN value for {bound.name}",
                    context={"metric": bound.name, "value": "NaN"},
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                    episode_idx=episode_idx,
                ))
            continue
        
        # Check for inf (handles both float and np.floating)
        if _is_inf(value):
            if not bound.allow_inf:
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.NUMERIC_SANITY,
                    severity=InvariantSeverity.ERROR,
                    message=f"Infinite value for {bound.name}",
                    context={"metric": bound.name, "value": str(value)},
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                    episode_idx=episode_idx,
                ))
            continue
        
        # Check bounds
        try:
            fval = float(value)
            if not _is_finite(fval):
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.NUMERIC_SANITY,
                    severity=InvariantSeverity.ERROR,
                    message=f"Non-finite value for {bound.name}",
                    context={"metric": bound.name, "value": str(value)},
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                    episode_idx=episode_idx,
                ))
                continue
            if fval < bound.min_value or fval > bound.max_value:
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.METRIC_BOUNDS,
                    severity=InvariantSeverity.WARNING,
                    message=f"{bound.name}={fval:.4f} outside [{bound.min_value}, {bound.max_value}]",
                    context={
                        "metric": bound.name,
                        "value": fval,
                        "min": bound.min_value,
                        "max": bound.max_value,
                    },
                    stage_name=stage_name,
                    stage_epoch=stage_epoch,
                    episode_idx=episode_idx,
                ))
        except (TypeError, ValueError) as e:
            logger.debug(f"Skipping non-numeric metric {bound.name}: {type(value).__name__}")
    
    return violations


# =============================================================================
# Exit Distribution Checker  
# =============================================================================

def check_exit_distribution(
    trailing_stop_exits: int,
    agent_close_exits: int,
    hard_stop_exits: int,
    risk_liquidation_exits: int,
    other_exits: int,
    trade_count: int,
    pooled_trades: Optional[int] = None,
    stage_name: str = "",
    stage_epoch: int = 0,
    episode_idx: int = 0,
) -> List[InvariantViolation]:
    """
    Check exit distribution for sanity.
    
    Returns list of violations found.
    """
    violations = []
    
    total_exits = trailing_stop_exits + agent_close_exits + hard_stop_exits + risk_liquidation_exits + other_exits
    trade_count = max(0, int(trade_count))
    denom = int(pooled_trades) if pooled_trades is not None and int(pooled_trades) > 0 else trade_count
    
    # Total exits should approximately match trade count
    if denom > 0:
        ratio = total_exits / denom
        if ratio > 1.5:  # More than 50% extra exits
            violations.append(InvariantViolation(
                invariant_type=InvariantType.EXIT_DISTRIBUTION,
                severity=InvariantSeverity.WARNING,
                message=f"Exit count ({total_exits}) >> trade count ({denom})",
                context={
                    "total_exits": total_exits,
                    "trade_count": denom,
                    "ratio": ratio,
                    "pooled_trades": pooled_trades,
                },
                stage_name=stage_name,
                stage_epoch=stage_epoch,
                episode_idx=episode_idx,
            ))
    
    # High risk liquidation rate is concerning
    if total_exits > 10 and risk_liquidation_exits > 0:
        liq_rate = risk_liquidation_exits / total_exits
        if liq_rate > 0.1:  # More than 10% liquidations
            violations.append(InvariantViolation(
                invariant_type=InvariantType.EXIT_DISTRIBUTION,
                severity=InvariantSeverity.WARNING,
                message=f"High risk liquidation rate: {liq_rate:.1%}",
                context={
                    "risk_liquidations": risk_liquidation_exits,
                    "total_exits": total_exits,
                    "rate": liq_rate,
                },
                stage_name=stage_name,
                stage_epoch=stage_epoch,
                episode_idx=episode_idx,
            ))
    
    return violations


# =============================================================================
# Comprehensive Invariant Checker
# =============================================================================

class CurriculumInvariantChecker:
    """
    Comprehensive invariant checker for curriculum manager.
    
    Runs at episode record time to validate:
    - Trade accounting
    - Metric bounds
    - Exit distribution
    - Stage progression rules
    - Counter integrity
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = RateLimitedInvariantLogger()
        self.anti_gaming = AntiGamingChecker()
        
        # Track for stage progression invariants
        self._last_stage: Optional[str] = None
        self._stage_history: List[Tuple[str, int]] = []  # (stage_name, epoch)
    
    def check_episode(
        self,
        metrics: Dict[str, Any],
        stage_name: str,
        stage_epoch: int,
        episode_idx: int,
    ) -> List[InvariantViolation]:
        """
        Run all invariant checks on an episode.
        
        Args:
            metrics: Episode metrics dict
            stage_name: Current stage name
            stage_epoch: Current stage epoch
            episode_idx: Global episode index
            
        Returns:
            List of violations found
        """
        all_violations = []
        
        # 1. Trade accounting
        trade_result = reconcile_trade_accounting(
            trade_count=_finite_int(metrics.get("trade_count"), 0),
            winning_trades=_finite_int(metrics.get("winning_trades"), 0),
            losing_trades=_finite_int(metrics.get("losing_trades"), 0),
            win_rate=_finite_or(metrics.get("win_rate"), 0.0),
            stage_name=stage_name,
            stage_epoch=stage_epoch,
            episode_idx=episode_idx,
        )
        if trade_result.violation:
            all_violations.append(trade_result.violation)
        
        # 2. Metric bounds
        all_violations.extend(check_metric_bounds(
            metrics=metrics,
            stage_name=stage_name,
            stage_epoch=stage_epoch,
            episode_idx=episode_idx,
        ))
        
        # 3. Exit distribution
        all_violations.extend(check_exit_distribution(
            trailing_stop_exits=_si(metrics.get("trailing_stop_exits"), 0),
            agent_close_exits=_si(metrics.get("agent_close_exits"), 0),
            hard_stop_exits=_si(metrics.get("hard_stop_exits"), 0),
            risk_liquidation_exits=_si(metrics.get("risk_liquidation_exits"), 0),
            other_exits=_si(metrics.get("other_exits"), 0),
            trade_count=trade_result.reconciled_trade_count,
            pooled_trades=trade_result.pooled_trades,
            stage_name=stage_name,
            stage_epoch=stage_epoch,
            episode_idx=episode_idx,
        ))
        
        # Log violations (rate-limited)
        for v in all_violations:
            self.logger.log(v)
        
        return all_violations
    
    def check_rolling_stats(
        self,
        stats: Dict[str, Any],
        stage_name: str,
        stage_epoch: int,
    ) -> Tuple[List[InvariantViolation], Dict[str, AntiGamingCheckResult]]:
        """
        Run checks on rolling statistics including anti-gaming.
        
        Returns:
            Tuple of (violations, anti_gaming_results)
        """
        violations = []
        
        # Metric bounds on stats
        violations.extend(check_metric_bounds(
            metrics=stats,
            stage_name=stage_name,
            stage_epoch=stage_epoch,
        ))
        
        # Anti-gaming checks
        gaming_results = self.anti_gaming.run_all_checks(stats)
        
        aggregate_score, concerns = self.anti_gaming.get_aggregate_gaming_score(gaming_results)
        
        if aggregate_score > 0.5:
            violations.append(InvariantViolation(
                invariant_type=InvariantType.ANTI_GAMING,
                severity=InvariantSeverity.WARNING,
                message=f"Potential gaming detected (score={aggregate_score:.2f})",
                context={
                    "gaming_score": aggregate_score,
                    "concerns": concerns,
                },
                stage_name=stage_name,
                stage_epoch=stage_epoch,
            ))
        
        for v in violations:
            self.logger.log(v)
        
        return violations, gaming_results
    
    def check_stage_transition(
        self,
        old_stage: str,
        new_stage: str,
        is_promotion: bool,
        is_demotion: bool,
        cooldown_remaining: int,
        old_stage_index: Optional[int] = None,
        new_stage_index: Optional[int] = None,
        min_stage_index: int = 0,
    ) -> List[InvariantViolation]:
        """
        Check stage transition invariants.
        
        Rules:
        - Cannot demote below foundation (stage index 0)
        - Transitions should not happen during cooldown
        - Promotion and demotion are mutually exclusive
        """
        violations = []
        
        if is_promotion and is_demotion:
            violations.append(InvariantViolation(
                invariant_type=InvariantType.STAGE_PROGRESSION,
                severity=InvariantSeverity.ERROR,
                message="Both promotion and demotion flagged",
                context={"old_stage": old_stage, "new_stage": new_stage},
            ))
        
        if cooldown_remaining > 0 and (is_promotion or is_demotion):
            violations.append(InvariantViolation(
                invariant_type=InvariantType.STAGE_PROGRESSION,
                severity=InvariantSeverity.WARNING,
                message=f"Transition during cooldown (remaining={cooldown_remaining})",
                context={
                    "old_stage": old_stage,
                    "new_stage": new_stage,
                    "cooldown_remaining": cooldown_remaining,
                },
            ))

        # Optional: cannot demote below foundation (if indices are provided by caller)
        if is_demotion and new_stage_index is not None:
            if int(new_stage_index) < int(min_stage_index):
                violations.append(InvariantViolation(
                    invariant_type=InvariantType.STAGE_PROGRESSION,
                    severity=InvariantSeverity.ERROR,
                    message="Demotion below minimum stage index",
                    context={
                        "old_stage": old_stage,
                        "new_stage": new_stage,
                        "old_stage_index": old_stage_index,
                        "new_stage_index": new_stage_index,
                        "min_stage_index": min_stage_index,
                    },
                ))
        
        for v in violations:
            self.logger.log(v)
        
        return violations
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all violations."""
        return {
            "violation_counts": self.logger.get_summary(),
            "total_violations": len(self.logger._violations),
        }
    
    def export_violations(self, clear: bool = True) -> List[Dict[str, Any]]:
        """Export all violations as JSON-serializable dicts."""
        return self.logger.get_violations(clear=clear)
