# envs/validation_gates.py
"""
Validation Gates for Curriculum Progression
============================================

Phase 3: Reliability under distribution shift.

Makes validation a FIRST-CLASS gate for promotion, not optional.
Maintains fixed validation scenarios that test generalization:
- Different volatility conditions
- Different spread/slippage levels
- Different start dates
- Stress conditions (adversarial)

Promotion requires:
1. Performance ratio threshold (train vs validation)
2. Safety threshold (DD breaches near zero)
3. No catastrophic behavior (overtrading, liquidation spikes)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from envs.core.shared_utils import (
    clamp as _clamp,
)
from envs.core.shared_utils import (
    get_envs_logger,
    iso_timestamp,
)
from envs.core.shared_utils import (
    safe_float as _sf,
)

logger = get_envs_logger("validation_gates")


def _is_finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _finite_or(x: Any, default: float) -> float:
    v = _sf(x, default)
    return v if _is_finite(v) else default


def _finite_mean(xs: List[float], default: float = 0.0) -> float:
    vals = [float(v) for v in xs if _is_finite(v)]
    return float(np.mean(vals)) if vals else float(default)


def _finite_max(xs: List[float], default: float = 0.0) -> float:
    vals = [float(v) for v in xs if _is_finite(v)]
    return float(max(vals)) if vals else float(default)


# =============================================================================
# Unified Performance Scoring
# =============================================================================

def compute_performance_score(
    mean_win_rate: float,
    mean_profit_factor: float,
    mean_r_multiple: float,
) -> float:
    """
    Compute a unified performance score.
    
    This SAME function must be used for both training baseline and validation
    scenarios to ensure performance_ratio is meaningful.
    
    Components:
    - Win rate normalized to ~1.0 at 50% (capped to [0, 1.5])
    - Profit factor capped and normalized (capped to [0, 1.5])
    - R-multiple shifted to positive range (capped to [0, 1.5])
    """
    mean_win_rate = _finite_or(mean_win_rate, 0.0)
    mean_profit_factor = _finite_or(mean_profit_factor, 0.0)
    mean_r_multiple = _finite_or(mean_r_multiple, 0.0)

    # Cap each component individually BEFORE averaging to prevent unbounded values
    wr_component = _clamp(mean_win_rate / 0.5, 0.0, 1.5)           # Normalize: 50% WR -> 1.0
    pf_component = _clamp(mean_profit_factor / 1.5, 0.0, 1.5)      # Cap at 1.5
    rm_component = _clamp((mean_r_multiple + 0.5) / 1.0, 0.0, 1.5) # Shift and normalize
    components = [wr_component, pf_component, rm_component]
    return _clamp(float(np.mean(components)), 0.0, 1.5)


def _get_episode_value(ep: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """
    Get a value from an episode dict, trying multiple possible key names.
    
    This handles metric naming drift across different parts of the codebase:
    - total_pnl vs pnl vs episode_pnl
    - avg_r_multiple vs mean_r_multiple vs r_multiple
    - profit_factor vs pf
    """
    for key in keys:
        if key in ep and ep[key] is not None:
            v = _sf(ep[key], default)
            if _is_finite(v):
                return v
    return default


def _get_training_value(stats: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """Training-stats accessor with naming drift handling and finite-only values."""
    for key in keys:
        if key in stats and stats[key] is not None:
            v = _sf(stats[key], default)
            if _is_finite(v):
                return v
    return default


def _estimate_training_confidence(stats: Dict[str, Any]) -> float:
    """
    Estimate confidence in training statistics.
    Preference order:
      1) explicit 'confidence' / 'training_confidence'
      2) episode count / window size heuristics
    """
    conf = _get_training_value(stats, "confidence", "training_confidence", default=-1.0)
    if conf >= 0.0:
        return float(_clamp(conf, 0.0, 1.0))

    n = _get_training_value(
        stats,
        "episodes_in_window",
        "n_episodes",
        "num_episodes",
        "window_size",
        default=0.0,
    )
    # Heuristic: ~100 episodes ~= high confidence
    return float(_clamp(n / 100.0, 0.0, 1.0))


# =============================================================================
# Validation Scenario Types
# =============================================================================

class ValidationScenarioType(Enum):
    """Types of validation scenarios."""
    STANDARD = "standard"           # Normal conditions, different dates
    HIGH_VOLATILITY = "high_vol"    # High ATR periods
    LOW_VOLATILITY = "low_vol"      # Low ATR periods
    WIDE_SPREAD = "wide_spread"     # Elevated transaction costs
    HIGH_SLIPPAGE = "high_slip"     # Increased execution uncertainty
    TREND_REGIME = "trend"          # Strong trending periods
    RANGE_REGIME = "range"          # Sideways/choppy periods
    NEWS_PERIODS = "news"           # Around major economic releases
    STRESS_TEST = "stress"          # Adversarial conditions


@dataclass
class ValidationScenario:
    """Definition of a validation scenario."""
    name: str
    scenario_type: ValidationScenarioType
    description: str
    
    # Data selection
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbol: str = "XAUUSD"
    
    # Environment modifications
    spread_multiplier: float = 1.0      # Multiply base spread
    slippage_multiplier: float = 1.0    # Multiply base slippage
    volatility_filter: Optional[str] = None  # "high", "low", or None
    trend_filter: Optional[str] = None       # "trending", "ranging", or None
    
    # Expected difficulty (for weighting)
    difficulty: float = 1.0  # 1.0 = normal, higher = harder
    
    # Minimum requirements
    min_episodes: int = 10
    min_trades: int = 50
    # Optional success criteria for behavioral/patience gating
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.scenario_type.value,
            "description": self.description,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "spread_multiplier": self.spread_multiplier,
            "slippage_multiplier": self.slippage_multiplier,
            "volatility_filter": self.volatility_filter,
            "trend_filter": self.trend_filter,
            "difficulty": self.difficulty,
            "min_episodes": self.min_episodes,
            "min_trades": self.min_trades,
            "success_criteria": self.success_criteria,
        }


# =============================================================================
# Predefined Validation Scenarios
# =============================================================================

def get_default_validation_scenarios() -> List[ValidationScenario]:
    """Get the default suite of validation scenarios."""
    return [
        # Standard validation (different date range)
        ValidationScenario(
            name="standard_validation",
            scenario_type=ValidationScenarioType.STANDARD,
            description="Standard conditions, held-out date range",
            difficulty=1.0,
        ),
        
        # High volatility stress
        ValidationScenario(
            name="high_volatility",
            scenario_type=ValidationScenarioType.HIGH_VOLATILITY,
            description="High ATR periods only",
            volatility_filter="high",
            difficulty=1.3,
        ),
        
        # Wide spread stress
        ValidationScenario(
            name="wide_spreads",
            scenario_type=ValidationScenarioType.WIDE_SPREAD,
            description="Spreads 2x normal",
            spread_multiplier=2.0,
            difficulty=1.4,
        ),
        
        # High slippage stress
        ValidationScenario(
            name="high_slippage",
            scenario_type=ValidationScenarioType.HIGH_SLIPPAGE,
            description="Slippage 2x normal",
            slippage_multiplier=2.0,
            difficulty=1.3,
        ),
        
        # Ranging market
        ValidationScenario(
            name="ranging_market",
            scenario_type=ValidationScenarioType.RANGE_REGIME,
            description="Low ADX, sideways periods",
            trend_filter="ranging",
            difficulty=1.2,
        ),
        
        # Combined stress test
        ValidationScenario(
            name="stress_test",
            scenario_type=ValidationScenarioType.STRESS_TEST,
            description="Wide spreads + high slippage + high vol",
            spread_multiplier=1.5,
            slippage_multiplier=1.5,
            volatility_filter="high",
            difficulty=2.0,
        ),
    ]


# =============================================================================
# Validation Results
# =============================================================================

@dataclass
class ScenarioResult:
    """Results from running one validation scenario."""
    scenario_name: str
    scenario_type: str
    
    # Episode stats
    episodes_run: int = 0
    total_trades: int = 0
    
    # Performance metrics
    mean_win_rate: float = 0.0
    mean_profit_factor: float = 0.0
    mean_pnl: float = 0.0
    mean_r_multiple: float = 0.0
    
    # Risk metrics
    max_drawdown_seen: float = 0.0
    dd_breach_count: int = 0
    dd_breach_rate: float = 0.0
    
    # Behavior metrics
    mean_trade_count: float = 0.0
    liquidation_count: int = 0
    liquidation_rate: float = 0.0
    
    # Computed scores
    performance_score: float = 0.0  # Composite performance
    safety_score: float = 0.0       # Safety/risk score
    behavior_score: float = 0.0     # No catastrophic behavior
    
    # Pass/fail
    passed: bool = False
    failure_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationGateResult:
    """Aggregate result from all validation scenarios."""
    timestamp: str = field(default_factory=iso_timestamp)
    stage_name: str = ""
    stage_epoch: int = 0
    
    # Scenario results
    scenario_results: Dict[str, ScenarioResult] = field(default_factory=dict)
    
    # Aggregate metrics
    scenarios_passed: int = 0
    scenarios_total: int = 0
    pass_rate: float = 0.0
    
    # Weighted scores
    weighted_performance: float = 0.0
    weighted_safety: float = 0.0
    weighted_behavior: float = 0.0
    
    # Performance ratio (validation vs training)
    performance_ratio: float = 0.0
    
    # Final verdict
    gate_passed: bool = False
    gate_confidence: float = 0.0
    blocking_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "stage_name": self.stage_name,
            "stage_epoch": self.stage_epoch,
            "scenarios_passed": self.scenarios_passed,
            "scenarios_total": self.scenarios_total,
            "pass_rate": self.pass_rate,
            "weighted_performance": self.weighted_performance,
            "weighted_safety": self.weighted_safety,
            "weighted_behavior": self.weighted_behavior,
            "performance_ratio": self.performance_ratio,
            "gate_passed": self.gate_passed,
            "gate_confidence": self.gate_confidence,
            "blocking_reasons": self.blocking_reasons,
            "scenario_results": {k: v.to_dict() for k, v in self.scenario_results.items()},
        }


# =============================================================================
# Validation Gate Checker
# =============================================================================

@dataclass
class ValidationGateConfig:
    """Configuration for validation gating."""
    # Enable/disable validation gating
    enabled: bool = True
    
    # Minimum requirements
    min_scenarios_passed: int = 4         # Out of 6 default scenarios
    min_pass_rate: float = 0.67           # At least 67% of scenarios
    
    # Performance ratio thresholds
    # validation_performance / training_performance must exceed this
    min_performance_ratio: float = 0.7    # Val can be 30% worse than train
    target_performance_ratio: float = 0.85  # Ideal: val within 15% of train
    
    # Safety thresholds
    max_dd_breach_rate: float = 0.05      # Max 5% episodes with DD breach
    max_liquidation_rate: float = 0.02    # Max 2% liquidations
    min_weighted_safety: float = 0.7      # Minimum weighted safety score
    
    # Behavior thresholds
    max_trade_count_ratio: float = 2.0    # Val trades / train trades
    min_trade_count_ratio: float = 0.5    # Don't under-trade in val
    
    # Training stats completeness requirements
    require_training_stats: bool = True   # Block if training stats missing
    min_training_confidence: float = 0.3  # Minimum confidence in training stats
    
    # Stage-specific adjustments
    early_stage_relaxation: float = 0.8   # Relax thresholds for early stages
    late_stage_strictness: float = 1.2    # Tighten for late stages


class ValidationGateChecker:
    """
    Checks validation performance as a promotion gate.
    
    For a promotion to proceed:
    1. Agent must pass minimum scenarios
    2. Performance ratio (val/train) must meet threshold
    3. Safety metrics (DD breaches, liquidations) must be acceptable
    4. No catastrophic behavior patterns
    """
    
    def __init__(
        self,
        config: Optional[ValidationGateConfig] = None,
        scenarios: Optional[List[ValidationScenario]] = None,
    ):
        self.config = config or ValidationGateConfig()
        self.scenarios = scenarios or get_default_validation_scenarios()
        
        # Cache for results
        self._last_result: Optional[ValidationGateResult] = None

    def add_patience_scenarios(self, stage_index: int) -> List[ValidationScenario]:
        """Add patience/discipline scenarios for later stages."""
        scenarios: List[ValidationScenario] = []
        if stage_index >= 4:  # TIMING_STUDENT+
            scenarios.append(
                ValidationScenario(
                    name="low_volatility_patience",
                    scenario_type=ValidationScenarioType.LOW_VOLATILITY,
                    description="Test patience in low volatility (choppy) market",
                    volatility_filter="low",
                    difficulty=1.2,
                    min_episodes=8,
                    min_trades=2,
                    success_criteria={
                        "max_trades_per_episode": 2,
                        "min_bars_between_trades": 8.0,
                        "min_win_rate": 0.60,
                    },
                )
            )
        if stage_index >= 6:  # RISK_MANAGER+
            scenarios.append(
                ValidationScenario(
                    name="fomo_resistance",
                    scenario_type=ValidationScenarioType.NEWS_PERIODS,
                    description="Test resistance to FOMO after missed opportunity",
                    volatility_filter="high",
                    difficulty=1.4,
                    min_episodes=6,
                    min_trades=1,
                    success_criteria={
                        "max_fomo_trades": 1,
                        "max_revenge_trades": 0,
                    },
                )
            )
        return scenarios

    def get_scenarios_for_stage(self, stage_index: int) -> List[ValidationScenario]:
        """Return base scenarios plus any stage-specific additions."""
        scenarios = list(self.scenarios)
        scenarios.extend(self.add_patience_scenarios(stage_index))
        return scenarios
    
    def evaluate_scenario(
        self,
        scenario: ValidationScenario,
        episodes: List[Dict[str, Any]],
        training_stats: Dict[str, Any],
    ) -> ScenarioResult:
        """
        Evaluate a single validation scenario.
        
        Args:
            scenario: The scenario definition
            episodes: List of episode results from validation run
            training_stats: Reference training statistics for comparison
            
        Returns:
            ScenarioResult with pass/fail determination
        """
        result = ScenarioResult(
            scenario_name=scenario.name,
            scenario_type=scenario.scenario_type.value,
        )
        
        if not episodes:
            result.failure_reasons.append("No episodes")
            return result
        
        result.episodes_run = len(episodes)
        
        # Aggregate episode metrics
        win_rates = []
        profit_factors = []
        pnls = []
        r_multiples = []
        trade_counts = []
        drawdowns = []
        bars_between = []
        setup_qualities = []
        entry_certainties = []
        fomo_counts = []
        revenge_counts = []
        dd_breaches = 0
        liquidations = 0
        
        for ep in episodes:
            # Use fallback keys to handle metric naming variations
            win_rates.append(_get_episode_value(ep, "win_rate", "wr", default=0.0))
            profit_factors.append(_get_episode_value(
                ep, "profit_factor", "pf", "mean_profit_factor", default=0.0
            ))
            pnls.append(_get_episode_value(
                ep, "total_pnl", "pnl", "episode_pnl", "net_pnl", default=0.0
            ))
            r_multiples.append(_get_episode_value(
                ep, "avg_r_multiple", "mean_r_multiple", "r_multiple", default=0.0
            ))
            trade_counts.append(_get_episode_value(
                ep, "trade_count", "trades", "num_trades", default=0.0
            ))
            drawdowns.append(_get_episode_value(
                ep, "max_drawdown", "drawdown", "dd", default=0.0
            ))
            bars_between.append(_get_episode_value(
                ep, "avg_bars_between_trades", "mean_bars_between_trades", default=float("nan")
            ))
            setup_qualities.append(_get_episode_value(
                ep, "avg_setup_quality", "setup_quality", default=float("nan")
            ))
            entry_certainties.append(_get_episode_value(
                ep, "avg_entry_certainty", "entry_certainty", default=float("nan")
            ))
            fomo_counts.append(_get_episode_value(
                ep, "fomo_trade_count", "fomo_trades", default=float("nan")
            ))
            revenge_counts.append(_get_episode_value(
                ep, "revenge_trade_count", "revenge_trades", default=float("nan")
            ))

            # DD breach: accept boolean or numeric episode-level indicators
            if bool(ep.get("dd_breach", False)):
                dd_breaches += 1
            else:
                dd_breaches += int(_finite_or(ep.get("dd_breach_count", 0), 0.0))

            # Liquidations: treat as episode-level "had any liquidation"
            liq = 0.0
            liq = max(liq, _finite_or(ep.get("risk_liquidation_exits", 0), 0.0))
            liq = max(liq, _finite_or(ep.get("liquidation_exits", 0), 0.0))
            liq = max(liq, _finite_or(ep.get("liquidations", 0), 0.0))
            liq = max(liq, _finite_or(ep.get("liquidation_count", 0), 0.0))
            if liq > 0:
                liquidations += 1

        # Finite-only aggregation (prevents NaN poisoning)
        safe_trade_counts = [float(v) for v in trade_counts if _is_finite(v)]
        result.total_trades = int(sum(safe_trade_counts)) if safe_trade_counts else 0
        result.mean_win_rate = _finite_mean(win_rates, 0.0)
        result.mean_profit_factor = _finite_mean(profit_factors, 0.0)
        result.mean_pnl = _finite_mean(pnls, 0.0)
        result.mean_r_multiple = _finite_mean(r_multiples, 0.0)
        result.mean_trade_count = _finite_mean(safe_trade_counts, 0.0)
        result.max_drawdown_seen = _finite_max(drawdowns, 0.0)
        result.dd_breach_count = dd_breaches
        result.dd_breach_rate = dd_breaches / max(len(episodes), 1)
        result.liquidation_count = liquidations
        result.liquidation_rate = liquidations / max(len(episodes), 1)
        
        # Check minimum requirements
        if result.episodes_run < scenario.min_episodes:
            result.failure_reasons.append(
                f"Insufficient episodes ({result.episodes_run} < {scenario.min_episodes})"
            )
        
        if result.total_trades < scenario.min_trades:
            result.failure_reasons.append(
                f"Insufficient trades ({result.total_trades} < {scenario.min_trades})"
            )
        
        # Compute scores
        
        # Performance score: use unified scoring function
        result.performance_score = compute_performance_score(
            result.mean_win_rate,
            result.mean_profit_factor,
            result.mean_r_multiple,
        )
        
        # Safety score: based on DD and liquidations
        dd_penalty = result.dd_breach_rate * 2  # Each 1% breach costs 2%
        liq_penalty = result.liquidation_rate * 5  # Liquidations are severe
        result.safety_score = _clamp(1.0 - dd_penalty - liq_penalty, 0.0, 1.0)
        
        # Behavior score: check for catastrophic patterns
        train_trade_count = _get_training_value(
            training_stats,
            "mean_trade_count",
            "mean_trade_count_avg",
            "avg_trade_count",
            default=float("nan"),
        )

        behavior_ok = True
        # Only check trade ratio if training trade count is available and meaningful
        if _is_finite(train_trade_count) and float(train_trade_count) > 1.0:
            trade_ratio = result.mean_trade_count / max(float(train_trade_count), 1e-6)

            if trade_ratio > self.config.max_trade_count_ratio:
                result.failure_reasons.append(
                    f"Overtrading in validation ({trade_ratio:.1f}x training)"
                )
                behavior_ok = False
            if trade_ratio < self.config.min_trade_count_ratio:
                result.failure_reasons.append(
                    f"Under-trading in validation ({trade_ratio:.1f}x training)"
                )
                behavior_ok = False
        else:
            # Training trade count unavailable - log warning but don't fail
            logger.debug("Skipping trade ratio check: training trade count unavailable")
        
        result.behavior_score = 1.0 if behavior_ok else 0.5
        
        # Check safety thresholds
        if result.dd_breach_rate > self.config.max_dd_breach_rate:
            result.failure_reasons.append(
                f"DD breach rate too high ({result.dd_breach_rate:.1%} > {self.config.max_dd_breach_rate:.1%})"
            )
        
        if result.liquidation_rate > self.config.max_liquidation_rate:
            result.failure_reasons.append(
                f"Liquidation rate too high ({result.liquidation_rate:.1%} > {self.config.max_liquidation_rate:.1%})"
            )

        # Scenario-specific success criteria (patience/discipline checks)
        criteria = getattr(scenario, "success_criteria", {}) or {}
        if criteria:
            max_trade_count = _finite_max(trade_counts, float("nan"))
            mean_bars_between = _finite_mean(bars_between, float("nan"))
            mean_setup_quality = _finite_mean(setup_qualities, float("nan"))
            mean_entry_certainty = _finite_mean(entry_certainties, float("nan"))
            max_fomo = _finite_max(fomo_counts, float("nan"))
            max_revenge = _finite_max(revenge_counts, float("nan"))

            def _check(name: str, required: float, actual: float) -> None:
                if not _is_finite(actual):
                    result.failure_reasons.append(f"Missing metric for criteria '{name}'")
                    return
                if name.startswith("min_"):
                    if actual < required:
                        result.failure_reasons.append(
                            f"{name} failed ({actual:.3f} < {required:.3f})"
                        )
                elif name.startswith("max_"):
                    if actual > required:
                        result.failure_reasons.append(
                            f"{name} failed ({actual:.3f} > {required:.3f})"
                        )

            for key, req in criteria.items():
                try:
                    req_val = float(req)
                except Exception:
                    continue
                if key == "max_trades_per_episode":
                    _check("max_trades_per_episode", req_val, max_trade_count)
                elif key == "min_bars_between_trades":
                    _check("min_bars_between_trades", req_val, mean_bars_between)
                elif key == "min_setup_quality":
                    _check("min_setup_quality", req_val, mean_setup_quality)
                elif key == "min_entry_certainty":
                    _check("min_entry_certainty", req_val, mean_entry_certainty)
                elif key == "min_win_rate":
                    _check("min_win_rate", req_val, result.mean_win_rate)
                elif key == "max_fomo_trades":
                    _check("max_fomo_trades", req_val, max_fomo)
                elif key == "max_revenge_trades":
                    _check("max_revenge_trades", req_val, max_revenge)

        # Final pass determination
        result.passed = len(result.failure_reasons) == 0
        
        return result
    
    def evaluate_all(
        self,
        validation_results: Dict[str, List[Dict[str, Any]]],  # scenario_name -> episodes
        training_stats: Dict[str, Any],
        stage_name: str = "",
        stage_epoch: int = 0,
        stage_index: int = 0,
        scenarios: Optional[List[ValidationScenario]] = None,
    ) -> ValidationGateResult:
        """
        Evaluate all validation scenarios and compute gate result.
        
        Args:
            validation_results: Dict mapping scenario name to list of episode results
            training_stats: Training statistics for comparison
            stage_name: Current curriculum stage
            stage_epoch: Current stage epoch
            stage_index: Stage index for threshold adjustment
            
        Returns:
            ValidationGateResult with aggregate determination
        """
        result = ValidationGateResult(
            stage_name=stage_name,
            stage_epoch=stage_epoch,
        )
        
        if not self.config.enabled:
            result.gate_passed = True
            result.gate_confidence = 0.5
            result.blocking_reasons.append("Validation gating disabled")
            self._last_result = result
            return result
        
        # Apply stage-based threshold adjustment
        if stage_index <= 2:  # Early stages (EXPLORER, EXPERIMENTER, TREND_STUDENT)
            threshold_mult = self.config.early_stage_relaxation
        elif stage_index >= 8:  # Late stages (PROFESSIONAL, LIVE_READY)
            threshold_mult = self.config.late_stage_strictness
        else:
            threshold_mult = 1.0

        # Training stats completeness / confidence checks (previously unused config)
        train_conf = _estimate_training_confidence(training_stats or {})
        essential_train_keys = ["mean_win_rate", "mean_profit_factor", "mean_r_multiple"]
        missing_train = [k for k in essential_train_keys if training_stats.get(k, None) is None]
        if self.config.require_training_stats:
            if missing_train:
                result.blocking_reasons.append(
                    f"Training stats missing required keys: {', '.join(missing_train)}"
                )
            if train_conf < self.config.min_training_confidence:
                result.blocking_reasons.append(
                    f"Training stats confidence too low ({train_conf:.2f} < {self.config.min_training_confidence:.2f})"
                )

        # Evaluate each scenario
        scenario_weights = []
        scenario_list = scenarios or self.get_scenarios_for_stage(stage_index)
        for scenario in scenario_list:
            scenario_name = scenario.name
            episodes = validation_results.get(scenario_name, [])
            
            scenario_result = self.evaluate_scenario(scenario, episodes, training_stats)
            result.scenario_results[scenario_name] = scenario_result
            
            if scenario_result.passed:
                result.scenarios_passed += 1
            
            # Weight by difficulty for aggregation
            scenario_weights.append((scenario_result, scenario.difficulty))
        
        result.scenarios_total = len(scenario_list)
        result.pass_rate = result.scenarios_passed / max(result.scenarios_total, 1)
        
        # Compute weighted aggregates
        total_weight = sum(w for _, w in scenario_weights)
        if total_weight > 0:
            result.weighted_performance = sum(
                r.performance_score * w for r, w in scenario_weights
            ) / total_weight
            result.weighted_safety = sum(
                r.safety_score * w for r, w in scenario_weights
            ) / total_weight
            result.weighted_behavior = sum(
                r.behavior_score * w for r, w in scenario_weights
            ) / total_weight
        
        # Compute performance ratio (validation vs training)
        # CRITICAL: Use the SAME scoring function for train and validation
        train_perf = compute_performance_score(
            _finite_or(training_stats.get("mean_win_rate"), 0.5),
            _finite_or(training_stats.get("mean_profit_factor"), 1.0),
            _finite_or(training_stats.get("mean_r_multiple"), 0.0),
        )

        if train_perf > 1e-8:
            result.performance_ratio = result.weighted_performance / train_perf
        else:
            result.performance_ratio = 1.0

        # Gate determination
        adjusted_min_pass_rate = self.config.min_pass_rate * threshold_mult
        adjusted_min_perf_ratio = self.config.min_performance_ratio * threshold_mult
        # Relax/tighten scenario-count requirements with stage, too
        adjusted_min_scenarios_passed = int(np.ceil(self.config.min_scenarios_passed * threshold_mult))
        adjusted_min_scenarios_passed = int(_clamp(adjusted_min_scenarios_passed, 1, len(scenario_list)))

        blocking = []

        if result.scenarios_passed < adjusted_min_scenarios_passed:
            blocking.append(
                f"Too few scenarios passed ({result.scenarios_passed}/{adjusted_min_scenarios_passed})"
            )
        
        if result.pass_rate < adjusted_min_pass_rate:
            blocking.append(
                f"Pass rate too low ({result.pass_rate:.0%} < {adjusted_min_pass_rate:.0%})"
            )
        
        # Check for empty scenarios - this is critical in prop mode
        no_data_scenarios = [
            name for name, sr in result.scenario_results.items()
            if sr.episodes_run == 0
        ]
        if no_data_scenarios and stage_index >= 6:  # Prop stages require data
            blocking.append(
                f"Missing validation data for scenarios: {', '.join(no_data_scenarios)}"
            )
        
        if result.performance_ratio < adjusted_min_perf_ratio:
            blocking.append(
                f"Performance degradation too large (ratio={result.performance_ratio:.2f} < {adjusted_min_perf_ratio:.2f})"
            )
        
        # Apply threshold_mult to safety check for consistency
        adjusted_min_safety = self.config.min_weighted_safety * threshold_mult
        if result.weighted_safety < adjusted_min_safety:
            blocking.append(
                f"Safety score too low ({result.weighted_safety:.2f} < {adjusted_min_safety:.2f})"
            )

        # No-catastrophe rule for later stages:
        # Even if pass-rate is sufficient, catastrophic safety/behavior failures should block promotion.
        if stage_index >= 6:
            catastrophic = []
            for name, sr in result.scenario_results.items():
                # Safety catastrophes
                if sr.dd_breach_rate > self.config.max_dd_breach_rate:
                    catastrophic.append(f"{name}: DD breaches")
                if sr.liquidation_rate > self.config.max_liquidation_rate:
                    catastrophic.append(f"{name}: liquidations")
                # Behavior catastrophes
                if any("Overtrading" in r or "Under-trading" in r for r in sr.failure_reasons):
                    catastrophic.append(f"{name}: trade-pattern drift")
            if catastrophic:
                blocking.append("Catastrophic behavior/safety failures in validation: " + ", ".join(sorted(set(catastrophic))))

        # Merge any earlier training-stat blocks
        result.blocking_reasons = list(dict.fromkeys(result.blocking_reasons + blocking))
        result.gate_passed = len(blocking) == 0
        
        # Confidence based on sample size and consistency
        total_episodes = sum(sr.episodes_run for sr in result.scenario_results.values())
        sample_confidence = min(1.0, total_episodes / 100)
        # Clamp consistency_confidence to [0, 1] - std can exceed 1.0 in rare cases
        consistency_confidence = _clamp(
            1.0 - float(np.std([
                sr.performance_score for sr in result.scenario_results.values()
            ])) if result.scenario_results else 0.5,
            0.0,
            1.0,
        )
        
        result.gate_confidence = (sample_confidence + consistency_confidence) / 2
        
        self._last_result = result
        return result
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations based on last validation result."""
        if not self._last_result:
            return ["Run validation first"]
        
        recs = []
        result = self._last_result
        
        # Analyze failure patterns
        perf_failures = []
        safety_failures = []
        behavior_failures = []
        
        for name, sr in result.scenario_results.items():
            if not sr.passed:
                for reason in sr.failure_reasons:
                    if "DD" in reason or "liquidation" in reason.lower():
                        safety_failures.append((name, reason))
                    elif "trading" in reason.lower():
                        behavior_failures.append((name, reason))
                    else:
                        perf_failures.append((name, reason))
        
        if safety_failures:
            recs.append(
                f"Improve risk management: {len(safety_failures)} scenarios had safety issues"
            )
        
        if behavior_failures:
            recs.append(
                f"Address behavioral drift: {len(behavior_failures)} scenarios had trading pattern issues"
            )
        
        if result.performance_ratio < 0.8:
            recs.append(
                f"Reduce overfitting: validation performance is {result.performance_ratio:.0%} of training"
            )
        
        # Scenario-specific recommendations
        stress = result.scenario_results.get("stress_test")
        if stress and not stress.passed:
            recs.append("Focus on robustness: failing stress test scenario")
        
        wide_spread = result.scenario_results.get("wide_spreads")
        if wide_spread and not wide_spread.passed:
            recs.append("Improve cost efficiency: underperforming with wider spreads")
        
        return recs


# =============================================================================
# Stress Test Framework
# =============================================================================

@dataclass
class StressTestConfig:
    """Configuration for adversarial stress testing."""
    enabled: bool = True
    
    # Spread stress
    spread_multipliers: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0])
    
    # Slippage stress
    slippage_multipliers: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0])
    
    # Latency simulation (bars of delay)
    latency_delays: List[int] = field(default_factory=lambda: [0, 1, 2])
    
    # Gap injection (skip random bars)
    gap_probabilities: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.05])
    
    # Session shift (hours)
    session_shifts: List[int] = field(default_factory=lambda: [0, -2, 2])
    
    # Minimum episodes per stress level
    min_episodes_per_level: int = 5
    
    # Pass criteria
    max_performance_degradation: float = 0.5  # Can lose up to 50% perf
    must_remain_profitable: bool = False      # Optional: require positive PnL

    # Optional patience/discipline stress scenarios
    include_patience_scenarios: bool = False


@dataclass
class StressTestResult:
    """Result from stress testing."""
    stress_type: str
    stress_level: float
    
    episodes_run: int = 0
    mean_win_rate: float = 0.0
    mean_pnl: float = 0.0
    performance_vs_baseline: float = 1.0
    
    passed: bool = True
    degradation_graceful: bool = True
    failure_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StressTestRunner:
    """
    Run adversarial stress tests on the policy.
    
    Tests:
    - Widen spreads progressively
    - Increase slippage
    - Add execution latency
    - Inject missing ticks/gaps
    - Shift session boundaries
    
    Tracks how performance degrades - should be GRACEFUL, not catastrophic.
    """
    
    def __init__(self, config: Optional[StressTestConfig] = None):
        self.config = config or StressTestConfig()
        self._results: List[StressTestResult] = []
    
    def get_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Generate all stress test scenarios."""
        scenarios = []
        
        # Spread stress
        for mult in self.config.spread_multipliers:
            scenarios.append({
                "name": f"spread_{mult}x",
                "type": "spread",
                "level": mult,
                "spread_multiplier": mult,
            })
        
        # Slippage stress
        for mult in self.config.slippage_multipliers:
            scenarios.append({
                "name": f"slippage_{mult}x",
                "type": "slippage",
                "level": mult,
                "slippage_multiplier": mult,
            })
        
        # Latency stress
        for delay in self.config.latency_delays:
            if delay > 0:
                scenarios.append({
                    "name": f"latency_{delay}bars",
                    "type": "latency",
                    "level": delay,
                    "latency_bars": delay,
                })
        
        # Gap injection
        for prob in self.config.gap_probabilities:
            if prob > 0:
                scenarios.append({
                    "name": f"gaps_{prob*100:.0f}pct",
                    "type": "gaps",
                    "level": prob,
                    "gap_probability": prob,
                })

        # Optional patience/discipline scenarios
        if bool(getattr(self.config, "include_patience_scenarios", False)):
            scenarios.extend(self.get_patience_stress_scenarios())
        
        return scenarios

    def get_patience_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Return scenarios that stress patience and discipline."""
        return [
            {
                "name": "news_volatility_patience",
                "type": "patience",
                "level": 1.0,
                "simulate_news_event": True,
                "volatility_spike": 3.0,
                "duration_bars": 30,
                "evaluation": {
                    "max_trades_per_episode": 1,
                    "min_bars_between_trades": 5.0,
                    "min_win_rate": 0.67,
                },
            },
            {
                "name": "choppy_ranging_patience",
                "type": "patience",
                "level": 1.0,
                "volatility_percentile": (0.7, 0.9),
                "trend_clarity": 0.0,
                "fake_breakouts": True,
                "evaluation": {
                    "max_trades_per_episode": 2,
                    "min_setup_quality": 0.8,
                },
            },
        ]
    
    def evaluate_stress_result(
        self,
        scenario: Dict[str, Any],
        episodes: List[Dict[str, Any]],
        baseline_stats: Dict[str, Any],
    ) -> StressTestResult:
        """
        Evaluate results from a stress scenario.
        
        Key metric: Is degradation GRACEFUL or CATASTROPHIC?
        """
        result = StressTestResult(
            stress_type=scenario["type"],
            stress_level=scenario["level"],
            episodes_run=len(episodes),
        )
        
        if not episodes:
            result.passed = False
            result.failure_reason = "No episodes"
            return result

        # Compute metrics (finite-only)
        win_rates = [_get_episode_value(ep, "win_rate", "wr", default=0.0) for ep in episodes]
        profit_factors = [_get_episode_value(ep, "profit_factor", "pf", default=0.0) for ep in episodes]
        r_multiples = [_get_episode_value(ep, "avg_r_multiple", "mean_r_multiple", "r_multiple", default=0.0) for ep in episodes]
        pnls = [_get_episode_value(ep, "total_pnl", "pnl", "episode_pnl", "net_pnl", default=0.0) for ep in episodes]
        trade_counts = [_get_episode_value(ep, "trade_count", "trades", "num_trades", default=float("nan")) for ep in episodes]
        bars_between = [_get_episode_value(ep, "avg_bars_between_trades", "mean_bars_between_trades", default=float("nan")) for ep in episodes]
        setup_qualities = [_get_episode_value(ep, "avg_setup_quality", "setup_quality", default=float("nan")) for ep in episodes]
        entry_certainties = [_get_episode_value(ep, "avg_entry_certainty", "entry_certainty", default=float("nan")) for ep in episodes]
        fomo_counts = [_get_episode_value(ep, "fomo_trade_count", "fomo_trades", default=float("nan")) for ep in episodes]
        revenge_counts = [_get_episode_value(ep, "revenge_trade_count", "revenge_trades", default=float("nan")) for ep in episodes]

        result.mean_win_rate = _finite_mean(win_rates, 0.0)
        result.mean_pnl = _finite_mean(pnls, 0.0)

        # Compare to baseline using the SAME unified score as validation
        stress_perf = compute_performance_score(
            result.mean_win_rate,
            _finite_mean(profit_factors, 0.0),
            _finite_mean(r_multiples, 0.0),
        )
        baseline_perf = compute_performance_score(
            _finite_or(baseline_stats.get("mean_win_rate"), 0.5),
            _finite_or(baseline_stats.get("mean_profit_factor"), 1.0),
            _finite_or(baseline_stats.get("mean_r_multiple"), 0.0),
        )
        result.performance_vs_baseline = (stress_perf / baseline_perf) if baseline_perf > 1e-8 else 1.0

        # Check degradation
        max_allowed_degradation = self.config.max_performance_degradation
        
        # Scale allowed degradation by stress level
        # Higher stress = more degradation allowed
        stress_level = scenario["level"]
        if scenario["type"] in ["spread", "slippage"]:
            # Multiplicative stress: allow (level-1)*20% extra degradation
            adjusted_max = max_allowed_degradation * (1 + (stress_level - 1) * 0.2)
        else:
            adjusted_max = max_allowed_degradation
        
        if result.performance_vs_baseline < (1 - adjusted_max):
            result.degradation_graceful = False
            result.failure_reason = (
                f"Catastrophic degradation: {result.performance_vs_baseline:.0%} of baseline "
                f"(min allowed: {(1-adjusted_max):.0%})"
            )
            result.passed = False
        
        # Optional: must remain profitable
        if self.config.must_remain_profitable and result.mean_pnl < 0:
            result.passed = False
            result.failure_reason = f"Unprofitable under stress: PnL={result.mean_pnl:.4f}"

        # Optional: evaluate scenario-specific criteria (patience stress)
        eval_criteria = scenario.get("evaluation", {}) or {}
        if eval_criteria:
            max_trade_count = _finite_max(trade_counts, float("nan"))
            mean_bars_between = _finite_mean(bars_between, float("nan"))
            mean_setup_quality = _finite_mean(setup_qualities, float("nan"))
            mean_entry_certainty = _finite_mean(entry_certainties, float("nan"))
            max_fomo = _finite_max(fomo_counts, float("nan"))
            max_revenge = _finite_max(revenge_counts, float("nan"))

            failures = []

            def _check(name: str, required: float, actual: float) -> None:
                if not _is_finite(actual):
                    failures.append(f"Missing metric for criteria '{name}'")
                    return
                if name.startswith("min_"):
                    if actual < required:
                        failures.append(f"{name} failed ({actual:.3f} < {required:.3f})")
                elif name.startswith("max_"):
                    if actual > required:
                        failures.append(f"{name} failed ({actual:.3f} > {required:.3f})")

            for key, req in eval_criteria.items():
                try:
                    req_val = float(req)
                except Exception:
                    continue
                if key == "max_trades_per_episode":
                    _check("max_trades_per_episode", req_val, max_trade_count)
                elif key == "min_bars_between_trades":
                    _check("min_bars_between_trades", req_val, mean_bars_between)
                elif key == "min_setup_quality":
                    _check("min_setup_quality", req_val, mean_setup_quality)
                elif key == "min_entry_certainty":
                    _check("min_entry_certainty", req_val, mean_entry_certainty)
                elif key == "min_win_rate":
                    _check("min_win_rate", req_val, result.mean_win_rate)
                elif key == "max_fomo_trades":
                    _check("max_fomo_trades", req_val, max_fomo)
                elif key == "max_revenge_trades":
                    _check("max_revenge_trades", req_val, max_revenge)

            if failures:
                result.passed = False
                msg = "; ".join(failures)
                if result.failure_reason:
                    result.failure_reason = f"{result.failure_reason}; {msg}"
                else:
                    result.failure_reason = msg
        
        return result
    
    def get_robustness_score(self, results: List[StressTestResult]) -> float:
        """
        Compute overall robustness score from stress test results.
        
        Score is based on:
        - How many stress levels passed
        - How gracefully performance degraded
        """
        if not results:
            return 0.5
        
        passed_count = sum(1 for r in results if r.passed)
        pass_rate = passed_count / len(results)
        
        # Average performance retention
        perf_retention = float(np.mean([r.performance_vs_baseline for r in results]))
        
        # Graceful degradation bonus
        graceful_count = sum(1 for r in results if r.degradation_graceful)
        graceful_rate = graceful_count / len(results)
        
        # Weighted score
        score = pass_rate * 0.4 + perf_retention * 0.4 + graceful_rate * 0.2
        
        return _clamp(score, 0.0, 1.0)
    
    def get_summary(self, results: List[StressTestResult]) -> Dict[str, Any]:
        """Get summary of stress test results."""
        return {
            "total_scenarios": len(results),
            "passed": sum(1 for r in results if r.passed),
            "graceful_degradation": sum(1 for r in results if r.degradation_graceful),
            "robustness_score": self.get_robustness_score(results),
            "by_type": {
                t: {
                    "passed": sum(1 for r in results if r.stress_type == t and r.passed),
                    "total": sum(1 for r in results if r.stress_type == t),
                    "avg_retention": float(np.mean([
                        r.performance_vs_baseline 
                        for r in results if r.stress_type == t
                    ])) if any(r.stress_type == t for r in results) else 0.0,
                }
                for t in set(r.stress_type for r in results)
            },
            "results": [r.to_dict() for r in results],
        }
