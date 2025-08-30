# ─────────────────────────────────────────────────────────────
# File: modules/reward/risk_adjusted_reward.py
# [ROCKET] PRODUCTION-READY Enhanced Risk-Adjusted Reward System
# Fully bus-driven balance/equity resolution (no hardcoded 10k)
# Consistent BaseModule init order; robust fallbacks; contract-safe outputs
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
import threading
from modules.contracts import module_args
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum

from modules.core.module_base import BaseModule, module
from modules.core.mixins import (
    SmartInfoBusTradingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
)
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import AuditConfiguration, RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


def utcnow() -> str:
    return datetime.datetime.utcnow().isoformat()


class RewardMode(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    LIVE_TRADING = "live_trading"
    EMERGENCY = "emergency"
    OPTIMIZATION = "optimization"


@dataclass
class RewardConfig:
    """Configuration for Risk-Adjusted Reward System (bus-first)."""

    # IMPORTANT: No hardcoded initial balance; we prefer the bus/env.
    # If you *must* set one via config, do it at runtime (e.g. via genome/env or environment_config).
    initial_balance: Optional[float] = None

    history_size: int = 50
    min_trade_bonus: float = 0.5

    # Regime weights
    regime_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])

    # Penalty weights
    dd_pen_weight: float = 2.0
    risk_pen_weight: float = 0.1
    tail_pen_weight: float = 0.5
    mistake_pen_weight: float = 0.3
    no_trade_penalty_weight: float = 0.05

    # Bonus weights
    win_bonus_weight: float = 1.0
    consistency_bonus_weight: float = 0.5
    sharpe_bonus_weight: float = 0.3
    trade_frequency_bonus: float = 0.2

    # Advanced parameters
    volatility_adjustment: float = 1.0
    regime_bonus_weight: float = 0.2
    momentum_bonus_weight: float = 0.1

    # Performance thresholds
    max_processing_time_ms: float = 100
    circuit_breaker_threshold: int = 5
    min_reward_quality: float = 0.3

    # Adaptation parameters
    confidence_decay: float = 0.98
    performance_smoothing: float = 0.95
    adaptive_learning_rate: float = 0.01


@module(**module_args(
    "RiskAdjustedReward",
    description="Deterministic multi-window feature extraction with circuit breaker, monitoring, and explainability.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=120,
))
class RiskAdjustedReward(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Bus-first risk-adjusted reward module:
    - Never assumes a default 10k; resolves balance/equity from SmartInfoBus/env.
    - Robust analytics + adaptation.
    - Contract-safe outputs even under error/fallback.
    """

    # ─────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────
    def __init__(
        self,
        config: Optional[RewardConfig | Dict[str, Any]] = None,
        genome: Optional[Dict[str, Any]] = None,
        env: Any = None,
        **kwargs,
    ):
        # Logger & bus BEFORE BaseModule init (BaseModule may call _initialize)
        self.logger = RotatingLogger(
            name="RiskAdjustedReward",
            log_path="logs/reward/risk_adjusted_reward.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,

        )
        self.smart_bus = InfoBusManager.get_instance()
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("RiskAdjustedReward", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Normalize config to dataclass and store as self.cfg (consistent with other modules)
        if isinstance(config, dict):
            self.cfg = RewardConfig(**config)
        elif config is None:
            self.cfg = RewardConfig()
        else:
            self.cfg = config

        # External env (optional)
        self.env = env

        # Core state placeholders (must exist before BaseModule)
        self.current_mode: RewardMode = RewardMode.TRAINING
        self.mode_start_time = datetime.datetime.now()

        # Reward state
        self._reward_history: deque[float] = deque(maxlen=self.cfg.history_size)
        self._pnl_history: deque[float] = deque(maxlen=self.cfg.history_size)
        self._trade_count_history: deque[int] = deque(maxlen=20)
        self._reward_components_history: deque[Dict[str, Any]] = deque(maxlen=50)
        self._performance_analytics: defaultdict = defaultdict(list)
        self._regime_performance: defaultdict = defaultdict(lambda: {"rewards": [], "trades": [], "pnl": []})
        self._regime_transition_rewards: defaultdict = defaultdict(list)
        self._volatility_performance: defaultdict = defaultdict(list)
        self._session_analytics: defaultdict = defaultdict(list)
        self._error_recovery_metrics: List[Dict[str, Any]] = []
        self.audit_trail: List[Dict[str, Any]] = []
        self._audit_log_size: int = self.cfg.history_size

        # Performance metrics
        self._sharpe_ratio = 0.0
        self._consistency_score = 0.0
        self._win_rate = 0.0
        self._avg_reward = 0.0
        self._reward_volatility = 0.0
        self._reward_quality = 0.5
        self._last_reward = 0.0
        self._last_reason = ""
        self._call_count = 0

        # Circuit breaker & health
        self.circuit_breaker = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",
            "threshold": int(self.cfg.circuit_breaker_threshold),
        }
        self._health_status = "healthy"
        self._last_health_check = time.time()

        # Genome (evolution)
        self._initialize_genome_parameters(genome)

        # Adaptive params
        self._adaptive_params = {
            "dynamic_penalty_scaling": 1.0,
            "regime_sensitivity": 1.0,
            "activity_threshold": 1.0,
            "risk_tolerance": 1.0,
            "learning_momentum": 0.0,
            "adaptation_confidence": 0.5,
        }

        # Balance baselines (bus-first; updated on first successful extraction)
        self._baseline_balance: Optional[float] = None
        self._last_balance_observed: Optional[float] = None

        # Important: let BaseModule wire and call _initialize()
        super().__init__(config=asdict(self.cfg))

        self.logger.info(
            format_operator_message(
                "[TARGET]",
                "REWARD_SYSTEM_INITIALIZED",
                details=f"History={self.cfg.history_size}",
                result="Reward system ready (bus-first balance)",
                context="reward_initialization",
            )
        )

        # Start monitoring after init
        self._start_monitoring()

    # ─────────────────────────────────────────────────────────────
    # Initialization hooks
    # ─────────────────────────────────────────────────────────────
    def _initialize(self) -> None:
        """Called by BaseModule after registration."""
        try:
            # Publish an initial status
            self.smart_bus.set(
                "reward_performance",
                {
                    "current_mode": self.current_mode.value,
                    "reward_quality": self._reward_quality,
                    "avg_reward": self._avg_reward,
                    "sharpe_ratio": self._sharpe_ratio,
                    "win_rate": self._win_rate,
                },
                module="RiskAdjustedReward",
                thesis="Initial reward system performance metrics",
            )
        except Exception as e:
            self.logger.error(f"Reward system initialization failed: {e}")

    def _start_monitoring(self) -> None:
        def monitoring_loop() -> None:
            while getattr(self, "_monitoring_active", True):
                try:
                    self._update_reward_health()
                    self._analyze_reward_effectiveness()
                    self._adapt_parameters()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Reward monitoring error: {e}")

        self._monitoring_active = True
        t = threading.Thread(target=monitoring_loop, daemon=True)
        t.start()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            reward_data = await self._extract_reward_data(**inputs)
            if not reward_data:
                return await self._handle_no_data_fallback()

            reward_result = await self._calculate_enhanced_reward(reward_data)
            analytics_result = await self._update_reward_analytics(reward_result, reward_data)
            adaptation_result = await self._update_adaptive_learning(reward_result)

            result = {**reward_result, **analytics_result, **adaptation_result}
            thesis = await self._generate_reward_thesis(reward_data, result)

            # Prepare contract-safe blobs
            shaped_payload = {
                "reward": result.get("shaped_reward", 0.0),
                "components": result.get("reward_components", {}),
                "calculation_method": result.get("calculation_method", "enhanced_async"),
                "timestamp": utcnow(),
            }

            analytics_payload = result.get(
                "reward_analytics",
                {
                    "performance_metrics": {
                        "sharpe_ratio": self._sharpe_ratio,
                        "consistency_score": self._consistency_score,
                        "win_rate": self._win_rate,
                        "avg_reward": self._avg_reward,
                        "reward_volatility": self._reward_volatility,
                        "reward_quality": self._reward_quality,
                    },
                    "component_analysis": {},
                    "regime_analysis": {},
                },
            )

            performance_payload = {
                "reward_quality": self._reward_quality,
                "sharpe_ratio": self._sharpe_ratio,
                "consistency_score": self._consistency_score,
                "win_rate": self._win_rate,
                "avg_reward": self._avg_reward,
                "reward_volatility": self._reward_volatility,
                "adaptive_params": self._adaptive_params.copy(),
                "health_status": self._health_status,
                "circuit_breaker_state": self.circuit_breaker["state"],
            }

            await self._update_reward_smart_bus(result, thesis)

            # Record success
            self._record_success((time.time() - start_time) * 1000.0)

            # Return contract fields
            result.update(
                {
                    "shaped_reward": shaped_payload,
                    "reward_components": result.get("reward_components", {}),
                    "reward_analytics": analytics_payload,
                    "reward_performance": performance_payload,
                    "_thesis": thesis,
                    "success": True,
                }
            )
            return result

        except Exception as e:
            return await self._handle_reward_error(e, start_time)

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        try:
            base_confidence = 0.8
            base_confidence += (self._reward_quality - 0.5) * 0.3

            if abs(self._sharpe_ratio) < 2.0:
                base_confidence += 0.1
            elif abs(self._sharpe_ratio) > 5.0:
                base_confidence -= 0.2

            if len(self._reward_history) >= 10:
                recent = list(self._reward_history)[-10:]
                reward_stability = 1.0 - (np.std(recent) / (abs(np.mean(recent)) + 0.1))
                base_confidence += reward_stability * 0.1

            if self.circuit_breaker["state"] == "OPEN":
                base_confidence *= 0.5
            elif self.circuit_breaker["failures"] > 0:
                base_confidence *= 0.8

            if self.current_mode == RewardMode.EMERGENCY:
                base_confidence *= 0.7
            elif self.current_mode == RewardMode.LIVE_TRADING:
                base_confidence += 0.1

            return float(np.clip(base_confidence, 0.1, 1.0))
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.6

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        try:
            current_reward = self._last_reward
            reward_trend = "neutral"

            if len(self._reward_history) >= 5:
                recent = list(self._reward_history)[-5:]
                slope = np.polyfit(range(len(recent)), recent, 1)[0]
                if slope > 0.05:
                    reward_trend = "improving"
                elif slope < -0.05:
                    reward_trend = "declining"

            recommendations = []

            if self._sharpe_ratio < 0:
                recommendations.append(
                    {
                        "action": "reduce_risk_exposure",
                        "reason": f"Negative Sharpe ratio: {self._sharpe_ratio:.2f}",
                        "priority": "high",
                    }
                )

            if len(self._reward_history) >= 10:
                recent_avg = float(np.mean(list(self._reward_history)[-10:]))
                if recent_avg < -0.1:
                    recommendations.append(
                        {
                            "action": "review_trading_strategy",
                            "reason": f"Poor recent performance: {recent_avg:.3f}",
                            "priority": "high",
                        }
                    )

            if len(self._reward_history) >= 10:
                reward_volatility = float(np.std(list(self._reward_history)[-10:]))
                if reward_volatility > 1.0:
                    recommendations.append(
                        {
                            "action": "stabilize_reward_variance",
                            "reason": f"High reward volatility: {reward_volatility:.3f}",
                            "priority": "medium",
                        }
                    )

            if self._adaptive_params.get("dynamic_penalty_scaling", 1.0) > 1.5:
                recommendations.append(
                    {
                        "action": "reduce_penalty_scaling",
                        "reason": "High penalty scaling detected",
                        "priority": "low",
                    }
                )

            return {
                "action_type": "reward_optimization",
                "current_reward": current_reward,
                "reward_trend": reward_trend,
                "recommendations": recommendations,
                "sharpe_ratio": self._sharpe_ratio,
                "reward_quality": self._reward_quality,
                "confidence": await self.calculate_confidence({}, **inputs),
                "timestamp": datetime.datetime.now().isoformat(),
                "system_health": {
                    "circuit_breaker": self.circuit_breaker["state"],
                    "mode": self.current_mode.value,
                    "health_status": self._health_status,
                },
            }
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                "action_type": "reward_optimization",
                "error": str(e),
                "recommendations": [{"action": "system_check", "reason": "Error in reward analysis", "priority": "high"}],
                "confidence": 0.1,
            }

    # ─────────────────────────────────────────────────────────────
    # Data extraction (bus-first; no hidden defaults)
    # ─────────────────────────────────────────────────────────────
    async def _extract_reward_data(self, **inputs) -> Optional[Dict[str, Any]]:
        try:
            trade_data = self.smart_bus.get("trade_data", "RiskAdjustedReward") or {}
            risk_metrics = self.smart_bus.get("risk_metrics", "RiskAdjustedReward") or {}
            market_context = self.smart_bus.get("market_context", "RiskAdjustedReward") or {}
            performance_data = self.smart_bus.get("performance_data", "RiskAdjustedReward") or {}
            env_cfg = self.smart_bus.get("environment_config", "RiskAdjustedReward") or {}

            # Map market context fields to what we need
            # Prefer canonical keys if present; map hints for compatibility
            regime = market_context.get("regime", market_context.get("session_canonical", "unknown"))
            volatility_level = market_context.get("volatility_level", market_context.get("volatility_hint", "medium"))

            recent_trades = trade_data.get("recent_trades", [])
            actions = inputs.get("actions")
            raw_reward_inputs = inputs.get("reward_inputs", {})

            # Do NOT set a balance here. We compute it in the calc step from bus/env to avoid 10k fallbacks.
            return {
                "trades": recent_trades,
                "risk_metrics": risk_metrics,
                "market_context": market_context,
                "performance_data": performance_data,
                "env_config": env_cfg,
                "regime": regime,
                "volatility_level": volatility_level,
                "consensus": float(market_context.get("consensus", 0.5)),
                "actions": actions,
                "raw_inputs": raw_reward_inputs,
                "timestamp": datetime.datetime.now().isoformat(),
                "step_idx": inputs.get("step_idx", self._call_count),
            }
        except Exception as e:
            self.logger.error(f"Failed to extract reward data: {e}")
            return None

    # ─────────────────────────────────────────────────────────────
    # Balance resolution (bus/env first)
    # ─────────────────────────────────────────────────────────────
    def _resolve_balance(self, reward_data: Dict[str, Any]) -> Tuple[float, Optional[float]]:
        """
        Returns (balance_now, baseline_balance).
        baseline_balance is the fixed anchor for normalized components (once discovered).
        Priority:
          1) risk_metrics: balance/equity/account_equity/cash
          2) account_state / performance_data / environment_config
          3) env attributes (env.balance/env.equity/env.initial_balance)
          4) cfg.initial_balance (if explicitly set)
        Never fabricate numbers beyond that; final denominator always guarded by epsilon.
        """
        rm = reward_data.get("risk_metrics", {}) or {}
        perf = reward_data.get("performance_data", {}) or {}
        env_cfg = reward_data.get("env_config", {}) or {}

        # 1) Direct risk metrics (most authoritative during runtime)
        candidates = [
            rm.get("balance"),
            rm.get("equity"),
            rm.get("account_equity"),
            rm.get("cash"),
            rm.get("account_balance"),
        ]

        # 2) Account/perf/env config anchors (initials/baselines)
        candidates += [
            perf.get("balance"),
            perf.get("equity"),
            perf.get("initial_balance"),
            perf.get("starting_balance"),
            env_cfg.get("initial_balance"),
            env_cfg.get("starting_balance"),
        ]

        # 3) Env attributes (if present)
        if self.env is not None:
            for name in ("balance", "equity", "initial_balance", "starting_balance"):
                if hasattr(self.env, name):
                    try:
                        candidates.append(float(getattr(self.env, name)))
                    except Exception:
                        pass

        # 4) Config (only if explicitly provided)
        if self.cfg.initial_balance is not None:
            candidates.append(float(self.cfg.initial_balance))

        # Filter to finite positives
        vals = [float(x) for x in candidates if isinstance(x, (int, float)) and np.isfinite(x)]
        balance_now = vals[0] if vals else 0.0

        # Establish baseline once if we see a plausible starting balance
        if self._baseline_balance is None:
            # Prefer explicit initials if present
            initials = [
                rm.get("initial_balance"),
                perf.get("initial_balance"),
                perf.get("starting_balance"),
                env_cfg.get("initial_balance"),
                env_cfg.get("starting_balance"),
            ]
            initials = [float(x) for x in initials if isinstance(x, (int, float)) and np.isfinite(x) and x > 0]
            if initials:
                self._baseline_balance = initials[0]
            elif balance_now > 0:
                # Fall back to the first observed balance as baseline if nothing else is declared
                self._baseline_balance = float(balance_now)

        self._last_balance_observed = balance_now
        return float(balance_now), (float(self._baseline_balance) if self._baseline_balance else None)

    # ─────────────────────────────────────────────────────────────
    # Core reward calculation
    # ─────────────────────────────────────────────────────────────
    async def _calculate_enhanced_reward(self, reward_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            trades = reward_data.get("trades", [])
            consensus = float(reward_data.get("consensus", 0.5))
            actions = reward_data.get("actions")
            regime = reward_data.get("regime", "unknown")
            volatility_level = reward_data.get("volatility_level", "medium")
            rm = reward_data.get("risk_metrics", {}) or {}

            # Resolve balances from the bus/env
            balance_now, baseline_balance = self._resolve_balance(reward_data)
            denom = float(baseline_balance if (baseline_balance and baseline_balance > 0) else max(1e-9, balance_now))

            # Extract drawdown from bus; safe default to 0.0
            drawdown = float(rm.get("current_drawdown", rm.get("drawdown", 0.0)) or 0.0)

            # Realised P&L for this step (if provided per trade; else 0)
            realised_pnl = float(sum(t.get("pnl", 0.0) for t in trades))
            base_component = realised_pnl / denom

            # Start from P&L (keeps same semantics but normalized is available)
            reward = realised_pnl

            # Components (for audit)
            components: Dict[str, Any] = {
                "pnl": realised_pnl,
                "base_component": base_component,
                "balance_now": balance_now,
                "baseline_balance": baseline_balance,
                "drawdown": drawdown,
                "consensus": consensus,
                "trades_count": len(trades),
                "timestamp": reward_data.get("timestamp", utcnow()),
                "step_idx": reward_data.get("step_idx", 0),
                "market_regime": regime,
                "volatility_level": volatility_level,
                # initialized subcomponents
                "drawdown_penalty": 0.0,
                "risk_penalty": 0.0,
                "tail_penalty": 0.0,
                "mistake_penalty": 0.0,
                "no_trade_penalty": 0.0,
                "win_bonus": 0.0,
                "activity_bonus": 0.0,
                "consistency_bonus": 0.0,
                "sharpe_bonus": 0.0,
                "regime_bonus": 0.0,
                "volatility_adjustment": 0.0,
                "consensus_factor": 0.5 + consensus,
            }

            # ── PENALTIES ─────────────────────────────────────────
            if drawdown > 0.05:
                dd_penalty = (drawdown ** 2) * self.cfg.dd_pen_weight
                if regime == "volatile":
                    dd_penalty *= 0.8
                elif regime == "trending":
                    dd_penalty *= 1.2
                reward -= dd_penalty * self._adaptive_params["dynamic_penalty_scaling"]
                components["drawdown_penalty"] = dd_penalty

            if actions is not None:
                risk_penalty = min(float(np.linalg.norm(actions) * self.cfg.risk_pen_weight), 0.2)
                vol_mult = {"low": 1.2, "medium": 1.0, "high": 0.8, "extreme": 0.6}
                risk_penalty *= vol_mult.get(volatility_level, 1.0)
                reward -= risk_penalty
                components["risk_penalty"] = risk_penalty

            if trades:
                losses = [float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) < 0]
                if losses:
                    tail_penalty = abs(float(np.mean(losses))) * self.cfg.tail_pen_weight * 0.1
                    extreme_losses = [l for l in losses if l < -100]
                    if extreme_losses:
                        tail_penalty *= 1.5
                    reward -= tail_penalty
                    components["tail_penalty"] = tail_penalty

            mistake_penalty = await self._calculate_mistake_penalty(reward_data)
            if mistake_penalty > 0:
                reward -= mistake_penalty
                components["mistake_penalty"] = mistake_penalty

            # ── BONUSES ───────────────────────────────────────────
            if trades:
                win_ratio = sum(1 for t in trades if t.get("pnl", 0.0) > 0) / len(trades)
                win_bonus = win_ratio * self.cfg.win_bonus_weight
                if len(self._pnl_history) >= 3:
                    recent_wins = [p > 0 for p in list(self._pnl_history)[-3:]]
                    if all(recent_wins):
                        win_bonus *= 1.3
                reward += win_bonus
                components["win_bonus"] = win_bonus

                activity_bonus = min(len(trades) * 0.1, 0.3)
                if realised_pnl > 0:
                    activity_bonus *= 1.2
                reward += activity_bonus
                components["activity_bonus"] = activity_bonus
            else:
                no_trade_penalty = self.cfg.no_trade_penalty_weight * self._adaptive_params["activity_threshold"]
                if drawdown > 0.1:
                    no_trade_penalty *= 0.3
                elif volatility_level == "extreme":
                    no_trade_penalty *= 0.5
                reward -= no_trade_penalty
                components["no_trade_penalty"] = no_trade_penalty

            # ── SOPHISTICATED BONUSES ─────────────────────────────
            consistency_bonus = await self._calculate_consistency_bonus()
            sharpe_bonus = await self._calculate_enhanced_sharpe_bonus()
            regime_bonus = await self._calculate_regime_bonus(regime, realised_pnl)
            volatility_adjustment = await self._calculate_volatility_adjustment(volatility_level, realised_pnl)

            reward += consistency_bonus + sharpe_bonus + regime_bonus + volatility_adjustment
            components.update(
                {
                    "consistency_bonus": consistency_bonus,
                    "sharpe_bonus": sharpe_bonus,
                    "regime_bonus": regime_bonus,
                    "volatility_adjustment": volatility_adjustment,
                }
            )

            # Consensus factor
            reward *= components["consensus_factor"]

            # Final clip for training stability
            final_reward = float(np.clip(reward, -10.0, 10.0))
            components["final_reward"] = final_reward
            components["method"] = "enhanced_async_calculation"

            # Update module state
            await self._update_reward_state(trades, realised_pnl, final_reward)

            # Logging cadence
            if (self._call_count % 10 == 1) or (abs(final_reward) > 0.2) or (self.circuit_breaker["state"] == "OPEN"):
                self.logger.info(
                    format_operator_message(
                        "[TARGET]",
                        "REWARD_CALCULATED",
                        reward=f"{final_reward:.4f}",
                        pnl=f"{realised_pnl:.2f}",
                        trades=len(trades),
                        drawdown=f"{drawdown:.1%}",
                        balance_now=f"{balance_now:.2f}",
                        baseline=f"{(baseline_balance if baseline_balance else 0.0):.2f}",
                        regime=regime,
                        volatility=volatility_level,
                        context="reward_calculation",
                    )
                )

            await self._record_audit(components)

            return {"shaped_reward": final_reward, "reward_components": components, "calculation_method": "enhanced_async"}

        except Exception as e:
            self.logger.error(f"Enhanced reward calculation failed: {e}")
            raise

    async def _calculate_mistake_penalty(self, reward_data: Dict[str, Any]) -> float:
        try:
            mistake_data = self.smart_bus.get("mistake_memory", "RiskAdjustedReward")
            if mistake_data:
                score = float(mistake_data.get("current_score", 0.0))
                penalty = score * self.cfg.mistake_pen_weight * self._adaptive_params.get("dynamic_penalty_scaling", 1.0)
                return float(penalty)
            if self.env and hasattr(self.env, "mistake_memory"):
                mm = float(self.env.mistake_memory.get_observation_components()[0])
                return float(mm * self.cfg.mistake_pen_weight)
        except Exception as e:
            self.logger.warning(f"Mistake penalty calculation failed: {e}")
        return 0.0

    async def _calculate_enhanced_sharpe_bonus(self) -> float:
        if len(self._reward_history) < 5:
            return 0.0
        rewards = np.array(self._reward_history, dtype=np.float32)
        mean_reward = float(rewards.mean())
        std_reward = float(rewards.std())
        min_std = max(0.1, abs(mean_reward) * 0.1)
        std_reward = max(std_reward, min_std)
        sharpe = mean_reward / std_reward * np.sqrt(min(len(rewards), 252))
        if hasattr(self, "_last_regime"):
            regime_multiplier = {"trending": 1.2, "ranging": 1.0, "volatile": 0.8, "unknown": 0.9}
            sharpe *= regime_multiplier.get(self._last_regime, 1.0)
        sensitivity = self._adaptive_params.get("regime_sensitivity", 1.0)
        normalized_sharpe = np.tanh(sharpe / (6.0 / sensitivity))
        bonus = float(np.clip(normalized_sharpe * self.cfg.sharpe_bonus_weight, -0.5, 0.5))
        self._sharpe_ratio = float(sharpe)
        return bonus

    async def _calculate_consistency_bonus(self) -> float:
        if len(self._pnl_history) < 3:
            return 0.0
        recent_pnls = list(self._pnl_history)[-10:]
        positive_ratio = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
        consistency_score = positive_ratio**2
        if len(recent_pnls) >= 5:
            early_half = recent_pnls[: len(recent_pnls) // 2]
            late_half = recent_pnls[len(recent_pnls) // 2 :]
            if early_half and late_half:
                early_ratio = sum(1 for p in early_half if p > 0) / len(early_half)
                late_ratio = sum(1 for p in late_half if p > 0) / len(late_half)
                momentum = late_ratio - early_ratio
                consistency_score *= 1.0 + momentum * 0.2
        if len(recent_pnls) >= 5:
            streak = 0
            for pnl in reversed(recent_pnls):
                if pnl > 0:
                    streak += 1
                else:
                    break
            if streak >= 3:
                consistency_score *= 1.0 + streak * 0.1
        self._consistency_score = float(consistency_score)
        return float(consistency_score * self.cfg.consistency_bonus_weight)

    async def _calculate_regime_bonus(self, regime: str, pnl: float) -> float:
        self._last_regime = regime
        self._regime_performance[regime]["pnl"].append(pnl)
        regime_bonus = 0.0
        sensitivity = self._adaptive_params.get("regime_sensitivity", 1.0)

        if regime == "trending" and pnl > 0:
            regime_bonus = min(pnl / 100.0, 0.2) * self.cfg.regime_bonus_weight * sensitivity
        elif regime == "ranging" and abs(pnl) < 20:
            regime_bonus = 0.1 * self.cfg.regime_bonus_weight * sensitivity
        elif regime == "volatile":
            if pnl < -50:
                regime_bonus = -0.15 * self.cfg.regime_bonus_weight * sensitivity
            elif 0 < pnl < 30:
                regime_bonus = 0.1 * self.cfg.regime_bonus_weight * sensitivity

        if len(self._regime_performance) > 1:
            self._regime_transition_rewards[regime].append(pnl)

        return float(regime_bonus)

    async def _calculate_volatility_adjustment(self, volatility_level: str, pnl: float) -> float:
        self._volatility_performance[volatility_level].append(pnl)
        vol_multipliers = {"low": 1.1, "medium": 1.0, "high": 0.9, "extreme": 0.8}
        base_adjustment = (vol_multipliers.get(volatility_level, 1.0) - 1.0) * abs(pnl) * 0.1
        if volatility_level in ["high", "extreme"] and pnl > 0:
            base_adjustment += pnl * 0.05
        adaptive_factor = self._adaptive_params.get("risk_tolerance", 1.0)
        return float(base_adjustment * self.cfg.volatility_adjustment * adaptive_factor)

    async def _update_reward_state(self, trades: List[dict], pnl: float, reward: float) -> None:
        self._pnl_history.append(float(pnl))
        self._trade_count_history.append(int(len(trades)))
        self._reward_history.append(float(reward))
        if trades:
            for trade in trades:
                self._update_trading_metrics(trade)  # mixin sync is fine here
        session = datetime.datetime.now().strftime("%Y-%m-%d")
        self._session_analytics[session].append({"timestamp": time.time(), "pnl": pnl, "reward": reward, "trades": len(trades)})
        self._last_reward = float(reward)
        self._last_reason = "trade" if trades else "no-trade"
        self._call_count += 1

    async def _record_audit(self, details: Dict[str, Any]) -> None:
        details["timestamp"] = utcnow()
        details["call_count"] = self._call_count
        details["mode"] = self.current_mode.value
        details["circuit_breaker_state"] = self.circuit_breaker["state"]
        self.audit_trail.append(details)
        if len(self.audit_trail) > self._audit_log_size:
            self.audit_trail = self.audit_trail[-self._audit_log_size :]
        self._reward_components_history.append(
            {
                "timestamp": details["timestamp"],
                "final_reward": details.get("final_reward", 0.0),
                "pnl": details.get("pnl", 0.0),
                "regime": details.get("market_regime", "unknown"),
                "trades_count": details.get("trades_count", 0),
                "volatility_level": details.get("volatility_level", "medium"),
            }
        )

    # ─────────────────────────────────────────────────────────────
    # Analytics / Adaptation
    # ─────────────────────────────────────────────────────────────
    async def _update_reward_analytics(self, reward_result: Dict[str, Any], reward_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            await self._update_performance_metrics()
            component_analysis = await self._analyze_component_effectiveness(reward_result)
            regime_analysis = await self._update_regime_analysis(reward_data)
            return {
                "reward_analytics": {
                    "performance_metrics": {
                        "sharpe_ratio": self._sharpe_ratio,
                        "consistency_score": self._consistency_score,
                        "win_rate": self._win_rate,
                        "avg_reward": self._avg_reward,
                        "reward_volatility": self._reward_volatility,
                        "reward_quality": self._reward_quality,
                    },
                    "component_analysis": component_analysis,
                    "regime_analysis": regime_analysis,
                }
            }
        except Exception as e:
            self.logger.error(f"Reward analytics update failed: {e}")
            return {"reward_analytics": {"error": str(e)}}

    async def _update_performance_metrics(self) -> None:
        try:
            # Mixins provide _trades_processed/_winning_trades (ensure defaults)
            trades_processed = getattr(self, "_trades_processed", 0)
            winning_trades = getattr(self, "_winning_trades", 0)
            if trades_processed > 0:
                self._win_rate = float(winning_trades / trades_processed)
            if self._reward_history:
                arr = np.array(self._reward_history, dtype=np.float32)
                self._avg_reward = float(arr.mean())
                self._reward_volatility = float(arr.std())
            if len(self._reward_history) >= 10:
                recent = list(self._reward_history)[-10:]
                positive_ratio = sum(1 for r in recent if r > 0) / len(recent)
                stability = 1.0 - (np.std(recent) / (abs(np.mean(recent)) + 1e-8))
                self._reward_quality = float((positive_ratio + max(0.0, float(stability))) / 2.0)
        except Exception as e:
            self.logger.warning(f"Performance metrics update failed: {e}")

    async def _analyze_component_effectiveness(self, reward_result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            components = reward_result.get("reward_components", {})
            component_magnitudes: Dict[str, float] = {}
            total_magnitude = 0.0
            for key, value in components.items():
                if key.endswith("_penalty") or key.endswith("_bonus") or key == "volatility_adjustment":
                    mag = abs(float(value))
                    component_magnitudes[key] = mag
                    total_magnitude += mag
            contributions: Dict[str, float] = {}
            if total_magnitude > 0:
                for k, mag in component_magnitudes.items():
                    contributions[k] = float(mag / total_magnitude)
            return {
                "component_magnitudes": component_magnitudes,
                "component_contributions": contributions,
                "total_component_magnitude": float(total_magnitude),
            }
        except Exception as e:
            self.logger.warning(f"Component effectiveness analysis failed: {e}")
            return {}

    async def _update_regime_analysis(self, reward_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            regime = reward_data.get("regime", "unknown")
            stats: Dict[str, Any] = {}
            for reg, data in self._regime_performance.items():
                rewards = data.get("rewards", [])
                # Keep the compatibility: rewards may not be populated; use pnl instead
                pnls = data.get("pnl", [])
                window = pnls[-20:] if pnls else rewards[-20:]
                if window:
                    stats[reg] = {
                        "avg_reward": float(np.mean(window)),
                        "reward_count": int(len(window)),
                        "win_rate": float(sum(1 for r in window if r > 0) / len(window)),
                    }
            return {"current_regime": regime, "regime_stats": stats}
        except Exception as e:
            self.logger.warning(f"Regime analysis update failed: {e}")
            return {}

    async def _update_adaptive_learning(self, reward_result: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if len(self._reward_history) >= 20:
                recent = list(self._reward_history)[-20:]
                avg_reward = float(np.mean(recent))
                lr = float(self.cfg.adaptive_learning_rate)
                if avg_reward < -0.5:
                    self._adaptive_params["dynamic_penalty_scaling"] = min(
                        1.5, self._adaptive_params["dynamic_penalty_scaling"] * (1 + lr)
                    )
                elif avg_reward > 0.5:
                    self._adaptive_params["dynamic_penalty_scaling"] = max(
                        0.5, self._adaptive_params["dynamic_penalty_scaling"] * (1 - lr * 0.5)
                    )

            if len(self._trade_count_history) >= 10:
                avg_activity = float(np.mean(self._trade_count_history))
                if avg_activity < 0.3:
                    self._adaptive_params["activity_threshold"] = min(2.0, self._adaptive_params["activity_threshold"] * 1.05)
                elif avg_activity > 3.0:
                    self._adaptive_params["activity_threshold"] = max(0.5, self._adaptive_params["activity_threshold"] * 0.98)

            if self._reward_quality > 0.7:
                self._adaptive_params["adaptation_confidence"] = min(
                    1.0, self._adaptive_params["adaptation_confidence"] * 1.01
                )
            elif self._reward_quality < 0.3:
                self._adaptive_params["adaptation_confidence"] = max(
                    0.1, self._adaptive_params["adaptation_confidence"] * 0.99
                )

            return {
                "adaptive_learning": {
                    "adaptive_params": self._adaptive_params.copy(),
                    "reward_quality": self._reward_quality,
                    "learning_effectiveness": self._adaptive_params["adaptation_confidence"],
                }
            }
        except Exception as e:
            self.logger.warning(f"Adaptive learning update failed: {e}")
            return {"adaptive_learning": {"error": str(e)}}

    async def _generate_reward_thesis(self, reward_data: Dict[str, Any], result: Dict[str, Any]) -> str:
        try:
            reward = result.get("shaped_reward", 0.0)
            components = result.get("reward_components", {})
            pnl = float(components.get("pnl", 0.0))
            trades_count = int(components.get("trades_count", 0))
            regime = reward_data.get("regime", "unknown")
            volatility = reward_data.get("volatility_level", "medium")

            parts = [
                f"Reward: {reward:.4f} from {pnl:.2f} PnL over {trades_count} trades",
                f"Context: {regime.upper()} regime, {volatility.upper()} vol",
            ]

            if self._reward_quality > 0.7:
                parts.append(f"Quality: HIGH ({self._reward_quality:.2f}) | WinRate {self._win_rate:.1%}")
            elif self._reward_quality < 0.3:
                parts.append(f"Quality: LOW ({self._reward_quality:.2f})")
            else:
                parts.append(f"Quality: MODERATE ({self._reward_quality:.2f})")

            majors = []
            for k, v in components.items():
                if (k.endswith("_penalty") or k.endswith("_bonus")) and abs(float(v)) > 0.01:
                    majors.append(f"{k} {float(v):.3f}")
            if majors:
                parts.append("Major: " + ", ".join(majors[:3]))

            adapt_conf = self._adaptive_params.get("adaptation_confidence", 0.5)
            if adapt_conf > 0.8:
                parts.append("Adaptation: HIGH confidence")
            elif adapt_conf < 0.3:
                parts.append("Adaptation: LOW confidence")

            if self.circuit_breaker["state"] == "OPEN":
                parts.append("ALERT: Circuit breaker OPEN")
            elif self._health_status == "warning":
                parts.append("WARNING: Health degraded")

            return " | ".join(parts)
        except Exception as e:
            return f"Reward thesis generation failed: {e} - core calculation ok"

    async def _update_reward_smart_bus(self, result: Dict[str, Any], thesis: str) -> None:
        try:
            reward_data = {
                "reward": result.get("shaped_reward", 0.0),
                "components": result.get("reward_components", {}),
                "calculation_method": result.get("calculation_method", "enhanced_async"),
                "timestamp": utcnow(),
            }
            self.smart_bus.set("shaped_reward", reward_data, module="RiskAdjustedReward", thesis=thesis)

            self.smart_bus.set(
                "reward_components",
                result.get("reward_components", {}),
                module="RiskAdjustedReward",
                thesis="Reward components breakdown",
            )

            self.smart_bus.set(
                "reward_analytics",
                result.get("reward_analytics", {}),
                module="RiskAdjustedReward",
                thesis="Reward analytics & component effectiveness",
            )

            performance_payload = {
                "reward_quality": self._reward_quality,
                "sharpe_ratio": self._sharpe_ratio,
                "consistency_score": self._consistency_score,
                "win_rate": self._win_rate,
                "avg_reward": self._avg_reward,
                "reward_volatility": self._reward_volatility,
                "adaptive_params": self._adaptive_params.copy(),
                "health_status": self._health_status,
                "circuit_breaker_state": self.circuit_breaker["state"],
                "last_balance_observed": self._last_balance_observed,
                "baseline_balance": self._baseline_balance,
            }
            self.smart_bus.set(
                "reward_performance",
                performance_payload,
                module="RiskAdjustedReward",
                thesis="Real-time reward system performance & health",
            )
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    # ─────────────────────────────────────────────────────────────
    # Errors / Fallbacks
    # ─────────────────────────────────────────────────────────────
    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        self.logger.warning("No reward data available - using fallback calculation")
        fallback_reward = -0.1
        thesis = "No reward data available - fallback applied"
        return {
            "shaped_reward": {
                "reward": fallback_reward,
                "components": {"fallback_penalty": -0.1, "reason": "no_reward_data"},
                "calculation_method": "fallback",
                "timestamp": utcnow(),
            },
            "reward_components": {"fallback_penalty": -0.1, "reason": "no_reward_data"},
            "reward_analytics": {
                "performance_metrics": {
                    "sharpe_ratio": self._sharpe_ratio,
                    "consistency_score": self._consistency_score,
                    "win_rate": self._win_rate,
                    "avg_reward": self._avg_reward,
                    "reward_volatility": self._reward_volatility,
                    "reward_quality": self._reward_quality,
                }
            },
            "reward_performance": {
                "reward_quality": self._reward_quality,
                "sharpe_ratio": self._sharpe_ratio,
                "consistency_score": self._consistency_score,
                "win_rate": self._win_rate,
                "avg_reward": self._avg_reward,
                "reward_volatility": self._reward_volatility,
                "adaptive_params": self._adaptive_params.copy(),
                "health_status": self._health_status,
                "circuit_breaker_state": self.circuit_breaker["state"],
            },
            "_thesis": thesis,
            "success": True,
            "fallback_reason": "no_reward_data",
        }

    async def _handle_reward_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        processing_time = (time.time() - start_time) * 1000.0
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()
        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
            self.circuit_breaker["state"] = "OPEN"
            self._health_status = "warning"

        explanation = self.english_explainer.explain_error("RiskAdjustedReward", str(error), "reward calculation")
        self.logger.error(
            format_operator_message(
                "[CRASH]",
                "REWARD_CALCULATION_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                circuit_breaker_state=self.circuit_breaker["state"],
                context="reward_error",
            )
        )

        self._error_recovery_metrics.append(
            {
                "timestamp": time.time(),
                "error_type": type(error).__name__,
                "processing_time": processing_time,
                "circuit_breaker_state": self.circuit_breaker["state"],
            }
        )
        self._record_failure(error)
        return self._create_error_fallback_response(f"error: {str(error)}")

    def _create_error_fallback_response(self, reason: str) -> Dict[str, Any]:
        error_reward = -0.5 if self.circuit_breaker["state"] == "OPEN" else -0.2
        thesis = f"Reward error fallback: {reason}"
        return {
            "shaped_reward": {
                "reward": error_reward,
                "components": {"error_penalty": error_reward, "reason": reason},
                "calculation_method": "error_fallback",
                "timestamp": utcnow(),
            },
            "reward_components": {"error_penalty": error_reward, "reason": reason},
            "reward_analytics": {
                "performance_metrics": {
                    "sharpe_ratio": self._sharpe_ratio,
                    "consistency_score": self._consistency_score,
                    "win_rate": self._win_rate,
                    "avg_reward": self._avg_reward,
                    "reward_volatility": self._reward_volatility,
                    "reward_quality": self._reward_quality,
                }
            },
            "reward_performance": {
                "reward_quality": self._reward_quality,
                "sharpe_ratio": self._sharpe_ratio,
                "consistency_score": self._consistency_score,
                "win_rate": self._win_rate,
                "avg_reward": self._avg_reward,
                "reward_volatility": self._reward_volatility,
                "adaptive_params": self._adaptive_params.copy(),
                "health_status": self._health_status,
                "circuit_breaker_state": self.circuit_breaker["state"],
            },
            "_thesis": thesis,
            "success": False,
            "circuit_breaker_state": self.circuit_breaker["state"],
            "fallback_reason": reason,
        }

    # ─────────────────────────────────────────────────────────────
    # Health / Evolution / State IO
    # ─────────────────────────────────────────────────────────────
    def _update_reward_health(self) -> None:
        try:
            if self._reward_quality < self.cfg.min_reward_quality:
                self._health_status = "warning"
            else:
                self._health_status = "healthy"

            if self.circuit_breaker["state"] == "OPEN":
                self._health_status = "warning"

            if self._reward_volatility > 2.0:
                self._health_status = "warning"

            self._last_health_check = time.time()
        except Exception as e:
            self.logger.error(f"Reward health check failed: {e}")
            self._health_status = "warning"

    # ─────────────────────────────────────────────────────────────
    # Performance bookkeeping helpers (fix for Pylance errors)
    # ─────────────────────────────────────────────────────────────
    def _record_success(self, processing_time: float) -> None:
        """Record a successful reward calculation and reset breaker if needed."""
        try:
            self.performance_tracker.record_metric(
                'RiskAdjustedReward', 'reward_calculation', float(processing_time), True
            )
        except Exception as e:
            # Non-fatal; keep logs quiet in production
            self.logger.debug(f"Performance tracking (success) failed: {e}")

        # Reset circuit breaker on success
        if self.circuit_breaker.get('state') == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception) -> None:
        """Record a failed reward calculation."""
        try:
            self.performance_tracker.record_metric(
                'RiskAdjustedReward', 'reward_calculation', 0.0, False
            )
        except Exception as e:
            self.logger.debug(f"Performance tracking (failure) failed: {e}")


    def _analyze_reward_effectiveness(self) -> None:
        try:
            if len(self._reward_history) >= 10:
                recent = list(self._reward_history)[-10:]
                positive_ratio = sum(1 for r in recent if r > 0) / len(recent)
                if positive_ratio > 0.8:
                    self.logger.info(
                        format_operator_message(
                            "[TARGET]",
                            "HIGH_REWARD_EFFECTIVENESS",
                            positive_ratio=f"{positive_ratio:.2f}",
                            avg_reward=f"{float(np.mean(recent)):.4f}",
                            context="reward_analysis",
                        )
                    )
                elif positive_ratio < 0.2:
                    self.logger.warning(
                        format_operator_message(
                            "[WARN]",
                            "LOW_REWARD_EFFECTIVENESS",
                            positive_ratio=f"{positive_ratio:.2f}",
                            avg_reward=f"{float(np.mean(recent)):.4f}",
                            context="reward_analysis",
                        )
                    )
        except Exception as e:
            self.logger.error(f"Reward effectiveness analysis failed: {e}")

    def _adapt_parameters(self) -> None:
        try:
            if len(self._regime_performance) >= 2:
                regime_rewards: Dict[str, float] = {}
                for regime, data in self._regime_performance.items():
                    if data["rewards"]:
                        regime_rewards[regime] = float(np.mean(data["rewards"][-10:]))
                if regime_rewards:
                    var = float(np.var(list(regime_rewards.values())))
                    if var > 0.1:
                        self._adaptive_params["regime_sensitivity"] = min(
                            1.5, self._adaptive_params["regime_sensitivity"] * 1.001
                        )
                    else:
                        self._adaptive_params["regime_sensitivity"] = max(
                            0.7, self._adaptive_params["regime_sensitivity"] * 0.9999
                        )
        except Exception as e:
            self.logger.warning(f"Parameter adaptation failed: {e}")

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]) -> None:
        if genome:
            g = {
                "initial_balance": float(genome.get("initial_balance", self.cfg.initial_balance or 0.0))
                if genome.get("initial_balance", None) is not None
                else self.cfg.initial_balance,
                "history_size": int(genome.get("history_size", self.cfg.history_size)),
                "min_trade_bonus": float(genome.get("min_trade_bonus", self.cfg.min_trade_bonus)),
                "regime_weights": list(genome.get("regime_weights", self.cfg.regime_weights)),
                "dd_pen_weight": float(genome.get("dd_pen_weight", self.cfg.dd_pen_weight)),
                "risk_pen_weight": float(genome.get("risk_pen_weight", self.cfg.risk_pen_weight)),
                "tail_pen_weight": float(genome.get("tail_pen_weight", self.cfg.tail_pen_weight)),
                "mistake_pen_weight": float(genome.get("mistake_pen_weight", self.cfg.mistake_pen_weight)),
                "no_trade_penalty_weight": float(
                    genome.get("no_trade_penalty_weight", self.cfg.no_trade_penalty_weight)
                ),
                "win_bonus_weight": float(genome.get("win_bonus_weight", self.cfg.win_bonus_weight)),
                "consistency_bonus_weight": float(
                    genome.get("consistency_bonus_weight", self.cfg.consistency_bonus_weight)
                ),
                "sharpe_bonus_weight": float(genome.get("sharpe_bonus_weight", self.cfg.sharpe_bonus_weight)),
                "trade_frequency_bonus": float(genome.get("trade_frequency_bonus", self.cfg.trade_frequency_bonus)),
                "volatility_adjustment": float(genome.get("volatility_adjustment", self.cfg.volatility_adjustment)),
                "regime_bonus_weight": float(genome.get("regime_bonus_weight", self.cfg.regime_bonus_weight)),
                "momentum_bonus_weight": float(genome.get("momentum_bonus_weight", self.cfg.momentum_bonus_weight)),
                "confidence_decay": float(genome.get("confidence_decay", self.cfg.confidence_decay)),
                "performance_smoothing": float(genome.get("performance_smoothing", self.cfg.performance_smoothing)),
            }
            # Push into cfg where applicable
            for k, v in g.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
            self.genome = g
            self.regime_weights = np.array(self.cfg.regime_weights, dtype=np.float32)
            if self.regime_weights.sum() > 0:
                self.regime_weights = self.regime_weights / self.regime_weights.sum()
            else:
                self.regime_weights = np.array([0.3, 0.4, 0.3], dtype=np.float32)
        else:
            self.genome = {
                "initial_balance": self.cfg.initial_balance,
                "history_size": self.cfg.history_size,
                "min_trade_bonus": self.cfg.min_trade_bonus,
                "regime_weights": self.cfg.regime_weights,
                "dd_pen_weight": self.cfg.dd_pen_weight,
                "risk_pen_weight": self.cfg.risk_pen_weight,
                "tail_pen_weight": self.cfg.tail_pen_weight,
                "mistake_pen_weight": self.cfg.mistake_pen_weight,
                "no_trade_penalty_weight": self.cfg.no_trade_penalty_weight,
                "win_bonus_weight": self.cfg.win_bonus_weight,
                "consistency_bonus_weight": self.cfg.consistency_bonus_weight,
                "sharpe_bonus_weight": self.cfg.sharpe_bonus_weight,
                "trade_frequency_bonus": self.cfg.trade_frequency_bonus,
                "volatility_adjustment": self.cfg.volatility_adjustment,
                "regime_bonus_weight": self.cfg.regime_bonus_weight,
                "momentum_bonus_weight": self.cfg.momentum_bonus_weight,
                "confidence_decay": self.cfg.confidence_decay,
                "performance_smoothing": self.cfg.performance_smoothing,
            }
            self.regime_weights = np.array(self.cfg.regime_weights, dtype=np.float32)

    # Evolution helpers (kept for compatibility)
    def get_genome(self) -> Dict[str, Any]:
        return self.genome.copy()

    def set_genome(self, genome: Dict[str, Any]) -> None:
        for key, value in genome.items():
            if hasattr(self.cfg, key):
                if key == "regime_weights":
                    weights = np.array(value, dtype=np.float32)
                    self.regime_weights = weights / (weights.sum() + 1e-8) if weights.sum() > 0 else np.array(
                        [0.3, 0.4, 0.3], dtype=np.float32
                    )
                    setattr(self.cfg, key, self.regime_weights.tolist())
                else:
                    setattr(self.cfg, key, value)
        self.genome.update(genome)

    def mutate(self, mutation_rate: float = 0.2) -> None:
        g = self.genome.copy()
        mutations = []
        if np.random.rand() < mutation_rate:
            old = np.array(g["regime_weights"])
            nw = old + np.random.normal(0, 0.1, size=3)
            nw = np.clip(nw, 0.0, 1.0)
            nw = nw / (nw.sum() + 1e-8)
            g["regime_weights"] = nw.tolist()
            mutations.append(f"regime_weights: {old} → {nw}")
        perf_factor = max(0.5, min(2.0, 1.0 + float(self._avg_reward) * 2.0))
        for name, std, mx in [
            ("dd_pen_weight", 0.2, 5.0),
            ("risk_pen_weight", 0.1, 2.0),
            ("tail_pen_weight", 0.1, 2.0),
            ("win_bonus_weight", 0.1, 3.0),
            ("consistency_bonus_weight", 0.1, 2.0),
            ("sharpe_bonus_weight", 0.1, 2.0),
        ]:
            if np.random.rand() < mutation_rate * perf_factor:
                old_val = float(g[name])
                g[name] = float(np.clip(old_val + np.random.normal(0, std), 0.0, mx))
                mutations.append(f"{name}: {old_val:.3f} → {g[name]:.3f}")
        if mutations:
            self.logger.info(
                format_operator_message(
                    "🧬", "REWARD_SYSTEM_MUTATION", changes=", ".join(mutations), performance_factor=f"{perf_factor:.2f}", context="evolution"
                )
            )
        self.set_genome(g)

    # State IO / Monitoring
    def reset(self) -> None:
        self._reward_history.clear()
        self._pnl_history.clear()
        self._trade_count_history.clear()
        self._last_reward = 0.0
        self._last_reason = ""
        self._call_count = 0
        self._reward_components_history.clear()
        self._performance_analytics.clear()
        self._regime_performance.clear()
        self.audit_trail.clear()
        self._sharpe_ratio = 0.0
        self._consistency_score = 0.0
        self._win_rate = 0.0
        self._avg_reward = 0.0
        self._reward_volatility = 0.0
        self._reward_quality = 0.5
        self._adaptive_params.update(
            {
                "dynamic_penalty_scaling": 1.0,
                "regime_sensitivity": 1.0,
                "activity_threshold": 1.0,
                "risk_tolerance": 1.0,
                "learning_momentum": 0.0,
                "adaptation_confidence": 0.5,
            }
        )
        self.circuit_breaker["failures"] = 0
        self.circuit_breaker["state"] = "CLOSED"
        self._health_status = "healthy"
        self._baseline_balance = None
        self._last_balance_observed = None

    def get_observation_components(self) -> np.ndarray:
        try:
            if not self._reward_history:
                return np.zeros(10, np.float32)
            rewards = np.array(self._reward_history, np.float32)
            recent_mean = rewards[-10:].mean() if len(rewards) >= 10 else rewards.mean()
            recent_std = rewards[-10:].std() if len(rewards) >= 10 else 0.1
            win_rate = self._win_rate
            activity = float(np.mean(self._trade_count_history)) if self._trade_count_history else 0.0
            trend = 0.0
            if len(rewards) >= 5:
                trend = float(np.polyfit(range(5), rewards[-5:], 1)[0])
            consistency = self._consistency_score
            sharpe_norm = float(np.tanh(self._sharpe_ratio / 3.0))
            reward_quality = self._reward_quality
            adapt_conf = self._adaptive_params.get("adaptation_confidence", 0.5)
            return np.array(
                [
                    float(self._last_reward),
                    float(recent_mean),
                    float(recent_std),
                    float(win_rate),
                    float(activity),
                    float(trend),
                    float(consistency),
                    float(sharpe_norm),
                    float(reward_quality),
                    float(adapt_conf),
                ],
                np.float32,
            )
        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return np.zeros(10, dtype=np.float32)

    def get_last_audit(self) -> Dict[str, Any]:
        return self.audit_trail[-1] if self.audit_trail else {}

    def get_audit_trail(self, n: int = 20) -> List[Dict[str, Any]]:
        return self.audit_trail[-n:]

    def get_weights(self) -> Dict[str, Any]:
        return {
            "regime_weights": self.regime_weights.copy(),
            "dd_pen_weight": self.cfg.dd_pen_weight,
            "risk_pen_weight": self.cfg.risk_pen_weight,
            "tail_pen_weight": self.cfg.tail_pen_weight,
            "mistake_pen_weight": self.cfg.mistake_pen_weight,
            "no_trade_penalty_weight": self.cfg.no_trade_penalty_weight,
            "win_bonus_weight": self.cfg.win_bonus_weight,
            "consistency_bonus_weight": self.cfg.consistency_bonus_weight,
            "sharpe_bonus_weight": self.cfg.sharpe_bonus_weight,
            "trade_frequency_bonus": self.cfg.trade_frequency_bonus,
            "volatility_adjustment": self.cfg.volatility_adjustment,
            "regime_bonus_weight": self.cfg.regime_bonus_weight,
            "momentum_bonus_weight": self.cfg.momentum_bonus_weight,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "current_mode": self.current_mode.value,
            "reward_history": list(self._reward_history),
            "pnl_history": list(self._pnl_history),
            "trade_count_history": list(self._trade_count_history),
            "last_reward": self._last_reward,
            "last_reason": self._last_reason,
            "call_count": self._call_count,
            "genome": self.genome.copy(),
            "adaptive_params": self._adaptive_params.copy(),
            "performance_metrics": {
                "sharpe_ratio": self._sharpe_ratio,
                "consistency_score": self._consistency_score,
                "win_rate": self._win_rate,
                "avg_reward": self._avg_reward,
                "reward_volatility": self._reward_volatility,
                "reward_quality": self._reward_quality,
            },
            "circuit_breaker": self.circuit_breaker.copy(),
            "health_status": self._health_status,
            "regime_performance": {
                regime: {"rewards": data["rewards"][-20:], "pnl": data["pnl"][-20:]} for regime, data in self._regime_performance.items()
            },
            "baseline_balance": self._baseline_balance,
            "last_balance_observed": self._last_balance_observed,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        if "current_mode" in state:
            try:
                self.current_mode = RewardMode(state["current_mode"])
            except ValueError:
                self.current_mode = RewardMode.TRAINING

        self._reward_history = deque(state.get("reward_history", []), maxlen=self.cfg.history_size)
        self._pnl_history = deque(state.get("pnl_history", []), maxlen=self.cfg.history_size)
        self._trade_count_history = deque(state.get("trade_count_history", []), maxlen=20)
        self._last_reward = float(state.get("last_reward", 0.0))
        self._last_reason = state.get("last_reason", "")
        self._call_count = int(state.get("call_count", 0))

        if "genome" in state:
            self.set_genome(state["genome"])
        if "adaptive_params" in state:
            self._adaptive_params.update(state["adaptive_params"])

        perf = state.get("performance_metrics", {})
        self._sharpe_ratio = float(perf.get("sharpe_ratio", 0.0))
        self._consistency_score = float(perf.get("consistency_score", 0.0))
        self._win_rate = float(perf.get("win_rate", 0.0))
        self._avg_reward = float(perf.get("avg_reward", 0.0))
        self._reward_volatility = float(perf.get("reward_volatility", 0.0))
        self._reward_quality = float(perf.get("reward_quality", 0.5))

        if "circuit_breaker" in state:
            self.circuit_breaker.update(state["circuit_breaker"])
        if "health_status" in state:
            self._health_status = state["health_status"]

        self._baseline_balance = state.get("baseline_balance", self._baseline_balance)
        self._last_balance_observed = state.get("last_balance_observed", self._last_balance_observed)

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "status": self._health_status,
            "last_check": self._last_health_check,
            "circuit_breaker": self.circuit_breaker["state"],
            "current_mode": self.current_mode.value,
            "reward_quality": self._reward_quality,
            "avg_reward": self._avg_reward,
            "win_rate": self._win_rate,
            "sharpe_ratio": self._sharpe_ratio,
            "adaptation_confidence": self._adaptive_params.get("adaptation_confidence", 0.5),
            "baseline_balance": self._baseline_balance,
            "last_balance_observed": self._last_balance_observed,
        }

    def stop_monitoring(self) -> None:
        self._monitoring_active = False

    def get_reward_system_report(self) -> str:
        if self._avg_reward > 0.5:
            performance_status = "[ROCKET] Excellent"
        elif self._avg_reward > 0.0:
            performance_status = "[OK] Good"
        elif self._avg_reward > -0.5:
            performance_status = "[FAST] Fair"
        else:
            performance_status = "[WARN] Poor"

        if self._reward_quality > 0.8:
            quality_status = "[TARGET] High"
        elif self._reward_quality > 0.6:
            quality_status = "[OK] Good"
        elif self._reward_quality > 0.4:
            quality_status = "[FAST] Fair"
        else:
            quality_status = "[FAIL] Low"

        cb_status = "[RED] OPEN" if self.circuit_breaker["state"] == "OPEN" else "[GREEN] CLOSED"
        health_emoji = "[OK]" if self._health_status == "healthy" else "[WARN]"

        return f"""
[TARGET] ENHANCED RISK-ADJUSTED REWARD SYSTEM v4.1
═══════════════════════════════════════════════════
[STATS] Performance: {performance_status} ({self._avg_reward:.4f} avg)
[TARGET] Quality: {quality_status} ({self._reward_quality:.3f})
💎 Consistency: {self._consistency_score:.3f}
[CHART] Sharpe Ratio: {self._sharpe_ratio:.3f}
[MONEY] Win Rate: {self._win_rate:.1%}

[HEALTH] SYSTEM HEALTH
• Status: {health_emoji} {self._health_status.upper()}
• Circuit Breaker: {cb_status}
• Mode: {self.current_mode.value.upper()}
• Adaptation Confidence: {self._adaptive_params.get('adaptation_confidence', 0.5):.2f}

[ACCOUNT]
• Last Balance Observed: {self._last_balance_observed if self._last_balance_observed is not None else 'n/a'}
• Baseline Balance: {self._baseline_balance if self._baseline_balance is not None else 'n/a'}

[BALANCE] CONFIGURATION
• Regime Weights: [{', '.join(f'{w:.2f}' for w in self.regime_weights)}]
• Drawdown Penalty: {self.cfg.dd_pen_weight:.2f}
• Win Bonus: {self.cfg.win_bonus_weight:.2f}
• Consistency Bonus: {self.cfg.consistency_bonus_weight:.2f}

[TOOL] ADAPTIVE PARAMETERS
• Penalty Scaling: {self._adaptive_params['dynamic_penalty_scaling']:.2f}
• Regime Sensitivity: {self._adaptive_params['regime_sensitivity']:.2f}
• Activity Threshold: {self._adaptive_params['activity_threshold']:.2f}
• Risk Tolerance: {self._adaptive_params['risk_tolerance']:.2f}

[STATS] ACTIVITY
• Total Calls: {self._call_count:,}
• Reward History: {len(self._reward_history)} records
• Audit Trail: {len(self.audit_trail)} entries
• Last Reward: {self._last_reward:.4f} ({self._last_reason})
"""
