# ─────────────────────────────────────────────────────────────
# File: modules/risk/portfolio_risk_system.py
# [ROCKET] PRODUCTION-READY Enhanced Portfolio Risk System
# Advanced portfolio risk management with SmartInfoBus integration and intelligent automation
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
from modules.contracts import module_args
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


class RiskMode(Enum):
    """Portfolio risk operational modes"""
    INITIALIZATION = "initialization"
    BOOTSTRAP = "bootstrap"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PortfolioRiskConfig:
    """Configuration for Portfolio Risk System"""
    var_window: int = 20
    dd_limit: float = 0.20
    risk_mult: float = 2.0
    min_position_pct: float = 0.01
    max_position_pct: float = 0.25
    correlation_window: int = 50
    bootstrap_trades: int = 10
    var_confidence: float = 0.95
    max_portfolio_exposure: float = 1.0
    correlation_threshold: float = 0.8
    volatility_lookback: int = 30
    risk_budget_daily: float = 0.02

    # Performance thresholds
    max_processing_time_ms: float = 200
    circuit_breaker_threshold: int = 5
    min_risk_quality: float = 0.3

    # Adaptation parameters
    adaptive_learning_rate: float = 0.01
    risk_sensitivity: float = 1.0


@module(**module_args(
    "PortfolioRiskSystem",
    description="Advanced portfolio risk management with comprehensive VaR analysis and dynamic position limits",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
    # --- Voting additions ---
    is_voting_member=True,
))
class PortfolioRiskSystem(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    [ROCKET] Advanced portfolio risk system with SmartInfoBus integration.
    Contract guarantees:
      • Reads required keys from SmartInfoBus: market_context, market_data, position_data
      • Writes ONLY its provides: portfolio_risk, portfolio_risk_proposal, position_limits,
        risk_data, risk_metrics, risk_score, risk_signals, trade_data, trading_data
      • Returns ALL provides + '_thesis' + 'success' for success/fallback/error.

    NOTE: This module must NOT read bus keys that it provides itself
          (e.g., 'trade_data', 'risk_signals') to avoid first-run BUS MISS.
    """

    def __init__(
        self,
        config: Optional[Union[PortfolioRiskConfig, Dict[str, Any]]] = None,
        instruments: Optional[List[str]] = None,
        **kwargs
    ):
        # Keep a strongly-typed config separate from BaseModule.config (which is typically Dict[str, Any])
        if config is None:
            self._cfg = PortfolioRiskConfig()
        elif isinstance(config, dict):
            self._cfg = PortfolioRiskConfig(**config)
        else:
            self._cfg = config

        self.instruments = instruments or ["EUR_USD", "XAU_USD"]

        # Minimal pre-initialization so BaseModule.__init__ can safely call self._initialize()
        # without attribute errors.
        self._preinitialize_minimum()

        # Call Base init (may call self._initialize()). We do NOT overwrite BaseModule.config
        # with our dataclass; BaseModule may set up its own config.
        super().__init__()

        # Full initialization after base is ready
        self._initialize_advanced_systems()
        self._initialize_portfolio_state()

        self.logger.info(
            format_operator_message(
                message="PORTFOLIO_RISK_INITIALIZED",
                icon="💼",
                details=f"Instruments: {len(self.instruments)}, VaR window: {self._cfg.var_window}",
                result="Enhanced portfolio risk system ready",
                context="portfolio_initialization",
            )
        )

    def _preinitialize_minimum(self) -> None:
        """Set minimal state so _initialize can run safely if invoked early by BaseModule."""
        try:
            # Core services
            self.smart_bus = InfoBusManager.get_instance()
            self.logger = RotatingLogger(
                name="PortfolioRiskSystem",
                log_path="logs/risk/portfolio_risk_system.log",
                max_lines=5000,
                operator_mode=True,
                plain_english=True,
            )
        except Exception:
            # As a last resort, ensure attributes exist to avoid attribute errors
            if not hasattr(self, "smart_bus"):
                self.smart_bus = InfoBusManager.get_instance()
            if not hasattr(self, "logger"):
                class _Dummy:
                    def info(self, *a, **k):
                        pass
                    def warning(self, *a, **k):
                        pass
                    def error(self, *a, **k):
                        pass
                self.logger = _Dummy()

        # Minimal risk state used by _initialize bus writes
        self.current_mode = getattr(self, "current_mode", RiskMode.INITIALIZATION)
        self.bootstrap_mode = getattr(self, "bootstrap_mode", True)
        self.current_var = getattr(self, "current_var", 0.0)
        self.max_correlation = getattr(self, "max_correlation", 0.0)
        self.risk_adjustment = getattr(self, "risk_adjustment", 1.0)

    def _initialize_advanced_systems(self):
        """Initialize advanced systems for portfolio risk"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="PortfolioRiskSystem",
            log_path="logs/risk/portfolio_risk_system.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PortfolioRiskSystem", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Circuit breaker for risk operations
        self.circuit_breaker = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",
            "threshold": int(self._cfg.circuit_breaker_threshold),
        }

        # Health monitoring
        self._health_status = "healthy"
        self._last_health_check = time.time()
        # Note: Start monitoring after state init

    def _initialize_portfolio_state(self):
        """Initialize portfolio risk state"""
        # Initialize mixin states
        self._initialize_risk_state()
        self._initialize_trading_state()
        self._initialize_state_management()

        # Current operational mode
        self.current_mode = RiskMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()

        # Enhanced state tracking
        self.returns_history: Dict[str, deque] = {
            inst: deque(maxlen=max(self._cfg.var_window, self._cfg.correlation_window))
            for inst in self.instruments
        }
        self._last_prices: Dict[str, float] = {}
        self.portfolio_returns = deque(maxlen=self._cfg.var_window)
        self.current_positions: Dict[str, float] = {}
        self.position_history = deque(maxlen=100)
        self.trade_count = 0

        # Portfolio performance metrics
        self.performance_metrics = {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.5,
            "recent_pnl": 0.0,
            "total_pnl": 0.0,
            "volatility": 0.0,
            "var_95": 0.0,
            "total_exposure": 0.0,
            "risk_quality": 0.5,
        }

        # Risk factors
        self.risk_adjustment = 1.0
        self.min_risk_adjustment = 0.5
        self.max_risk_adjustment = 1.5

        # Position limits tracking
        self.position_limits: Dict[str, float] = {inst: float(self._cfg.max_position_pct) for inst in self.instruments}

        # VaR and correlation tracking
        self.current_var = 0.0
        self.correlation_matrix: Optional[np.ndarray] = None
        the_max = 0.0  # to avoid potential uninitialized warnings in static analyzers
        self.max_correlation = the_max

        # Market context awareness
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"

        # Bootstrap mode tracking
        self.bootstrap_mode = True

        # Risk budget tracking
        self.daily_risk_used = 0.0
        self.risk_budget_violations = 0

        # Performance analytics
        self.portfolio_analytics = defaultdict(list)
        self.regime_performance = defaultdict(lambda: defaultdict(list))

        # Risk events tracking
        self.risk_events: List[Dict[str, Any]] = []
        self.limit_violations = 0
        self.correlation_alerts = 0

        # Adaptive parameters
        self._adaptive_params = {
            "dynamic_limit_scaling": 1.0,
            "correlation_sensitivity": 1.0,
            "volatility_tolerance": 1.0,
            "risk_adaptation_confidence": 0.5,
        }

        # Start monitoring after all state is initialized
        self._start_monitoring()

    def _start_monitoring(self):
        """Start background monitoring for portfolio risk (idempotent)."""
        if getattr(self, "_monitoring_active", False):
            # already running
            return

        def monitoring_loop():
            import time as _time
            self.logger.info("[LOG] Portfolio risk monitoring loop started")
            while getattr(self, "_monitoring_active", True):
                try:
                    self._update_portfolio_health()
                    self._analyze_risk_effectiveness()
                    self._adapt_risk_parameters()
                    _time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Portfolio risk monitoring error: {e}")

        self._monitoring_active = True
        import threading as _threading
        t = _threading.Thread(target=monitoring_loop, daemon=True)
        t.start()
        self._monitoring_thread = t


    def _initialize(self) -> None:
        """Initialize module with SmartInfoBus integration"""
        try:
            # Defensive: ensure required components/attributes exist even if _initialize is called
            # before our __init__ finished (ordering differences across bases/environments).
            if not hasattr(self, "smart_bus"):
                self.smart_bus = InfoBusManager.get_instance()
            if not hasattr(self, "logger"):
                self.logger = RotatingLogger(
                    name="PortfolioRiskSystem",
                    log_path="logs/risk/portfolio_risk_system.log",
                    max_lines=5000,
                    operator_mode=True,
                    plain_english=True,
                )

            mode = getattr(self, "current_mode", RiskMode.INITIALIZATION)
            bootstrap_mode = getattr(self, "bootstrap_mode", True)
            current_var = float(getattr(self, "current_var", 0.0))
            max_correlation = float(getattr(self, "max_correlation", 0.0))
            risk_adjustment = float(getattr(self, "risk_adjustment", 1.0))

            # Initial portfolio risk status
            initial_status = {
                "current_mode": mode.value if isinstance(mode, RiskMode) else str(mode),
                "bootstrap_mode": bool(bootstrap_mode),
                "var_95": current_var,
                "max_correlation": max_correlation,
                "risk_adjustment": risk_adjustment,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            self.smart_bus.set(
                "portfolio_risk",
                initial_status,
                module="PortfolioRiskSystem",
                thesis="Initial portfolio risk system status",
            )

            # PRE-PROVIDE EMPTY BASELINES to avoid early consumer bus misses
            self.smart_bus.set(
                "risk_signals",
                {"violations": [], "mode": mode.value if isinstance(mode, RiskMode) else str(mode), "budget_violation": False},
                module="PortfolioRiskSystem",
                thesis="Initial risk signals (baseline)",
            )
            # Use namespaced key to avoid ownership conflict with Executor's 'trade_data'
            self.smart_bus.set(
                "portfolio_trade_data",
                {"recent_trades": [], "positions": []},
                module="PortfolioRiskSystem",
                thesis="Initial portfolio trade data (baseline)",
            )
            self.smart_bus.set(
                "trading_data",
                {"timestamp": datetime.datetime.now().isoformat(), "prices_available": False, "positions_count": 0},
                module="PortfolioRiskSystem",
                thesis="Initial trading data summary (baseline)",
            )

            # Provide baseline risk_data early for consumers like TimeAwareRiskScaling
            self.smart_bus.set(
                "risk_data",
                {
                    "current_mode": mode.value if isinstance(mode, RiskMode) else str(mode),
                    "metrics": {
                        "var_95": current_var,
                        "max_correlation": max_correlation,
                        "portfolio_volatility": 0.0,
                        "risk_quality": 0.5,
                        "total_exposure": 0.0,
                        "daily_risk_used": 0.0,
                    },
                    "limits": {
                        "position_limits": {},
                        "base_limit": self._cfg.max_position_pct if hasattr(self, "_cfg") else 0.25,
                        "risk_adjustment": risk_adjustment,
                        "bootstrap_mode": bool(bootstrap_mode),
                    },
                    "alerts": {"limit_violations": 0, "correlation_alerts": 0},
                },
                module="PortfolioRiskSystem",
                thesis="Initial risk_data baseline",
            )
        except Exception as e:
            self.logger.error(f"Portfolio risk initialization failed: {e}")

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process portfolio risk assessment with enhanced analytics"""
        start_time = time.time()

        try:
            # Extract portfolio data from SmartInfoBus (required keys only; NO self-provide reads)
            portfolio_data = await self._extract_portfolio_data(**inputs)

            if not portfolio_data:
                payload = await self._handle_no_data_fallback()
                # --- Voting additions (fallback voting) ---
                vote_payload = await self.vote()
                payload["PortfolioRiskSystem_voting_proposal"] = vote_payload
                payload["PortfolioRiskSystem_confidence"] = float(vote_payload.get("confidence", 0.0))
                # Ensure success flag even in fallback (no error)
                payload["success"] = True
                # Optional: write coordinator payload
                await self._write_voting_to_bus(vote_payload)
                return payload

            # Update market context
            context_result = await self._update_market_context_async(portfolio_data)

            # Update positions and returns
            position_result = await self._update_positions_and_returns(portfolio_data)

            # Calculate comprehensive risk metrics
            risk_result = await self._calculate_comprehensive_risk_metrics(portfolio_data)

            # Update position limits dynamically
            limits_result = await self._update_dynamic_position_limits(portfolio_data)

            # Check risk violations
            violations_result = await self._check_portfolio_risk_violations(portfolio_data)

            # Update mode based on risk level
            mode_result = await self._update_operational_mode(portfolio_data)

            # Combine results
            result = {
                **context_result,
                **position_result,
                **risk_result,
                **limits_result,
                **violations_result,
                **mode_result,
            }

            # Generate thesis
            thesis = await self._generate_portfolio_thesis(portfolio_data, result)

            # Build provides-compliant payload with thesis
            provides_payload = {
                **result,
                "portfolio_risk": {
                    "current_mode": self.current_mode.value,
                    "var_95": self.current_var,
                    "max_correlation": self.max_correlation,
                    "risk_adjustment": self.risk_adjustment,
                    "bootstrap_mode": self.bootstrap_mode,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                # Minimal risk_metrics bundle for contract compliance
                "risk_metrics": {
                    "var_95": self.current_var,
                    "correlation_matrix": self.correlation_matrix.tolist() if self.correlation_matrix is not None else None,
                    "max_correlation": self.max_correlation,
                    "portfolio_volatility": self.performance_metrics.get("volatility", 0.0),
                    "risk_quality": self.performance_metrics.get("risk_quality", 0.5),
                    "total_exposure": self.performance_metrics.get("total_exposure", 0.0),
                },
                # Position limits bundle for contract compliance
                "position_limits": {
                    "position_limits": self.position_limits.copy(),
                    "risk_adjustment": self.risk_adjustment,
                    "base_limit": self._cfg.max_position_pct,
                    "bootstrap_mode": self.bootstrap_mode,
                },
                # Consolidated risk data bundle
                "risk_data": {
                    "current_mode": self.current_mode.value,
                    "metrics": {
                        "var_95": self.current_var,
                        "max_correlation": self.max_correlation,
                        "portfolio_volatility": self.performance_metrics.get("volatility", 0.0),
                        "risk_quality": self.performance_metrics.get("risk_quality", 0.5),
                        "total_exposure": self.performance_metrics.get("total_exposure", 0.0),
                        "daily_risk_used": self.daily_risk_used,
                    },
                    "limits": {
                        "position_limits": self.position_limits.copy(),
                        "base_limit": self._cfg.max_position_pct,
                        "risk_adjustment": self.risk_adjustment,
                        "bootstrap_mode": self.bootstrap_mode,
                    },
                    "alerts": {"limit_violations": self.limit_violations, "correlation_alerts": self.correlation_alerts},
                },
                # Derived risk signals (warnings/info)
                "risk_signals": {
                    "violations": result.get("violations", []),
                    "mode": self.current_mode.value,
                    "budget_violation": result.get("budget_violation", False),
                },
                # Overall risk score (0-1)
                "risk_score": float(self.performance_metrics.get("risk_quality", 0.5)),
                # Trade and trading data passthrough (minimal, safe defaults)
                # Publish under portfolio_trade_data (Executor owns canonical trade_data)
                "portfolio_trade_data": {
                    "recent_trades": portfolio_data.get("trades", []),
                    "positions": portfolio_data.get("positions", []),
                },
                "trading_data": {
                    "timestamp": portfolio_data.get("timestamp"),
                    "prices_available": bool(portfolio_data.get("prices")),
                    "positions_count": len(portfolio_data.get("positions", [])),
                },
                "_thesis": thesis,
            }

            # Generate a proposal (required provides)
            try:
                proposal = await self.propose_action()
            except Exception:
                proposal = {
                    "timestamp": time.time(),
                    "confidence": 0.0,
                    "actions": [],
                    "status": self.get_current_risk_status(),
                    "error": "proposal_generation_failed",
                }
            provides_payload["portfolio_risk_proposal"] = proposal

            # --- Voting additions: build + attach proposal & confidence ---
            vote_payload = await self.vote()
            provides_payload["PortfolioRiskSystem_voting_proposal"] = vote_payload
            provides_payload["PortfolioRiskSystem_confidence"] = float(vote_payload.get("confidence", 0.0))
            # Optional: coordinator bus write (separate from our contract-provides keys)
            await self._write_voting_to_bus(vote_payload)

            # Update SmartInfoBus (writes only provides)
            await self._update_portfolio_smart_bus(provides_payload, thesis)

            # Record success
            processing_time = (time.time() - start_time) * 1000.0
            self._record_success(processing_time)

            # Contract success flag
            provides_payload["success"] = True
            return provides_payload

        except Exception as e:
            error_payload = await self._handle_portfolio_error(e, start_time)
            # --- Voting additions (error voting) ---
            try:
                vote_payload = await self.vote()
                error_payload["PortfolioRiskSystem_voting_proposal"] = vote_payload
                error_payload["PortfolioRiskSystem_confidence"] = float(vote_payload.get("confidence", 0.0))
                await self._write_voting_to_bus(vote_payload)
            except Exception:
                pass
            error_payload["success"] = False
            return error_payload

    async def _extract_portfolio_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract comprehensive portfolio data from SmartInfoBus (requires-only).
        IMPORTANT: Do NOT read self-provided keys here (e.g., 'trade_data', 'risk_signals').
        """
        try:
            # Required keys only (per contract)
            position_data = self.smart_bus.get("position_data", "PortfolioRiskSystem") or {}
            positions = position_data.get("positions", [])
            # Accept both dict-of-dict and list-of-dict schemas
            if isinstance(positions, dict):
                try:
                    positions = [{"instrument": inst, **(p or {})} for inst, p in positions.items()]
                except Exception:
                    # As a last resort, flatten to empty list to avoid type errors
                    positions = []

            market_data = self.smart_bus.get("market_data", "PortfolioRiskSystem") or {}
            prices = market_data.get("prices", {})

            # Do NOT read trade_data/risk_signals from the bus (we provide them)
            # Trades are optional inputs; fall back to empty list
            trades = inputs.get("trades", [])
            balance = inputs.get("balance", 0)
            portfolio_inputs = inputs.get("portfolio_data", {})

            return {
                "balance": balance,
                "trades": trades,
                "positions": positions,
                "prices": prices,
                "market_data": market_data,
                "portfolio_inputs": portfolio_inputs,
                "timestamp": datetime.datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Failed to extract portfolio data: {e}")
            return None

    async def _update_market_context_async(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update market context awareness asynchronously"""
        try:
            # Required: market_context
            market_context = self.smart_bus.get("market_context", "PortfolioRiskSystem") or {}

            # Update regime tracking (don't downgrade to 'unknown' if missing)
            old_regime = self.market_regime
            new_regime = market_context.get("regime")
            if new_regime and isinstance(new_regime, str):
                self.market_regime = new_regime
            self.volatility_regime = market_context.get("volatility_level", "medium")
            self.market_session = market_context.get("session", "unknown")

            if self.market_regime != old_regime:
                self.logger.info(
                    format_operator_message(
                        message="MARKET_REGIME_CHANGE",
                        icon="[STATS]",
                        old_regime=old_regime,
                        new_regime=self.market_regime,
                        volatility=self.volatility_regime,
                        session=self.market_session,
                        context="market_context",
                    )
                )
                self.regime_performance[self.market_regime]["regime_changes"].append(
                    {
                        "timestamp": portfolio_data.get("timestamp", datetime.datetime.now().isoformat()),
                        "from_regime": old_regime,
                        "to_regime": self.market_regime,
                    }
                )

            return {
                "market_context_updated": True,
                "current_regime": self.market_regime,
                "volatility_regime": self.volatility_regime,
                "market_session": self.market_session,
            }

        except Exception as e:
            self.logger.error(f"Market context update failed: {e}")
            return {"market_context_updated": False, "error": str(e)}
        
    def _infer_notional_eur(self, pos: dict, prices: dict, instrument: str) -> float:
        """
        Infer position notional in account currency (EUR).
        Priority:
        1) pos["notional_eur"]
        2) Executor alias pos["size"] (which is already notional per Executor)
        3) units * (entry_price or current_price)
        4) 0.0
        """
        # 1) Explicit notional
        n = pos.get("notional_eur")
        if n is not None:
            try:
                return abs(float(n))
            except Exception:
                pass

        # 2) Executor alias: 'size' is already notional (per Executor._publish_all comment)
        size = pos.get("size")
        if size is not None:
            try:
                return abs(float(size))
            except Exception:
                pass

        # 3) Reconstruct from units × price
        try:
            units = float(pos.get("units", 0.0) or 0.0)
            # prefer entry price; fallback to current price from pos or prices map
            entry_px = float(pos.get("entry_price", 0.0) or 0.0)
            px = entry_px
            if px <= 0.0:
                px = float(
                    pos.get(
                        "current_price",
                        prices.get(instrument, prices.get(instrument.replace("/", "").replace("_", ""), 0.0)),
                    )
                    or 0.0
                )
            if units > 0.0 and px > 0.0:
                return abs(units * px)
        except Exception:
            pass

        # 4) Nothing usable
        return 0.0


    async def _update_positions_and_returns(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update position tracking and returns history (correct exposure from notional/balance)."""
        try:
            positions = portfolio_data.get("positions", []) or []
            self.current_positions.clear()

            prices = portfolio_data.get("prices", {}) or {}

            # Resolve balance/equity to normalize exposures
            balance = float(portfolio_data.get("balance", 0.0) or 0.0)
            if balance <= 0.0:
                try:
                    pm = self.smart_bus.get("portfolio_metrics", "PortfolioRiskSystem") or {}
                    # prefer equity, fallback to balance
                    balance = float(pm.get("equity", pm.get("balance", 0.0)) or 0.0)
                except Exception:
                    balance = 0.0

            # Throttled warning if we still can't normalize exposures
            if balance <= 0.0:
                import time as _time
                now = _time.time()
                if not hasattr(self, "_last_exposure_warn_ts") or (now - getattr(self, "_last_exposure_warn_ts", 0.0) > 60.0):
                    self._last_exposure_warn_ts = now
                    self.logger.warning(
                        "[WARN] Exposure normalization skipped (balance/equity <= 0). "
                        "Check upstream portfolio_metrics/equity propagation."
                    )

            for pos in positions:
                instrument = pos.get("symbol", pos.get("instrument", "UNKNOWN"))

                # Compute notional safely (EUR)
                notional = self._infer_notional_eur(pos, prices, instrument)

                # Normalize to exposure (% of equity)
                if balance > 0.0:
                    exposure = notional / balance
                else:
                    exposure = 0.0  # avoid divide-by-zero blowups

                self.current_positions[instrument] = float(exposure)

            # Track trades and bootstrap transition
            trades = portfolio_data.get("trades", []) or []
            if trades:
                self.trade_count += len(trades)
                if self.bootstrap_mode and self.trade_count >= self._cfg.bootstrap_trades:
                    self.bootstrap_mode = False
                    self.logger.info(
                        format_operator_message(
                            message="BOOTSTRAP_COMPLETE",
                            icon="[CHART]",
                            trade_count=self.trade_count,
                            threshold=self._cfg.bootstrap_trades,
                            context="bootstrap",
                        )
                    )

            # Update returns history (uses prices + last_prices)
            await self._update_returns_history_async(portfolio_data)

            # Keep a short history of exposure snapshots
            if positions:
                import datetime as _dt
                self.position_history.append(
                    {
                        "timestamp": portfolio_data.get("timestamp", _dt.datetime.now().isoformat()),
                        "positions": dict(self.current_positions),
                        "trade_count": self.trade_count,
                    }
                )

            return {
                "positions_updated": True,
                "position_count": len(self.current_positions),
                "trade_count": self.trade_count,
                "bootstrap_mode": self.bootstrap_mode,
            }

        except Exception as e:
            self.logger.error(f"Position update failed: {e}")
            return {"positions_updated": False, "error": str(e)}


    async def _update_returns_history_async(self, portfolio_data: Dict[str, Any]):
        """Update returns history from market data"""
        try:
            prices = portfolio_data.get("prices", {})

            for instrument in self.instruments:
                if instrument in prices:
                    current_price = float(prices[instrument])
                    last_price = self._last_prices.get(instrument)
                    if last_price is not None and last_price > 0:
                        ret = (current_price - last_price) / last_price
                        self.returns_history[instrument].append(float(ret))
                    self._last_prices[instrument] = current_price

            if self.current_positions:
                portfolio_return = 0.0
                total_weight = 0.0
                for instrument, position in self.current_positions.items():
                    if instrument in self.returns_history and len(self.returns_history[instrument]) > 0:
                        inst_return = float(self.returns_history[instrument][-1])
                        # Weight by normalized exposure magnitude
                        weight = abs(position)
                        portfolio_return += inst_return * weight
                        total_weight += weight

                if total_weight > 0:
                    portfolio_return /= total_weight
                    self.portfolio_returns.append(float(portfolio_return))

        except Exception as e:
            self.logger.error(f"Returns history update failed: {e}")

    async def _calculate_comprehensive_risk_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            var_result = await self._calculate_portfolio_var_async()
            correlation_result = await self._calculate_correlation_matrix_async()
            volatility_result = await self._calculate_portfolio_volatility_async()
            performance_result = await self._update_portfolio_performance_async(portfolio_data)
            adjustment_result = await self._update_risk_adjustment_factor_async(portfolio_data)
            budget_result = await self._update_risk_budget_usage_async(portfolio_data)

            return {
                "risk_metrics_calculated": True,
                "var_95": self.current_var,
                "max_correlation": self.max_correlation,
                "portfolio_volatility": self.performance_metrics["volatility"],
                "risk_adjustment": self.risk_adjustment,
                "risk_budget_used": self.daily_risk_used,
                **var_result,
                **correlation_result,
                **volatility_result,
                **performance_result,
                **adjustment_result,
                **budget_result,
            }

        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {"risk_metrics_calculated": False, "error": str(e)}

    async def _calculate_portfolio_var_async(self) -> Dict[str, Any]:
        """Calculate portfolio Value at Risk asynchronously"""
        try:
            if len(self.portfolio_returns) < 10:
                self.current_var = 0.0
                return {"var_data_sufficient": False}

            returns = np.array(list(self.portfolio_returns))
            var_percentile = (1 - self._cfg.var_confidence) * 100.0
            self.current_var = float(abs(np.percentile(returns, var_percentile)))
            self.performance_metrics["var_95"] = float(self.current_var)

            return {
                "var_calculated": True,
                "var_data_points": len(self.portfolio_returns),
                "var_confidence": self._cfg.var_confidence,
            }

        except Exception as e:
            self.logger.warning(f"VaR calculation failed: {e}")
            self.current_var = 0.0
            return {"var_calculated": False, "error": str(e)}

    async def _calculate_correlation_matrix_async(self) -> Dict[str, Any]:
        """Calculate correlation matrix for instruments asynchronously"""
        try:
            n_inst = len(self.instruments)
            self.correlation_matrix = np.eye(n_inst)

            if any(len(self.returns_history[inst]) > 0 for inst in self.instruments):
                min_len = min(len(self.returns_history[inst]) for inst in self.instruments if len(self.returns_history[inst]) > 0)
            else:
                min_len = 0

            if min_len < 10:
                self.max_correlation = 0.0
                return {"correlation_data_sufficient": False}

            returns_matrix = []
            valid_instruments = []
            for inst in self.instruments:
                if len(self.returns_history[inst]) >= min_len:
                    returns_matrix.append(list(self.returns_history[inst])[-min_len:])
                    valid_instruments.append(inst)

            if len(returns_matrix) < 2:
                self.max_correlation = 0.0
                return {"correlation_pairs_insufficient": True}

            returns_matrix = np.array(returns_matrix)

            for i in range(len(valid_instruments)):
                for j in range(i + 1, len(valid_instruments)):
                    try:
                        corr = float(np.corrcoef(returns_matrix[i], returns_matrix[j])[0, 1])
                        if np.isfinite(corr):
                            if i < n_inst and j < n_inst:
                                self.correlation_matrix[i, j] = corr
                                self.correlation_matrix[j, i] = corr
                    except Exception:
                        continue

            off_diagonal = self.correlation_matrix[np.triu_indices(n_inst, k=1)]
            self.max_correlation = float(np.max(np.abs(off_diagonal))) if len(off_diagonal) > 0 else 0.0

            return {
                "correlation_calculated": True,
                "correlation_data_points": min_len,
                "valid_instruments": len(valid_instruments),
            }

        except Exception as e:
            self.logger.warning(f"Correlation calculation failed: {e}")
            self.max_correlation = 0.0
            return {"correlation_calculated": False, "error": str(e)}

    async def _calculate_portfolio_volatility_async(self) -> Dict[str, Any]:
        """Calculate portfolio volatility asynchronously"""
        try:
            if len(self.portfolio_returns) < 5:
                self.performance_metrics["volatility"] = 0.0
                return {"volatility_data_sufficient": False}

            returns = np.array(list(self.portfolio_returns)[-self._cfg.volatility_lookback :])
            volatility = float(np.std(returns) * np.sqrt(252))
            self.performance_metrics["volatility"] = volatility

            return {
                "volatility_calculated": True,
                "volatility_data_points": len(returns),
                "annualized_volatility": volatility,
            }

        except Exception as e:
            self.logger.warning(f"Volatility calculation failed: {e}")
            self.performance_metrics["volatility"] = 0.0
            return {"volatility_calculated": False, "error": str(e)}

    async def _update_portfolio_performance_async(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update portfolio performance metrics asynchronously"""
        try:
            recent_pnl = float(sum(float(trade.get("pnl", 0)) for trade in portfolio_data.get("trades", [])))
            balance = float(portfolio_data.get("balance", 0))

            self.performance_metrics["recent_pnl"] = recent_pnl
            self.performance_metrics["total_pnl"] += recent_pnl

            total_exposure = 0.0
            for pos in portfolio_data.get("positions", []):
                size = abs(float(pos.get("size", 0)))
                price = float(pos.get("current_price", pos.get("entry_price", 1.0)))
                total_exposure += size * price

            self.performance_metrics["total_exposure"] = (total_exposure / balance) if balance > 0 else 0.0

            if len(self.portfolio_returns) >= 20:
                returns = np.array(list(self.portfolio_returns)[-20:])
                if returns.std() > 0:
                    sharpe = float(np.sqrt(252.0) * returns.mean() / returns.std())
                    self.performance_metrics["sharpe"] = sharpe

            if len(self.position_history) > 0:
                profitable_periods = sum(
                    1 for period in self.position_history if any(pos > 0 for pos in period.get("positions", {}).values())
                )
                self.performance_metrics["win_rate"] = profitable_periods / len(self.position_history)

            risk_quality = self._calculate_risk_quality()
            self.performance_metrics["risk_quality"] = risk_quality

            return {
                "performance_updated": True,
                "recent_pnl": recent_pnl,
                "total_exposure_pct": self.performance_metrics["total_exposure"],
                "sharpe_ratio": self.performance_metrics["sharpe"],
                "risk_quality": risk_quality,
            }

        except Exception as e:
            self.logger.warning(f"Performance update failed: {e}")
            return {"performance_updated": False, "error": str(e)}

    def _calculate_risk_quality(self) -> float:
        """Calculate comprehensive risk quality score"""
        try:
            quality_factors: List[float] = []

            if self.current_var > 0:
                var_quality = max(0.0, float(1.0 - (self.current_var / 0.05)))
                quality_factors.append(var_quality)

            corr_quality = max(0.0, 1.0 - (self.max_correlation / 0.8))
            quality_factors.append(float(corr_quality))

            position_count = len([p for p in self.current_positions.values() if abs(p) > 0.001])
            diversification_quality = min(1.0, position_count / 5.0)
            quality_factors.append(float(diversification_quality))

            budget_quality = max(0.0, 1.0 - (self.daily_risk_used / max(self._cfg.risk_budget_daily, 1e-9)))
            quality_factors.append(float(budget_quality))

            return float(np.mean(quality_factors)) if quality_factors else 0.5

        except Exception as e:
            self.logger.warning(f"Risk quality calculation failed: {e}")
            return 0.5

    async def _update_risk_adjustment_factor_async(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update dynamic risk adjustment factor asynchronously"""
        try:
            balance = float(portfolio_data.get("balance", 0.0))
            drawdown = 0.0

            if hasattr(self, "_balance_history"):
                if balance < max(self._balance_history, default=balance):
                    peak_balance = max(self._balance_history) if self._balance_history else balance
                    drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0

            if drawdown <= 0.05:
                dd_factor = 1.0
            elif drawdown <= self._cfg.dd_limit:
                dd_factor = 1.0 - (drawdown - 0.05) / (self._cfg.dd_limit - 0.05) * 0.4
            else:
                dd_factor = float(0.6 * np.exp(-(drawdown - self._cfg.dd_limit) * 8.0))

            vol_factor = 1.0
            if self.performance_metrics["volatility"] > 0:
                if self.performance_metrics["volatility"] > 0.3:
                    vol_factor = 0.7
                elif self.performance_metrics["volatility"] > 0.2:
                    vol_factor = 0.85

            corr_factor = 1.0
            if self.max_correlation > self._cfg.correlation_threshold:
                excess_corr = self.max_correlation - self._cfg.correlation_threshold
                corr_factor = max(0.5, 1.0 - excess_corr * 2.0)

            regime_factor = 1.0
            if self.market_regime == "volatile":
                regime_factor = 0.8
            elif self.volatility_regime == "high":
                regime_factor = 0.85

            self.risk_adjustment = float(dd_factor * vol_factor * corr_factor * regime_factor)
            self.risk_adjustment = float(
                np.clip(self.risk_adjustment, self.min_risk_adjustment, self.max_risk_adjustment)
            )

            if not hasattr(self, "_balance_history"):
                self._balance_history = deque(maxlen=100)
            self._balance_history.append(balance)

            return {
                "risk_adjustment_updated": True,
                "drawdown": drawdown,
                "dd_factor": dd_factor,
                "vol_factor": vol_factor,
                "corr_factor": corr_factor,
                "regime_factor": regime_factor,
            }

        except Exception as e:
            self.logger.warning(f"Risk adjustment calculation failed: {e}")
            self.risk_adjustment = 0.8  # Conservative fallback
            return {"risk_adjustment_updated": False, "error": str(e)}

    async def _update_risk_budget_usage_async(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update daily risk budget usage asynchronously"""
        try:
            current_exposure = float(self.performance_metrics.get("total_exposure", 0.0))
            var_usage = float(self.current_var)

            self.daily_risk_used = max(current_exposure * 0.5, var_usage)

            budget_violation = False
            if self.daily_risk_used > self._cfg.risk_budget_daily:
                self.risk_budget_violations += 1
                budget_violation = True

            return {
                "risk_budget_updated": True,
                "daily_risk_used": self.daily_risk_used,
                "risk_budget_daily": self._cfg.risk_budget_daily,
                "budget_violation": budget_violation,
                "total_violations": self.risk_budget_violations,
            }

        except Exception as e:
            self.logger.warning(f"Risk budget update failed: {e}")
            return {"risk_budget_updated": False, "error": str(e)}

    async def _update_dynamic_position_limits(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update dynamic position limits based on risk conditions"""
        try:
            base_limit = float(self._cfg.max_position_pct)

            adjusted_limit = base_limit * float(self.risk_adjustment)

            if self.max_correlation > 0.7:
                correlation_penalty = 1.0 - (self.max_correlation - 0.7) * 2.0
                adjusted_limit *= max(0.5, correlation_penalty)

            if self.market_regime == "volatile":
                adjusted_limit *= 0.8
            elif self.market_regime == "trending":
                adjusted_limit *= 1.1

            if self.volatility_regime == "high":
                adjusted_limit *= 0.7
            elif self.volatility_regime == "low":
                adjusted_limit *= 1.2

            if self.bootstrap_mode:
                adjusted_limit *= 1.3

            final_limit = float(np.clip(adjusted_limit, self._cfg.min_position_pct, self._cfg.max_position_pct))
            old_limits = self.position_limits.copy()

            for instrument in self.instruments:
                self.position_limits[instrument] = final_limit

            return {
                "position_limits_updated": True,
                "base_limit": base_limit,
                "adjusted_limit": adjusted_limit,
                "final_limit": final_limit,
                "limits_changed": old_limits != self.position_limits,
            }

        except Exception as e:
            self.logger.warning(f"Position limits update failed: {e}")
            return {"position_limits_updated": False, "error": str(e)}

    async def _check_portfolio_risk_violations(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for portfolio risk violations"""
        try:
            violations: List[str] = []

            exposure = float(self.performance_metrics.get("total_exposure", 0.0))
            if exposure > self._cfg.max_portfolio_exposure:
                violations.append(
                    f"Portfolio exposure {exposure:.1%} > limit {self._cfg.max_portfolio_exposure:.1%}"
                )
                self.limit_violations += 1

            for instrument, position in self.current_positions.items():
                limit = float(self.position_limits.get(instrument, self._cfg.max_position_pct))
                if abs(position) > limit:
                    violations.append(f"{instrument} position {abs(position):.1%} > limit {limit:.1%}")
                    self.limit_violations += 1

            if self.current_var > 0.05:
                violations.append(f"Portfolio VaR {self.current_var:.1%} > 5% limit")

            if self.max_correlation > 0.9 and len(self.current_positions) > 1:
                violations.append(f"High correlation {self.max_correlation:.2f} with multiple positions")
                self.correlation_alerts += 1

            if violations:
                self.logger.warning(
                    format_operator_message(
                        message="PORTFOLIO_RISK_VIOLATIONS",
                        icon="[ALERT]",
                        violation_count=len(violations),
                        violations="; ".join(violations[:3]),
                        context="risk_violations",
                    )
                )
                for violation in violations:
                    self.risk_events.append(
                        {
                            "timestamp": datetime.datetime.now().isoformat(),
                            "type": "violation",
                            "description": violation,
                            "portfolio_data": portfolio_data.copy(),
                        }
                    )

            if len(self.risk_events) > 50:
                self.risk_events = self.risk_events[-50:]

            return {
                "violations_checked": True,
                "violations_found": len(violations),
                "violations": violations,
                "total_limit_violations": self.limit_violations,
                "correlation_alerts": self.correlation_alerts,
            }

        except Exception as e:
            self.logger.warning(f"Risk violation check failed: {e}")
            return {"violations_checked": False, "error": str(e)}

    async def _update_operational_mode(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update operational mode based on risk level"""
        try:
            old_mode = self.current_mode

            if self.current_var > 0.08 or self.daily_risk_used > self._cfg.risk_budget_daily * 1.5:
                new_mode = RiskMode.EMERGENCY
            elif self.current_var > 0.05 or self.limit_violations > 5:
                new_mode = RiskMode.CRITICAL
            elif self.current_var > 0.03 or self.max_correlation > 0.8:
                new_mode = RiskMode.ELEVATED
            elif self.bootstrap_mode:
                new_mode = RiskMode.BOOTSTRAP
            else:
                new_mode = RiskMode.NORMAL

            mode_changed = False
            if new_mode != old_mode:
                self.current_mode = new_mode
                self.mode_start_time = datetime.datetime.now()
                mode_changed = True
                self.logger.info(
                    format_operator_message(
                        message="RISK_MODE_CHANGE",
                        icon="[RELOAD]",
                        old_mode=old_mode.value,
                        new_mode=new_mode.value,
                        var=f"{self.current_var:.2%}",
                        correlation=f"{self.max_correlation:.2f}",
                        context="mode_transition",
                    )
                )

            return {
                "mode_updated": True,
                "current_mode": self.current_mode.value,
                "mode_changed": mode_changed,
                "old_mode": old_mode.value if mode_changed else None,
                "mode_duration": (datetime.datetime.now() - self.mode_start_time).total_seconds(),
            }

        except Exception as e:
            self.logger.warning(f"Mode update failed: {e}")
            return {"mode_updated": False, "error": str(e)}

    async def _generate_portfolio_thesis(self, portfolio_data: Dict[str, Any], result: Dict[str, Any]) -> str:
        """Generate comprehensive portfolio thesis"""
        try:
            var = float(self.current_var)
            correlation = float(self.max_correlation)
            exposure = float(self.performance_metrics.get("total_exposure", 0.0))
            mode = self.current_mode.value

            thesis_parts = [
                f"Portfolio Risk: {mode.upper()} mode with {var:.2%} VaR and {exposure:.1%} exposure",
                f"Risk Quality: {self.performance_metrics['risk_quality']:.2f} quality score",
            ]

            if var > 0.05:
                thesis_parts.append("HIGH RISK: VaR exceeds 5% threshold")
            elif correlation > 0.8:
                thesis_parts.append(f"CONCENTRATION RISK: {correlation:.1%} correlation detected")
            elif exposure > 0.8:
                thesis_parts.append(f"EXPOSURE RISK: {exposure:.1%} portfolio exposure")

            sharpe = float(self.performance_metrics.get("sharpe", 0.0))
            if sharpe > 1.0:
                thesis_parts.append(f"Strong performance: {sharpe:.2f} Sharpe ratio")
            elif sharpe < 0:
                thesis_parts.append(f"Poor performance: {sharpe:.2f} Sharpe ratio")

            thesis_parts.append(f"Market context: {self.market_regime.upper()} regime, {self.volatility_regime.upper()} volatility")

            violations = result.get("violations_found", 0)
            if int(violations) > 0:
                thesis_parts.append(f"VIOLATIONS: {violations} risk limit breaches detected")

            budget_used_pct = (self.daily_risk_used / max(self._cfg.risk_budget_daily, 1e-9)) * 100.0
            thesis_parts.append(f"Risk budget: {budget_used_pct:.0f}% utilized")

            return " | ".join(thesis_parts)

        except Exception as e:
            return f"Portfolio thesis generation failed: {str(e)} - Core risk monitoring functional"

    async def _update_portfolio_smart_bus(self, result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with portfolio results (ONLY provides)"""
        try:
            # Portfolio risk
            self.smart_bus.set("portfolio_risk", result.get("portfolio_risk", {}), module="PortfolioRiskSystem", thesis=thesis)

            # Risk metrics
            self.smart_bus.set(
                "risk_metrics",
                result.get("risk_metrics", {}),
                module="PortfolioRiskSystem",
                thesis="Comprehensive portfolio risk metrics and analysis",
            )

            # Position limits
            self.smart_bus.set(
                "position_limits",
                result.get("position_limits", {}),
                module="PortfolioRiskSystem",
                thesis="Dynamic position limits based on current risk conditions",
            )

            # Risk data (consolidated)
            self.smart_bus.set(
                "risk_data",
                result.get("risk_data", {}),
                module="PortfolioRiskSystem",
                thesis="Consolidated portfolio risk data",
            )

            # Risk signals
            self.smart_bus.set(
                "risk_signals",
                result.get("risk_signals", {}),
                module="PortfolioRiskSystem",
                thesis="Risk signals and alerts",
            )

            # Risk score
            self.smart_bus.set(
                "risk_score",
                result.get("risk_score", 0.0),
                module="PortfolioRiskSystem",
                thesis="Overall portfolio risk score",
            )

            # Trade data (passthrough) → use namespaced key to avoid owner conflict with Executor
            # Prefer an explicitly provided portfolio_trade_data payload; fall back to trade_data if present
            self.smart_bus.set(
                "portfolio_trade_data",
                result.get("portfolio_trade_data", result.get("trade_data", {})),
                module="PortfolioRiskSystem",
                thesis="Recent trades and positions (risk view)",
            )

            # Trading data summary (passthrough)
            self.smart_bus.set(
                "trading_data",
                result.get("trading_data", {}),
                module="PortfolioRiskSystem",
                thesis="Trading data summary (risk view)",
            )

            # Portfolio risk proposal (provides)
            if "portfolio_risk_proposal" in result:
                self.smart_bus.set(
                    "portfolio_risk_proposal",
                    result["portfolio_risk_proposal"],
                    module="PortfolioRiskSystem",
                    thesis="Portfolio risk proposal and recommendations",
                )

            # NOTE: We intentionally do NOT write non-provides (e.g., internal analytics) to the bus.

        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    # --- Voting helper: optional coordinator bus write (kept separate from provides) ---
    async def _write_voting_to_bus(self, vote_payload: Dict[str, Any]) -> None:
        """Publish module-scoped voting outputs to avoid canonical-key thrash.

        Coordinator (EnhancedVotingCommitteeCoordinator) owns the canonical
        committee decision. We publish namespaced keys here.
        """
        try:
            self.smart_bus.set(
                "PortfolioRiskSystem_voting_proposal",
                vote_payload,
                module="PortfolioRiskSystem",
                thesis="PortfolioRiskSystem voting proposal",
            )
            self.smart_bus.set(
                "PortfolioRiskSystem_confidence",
                float(vote_payload.get("confidence", 0.0)),
                module="PortfolioRiskSystem",
                thesis="PortfolioRiskSystem vote confidence",
            )
        except Exception as e:
            self.logger.warning(f"Voting bus write failed: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no portfolio data is available (contract-safe)"""
        self.logger.warning("No portfolio data available - using fallback mode")
        thesis = "Portfolio risk operating with cached state due to missing inputs"
        return {
            "portfolio_risk": {
                "current_mode": self.current_mode.value,
                "var_95": self.current_var,
                "max_correlation": self.max_correlation,
                "risk_adjustment": self.risk_adjustment,
                "bootstrap_mode": self.bootstrap_mode,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "position_limits": {
                "position_limits": self.position_limits.copy(),
                "risk_adjustment": self.risk_adjustment,
                "base_limit": self._cfg.max_position_pct,
                "bootstrap_mode": self.bootstrap_mode,
            },
            "risk_metrics": {
                "var_95": self.current_var,
                "correlation_matrix": self.correlation_matrix.tolist() if self.correlation_matrix is not None else None,
                "max_correlation": self.max_correlation,
                "portfolio_volatility": self.performance_metrics.get("volatility", 0.0),
                "risk_quality": self.performance_metrics.get("risk_quality", 0.5),
                "total_exposure": self.performance_metrics.get("total_exposure", 0.0),
            },
            "risk_data": {
                "current_mode": self.current_mode.value,
                "metrics": {
                    "var_95": self.current_var,
                    "max_correlation": self.max_correlation,
                    "portfolio_volatility": self.performance_metrics.get("volatility", 0.0),
                    "risk_quality": self.performance_metrics.get("risk_quality", 0.5),
                    "total_exposure": self.performance_metrics.get("total_exposure", 0.0),
                    "daily_risk_used": getattr(self, "daily_risk_used", 0.0),
                },
                "limits": {
                    "position_limits": self.position_limits.copy(),
                    "base_limit": self._cfg.max_position_pct,
                    "risk_adjustment": self.risk_adjustment,
                    "bootstrap_mode": self.bootstrap_mode,
                },
                "alerts": {
                    "limit_violations": getattr(self, "limit_violations", 0),
                    "correlation_alerts": getattr(self, "correlation_alerts", 0),
                },
            },
            "risk_signals": {"violations": [], "mode": self.current_mode.value, "budget_violation": False},
            "risk_score": float(self.performance_metrics.get("risk_quality", 0.5)),
            "trade_data": {"recent_trades": [], "positions": []},
            "trading_data": {
                "timestamp": datetime.datetime.now().isoformat(),
                "prices_available": False,
                "positions_count": 0,
            },
            "portfolio_risk_proposal": {
                "timestamp": time.time(),
                "confidence": 0.0,
                "actions": [],
                "status": self.get_current_risk_status(),
                "note": "fallback_no_data",
            },
            "_thesis": thesis,
            "fallback_reason": "no_portfolio_data",
        }

    async def _handle_portfolio_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle portfolio risk errors (contract-safe)"""
        processing_time = (time.time() - start_time) * 1000.0

        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()

        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
            self.circuit_breaker["state"] = "OPEN"
            self._health_status = "warning"

        error_context = self.error_pinpointer.analyze_error(error, "PortfolioRiskSystem")
        explanation = self.english_explainer.explain_error("PortfolioRiskSystem", str(error), "portfolio risk calculation")

        self.logger.error(
            format_operator_message(
                message="PORTFOLIO_RISK_ERROR",
                icon="[CRASH]",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                circuit_breaker_state=self.circuit_breaker["state"],
                context="portfolio_error",
            )
        )

        self._record_failure(error)

        payload = self._create_error_fallback_response(f"error: {str(error)}")
        # Ensure all provides + thesis + success flag present; success added by caller
        return payload

    def _create_error_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        thesis = f"Portfolio risk fallback: {reason}"
        return {
            "portfolio_risk": {
                "current_mode": RiskMode.EMERGENCY.value,
                "var_95": 0.1,
                "max_correlation": 0.9,
                "risk_adjustment": self.min_risk_adjustment,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            "position_limits": {
                "position_limits": self.position_limits.copy(),
                "risk_adjustment": self.risk_adjustment,
                "base_limit": self._cfg.max_position_pct,
                "bootstrap_mode": self.bootstrap_mode,
            },
            "risk_metrics": {
                "var_95": 0.1,
                "correlation_matrix": self.correlation_matrix.tolist() if self.correlation_matrix is not None else None,
                "max_correlation": 0.9,
                "portfolio_volatility": self.performance_metrics.get("volatility", 0.0),
                "risk_quality": self.performance_metrics.get("risk_quality", 0.5),
                "total_exposure": self.performance_metrics.get("total_exposure", 0.0),
            },
            "risk_data": {
                "current_mode": RiskMode.EMERGENCY.value,
                "metrics": {
                    "var_95": 0.1,
                    "max_correlation": 0.9,
                    "portfolio_volatility": self.performance_metrics.get("volatility", 0.0),
                    "risk_quality": self.performance_metrics.get("risk_quality", 0.5),
                    "total_exposure": self.performance_metrics.get("total_exposure", 0.0),
                    "daily_risk_used": getattr(self, "daily_risk_used", 0.0),
                },
                "limits": {
                    "position_limits": self.position_limits.copy(),
                    "base_limit": self._cfg.max_position_pct,
                    "risk_adjustment": self.risk_adjustment,
                    "bootstrap_mode": self.bootstrap_mode,
                },
                "alerts": {
                    "limit_violations": getattr(self, "limit_violations", 0),
                    "correlation_alerts": getattr(self, "correlation_alerts", 0),
                },
            },
            "risk_signals": {"violations": [], "mode": RiskMode.EMERGENCY.value, "budget_violation": False},
            "risk_score": float(self.performance_metrics.get("risk_quality", 0.5)),
            "trade_data": {"recent_trades": [], "positions": []},
            "trading_data": {
                "timestamp": datetime.datetime.now().isoformat(),
                "prices_available": False,
                "positions_count": 0,
            },
            "portfolio_risk_proposal": {
                "timestamp": time.time(),
                "confidence": 0.0,
                "actions": [],
                "status": self.get_current_risk_status(),
                "note": "fallback_error",
            },
            "_thesis": thesis,
            "circuit_breaker_state": self.circuit_breaker["state"],
            "fallback_reason": reason,
        }

    def _update_portfolio_health(self):
        """Update portfolio health metrics"""
        try:
            if not hasattr(self, "performance_metrics"):
                return

            if self.performance_metrics["risk_quality"] < self._cfg.min_risk_quality:
                self._health_status = "warning"
            else:
                self._health_status = "healthy"

            if self.circuit_breaker["state"] == "OPEN":
                self._health_status = "warning"

            if self.current_var > 0.08 or self.max_correlation > 0.95:
                self._health_status = "warning"

            self._last_health_check = time.time()

        except Exception as e:
            self.logger.error(f"Portfolio health check failed: {e}")
            self._health_status = "warning"

    def _analyze_risk_effectiveness(self):
        """Analyze risk management effectiveness"""
        try:
            if not hasattr(self, "position_history") or not hasattr(self, "performance_metrics"):
                return

            if len(self.position_history) >= 10:
                recent_performance = float(self.performance_metrics.get("risk_quality", 0.5))

                if recent_performance > 0.8:
                    self.logger.info(
                        format_operator_message(
                            message="HIGH_RISK_EFFECTIVENESS",
                            icon="[TARGET]",
                            quality_score=f"{recent_performance:.2f}",
                            var=f"{self.current_var:.2%}",
                            context="risk_analysis",
                        )
                    )
                elif recent_performance < 0.3:
                    self.logger.warning(
                        format_operator_message(
                            message="LOW_RISK_EFFECTIVENESS",
                            icon="[WARN]",
                            quality_score=f"{recent_performance:.2f}",
                            violations=self.limit_violations,
                            context="risk_analysis",
                        )
                    )

        except Exception as e:
            self.logger.error(f"Risk effectiveness analysis failed: {e}")

    def _adapt_risk_parameters(self):
        """Continuous risk parameter adaptation"""
        try:
            if not hasattr(self, "market_regime") or not hasattr(self, "_adaptive_params") or not hasattr(self, "performance_metrics"):
                return

            if self.market_regime == "volatile":
                self._adaptive_params["correlation_sensitivity"] = min(
                    1.5, self._adaptive_params["correlation_sensitivity"] * 1.01
                )
            else:
                self._adaptive_params["correlation_sensitivity"] = max(
                    0.7, self._adaptive_params["correlation_sensitivity"] * 0.999
                )

            if self.performance_metrics["volatility"] > 0.3:
                self._adaptive_params["volatility_tolerance"] = max(
                    0.6, self._adaptive_params["volatility_tolerance"] * 0.99
                )
            elif self.performance_metrics["volatility"] < 0.1:
                self._adaptive_params["volatility_tolerance"] = min(
                    1.4, self._adaptive_params["volatility_tolerance"] * 1.005
                )

        except Exception as e:
            self.logger.warning(f"Risk parameter adaptation failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            "PortfolioRiskSystem", "portfolio_risk_calculation", processing_time, True
        )
        if self.circuit_breaker["state"] == "OPEN":
            self.circuit_breaker["failures"] = 0
            self.circuit_breaker["state"] = "CLOSED"

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric("PortfolioRiskSystem", "portfolio_risk_calculation", 0.0, False)

    # ================== PUBLIC INTERFACE METHODS ==================

    def get_position_limits(self) -> Dict[str, float]:
        """Get current position limits for each instrument"""
        return self.position_limits.copy()

    def check_risk_limits(self, proposed_positions: Dict[str, float]) -> Tuple[bool, str]:
        """Check if proposed positions violate risk limits"""
        try:
            total_exposure = sum(abs(pos) for pos in proposed_positions.values())

            if total_exposure > self._cfg.max_portfolio_exposure:
                return (
                    False,
                    f"Total exposure {total_exposure:.1%} exceeds limit {self._cfg.max_portfolio_exposure:.1%}",
                )

            for inst, pos in proposed_positions.items():
                limit = float(self.position_limits.get(inst, self._cfg.max_position_pct))
                if abs(pos) > limit:
                    return False, f"{inst} position {abs(pos):.1%} exceeds limit {limit:.1%}"

            if self.current_var > 0.05:
                return False, f"Portfolio VaR {self.current_var:.1%} exceeds 5% limit"

            if self.max_correlation > 0.9 and len(proposed_positions) > 1:
                return False, f"High correlation {self.max_correlation:.2f} with multiple positions"

            return True, "All risk checks passed"

        except Exception as e:
            self.logger.error(f"Risk limit check failed: {e}")
            return False, "Risk limit check failed"

    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics"""
        return {
            "var": float(self.current_var),
            "max_correlation": float(self.max_correlation),
            "risk_adjustment": float(self.risk_adjustment),
            "total_exposure": float(self.performance_metrics["total_exposure"]),
            "position_count": float(len([p for p in self.current_positions.values() if abs(p) > 0.001])),
            "bootstrap_mode": float(1.0 if self.bootstrap_mode else 0.0),
            "portfolio_volatility": float(self.performance_metrics["volatility"]),
            "sharpe_ratio": float(self.performance_metrics["sharpe"]),
            "max_drawdown": float(self.performance_metrics["max_dd"]),
            "risk_budget_used": float(self.daily_risk_used),
            "risk_budget_available": float(max(0.0, self._cfg.risk_budget_daily - self.daily_risk_used)),
            "risk_quality": float(self.performance_metrics["risk_quality"]),
        }

    def get_observation_components(self) -> np.ndarray:
        """Get portfolio risk features for observation"""
        try:
            features = [
                float(self.current_var),
                float(self.max_correlation),
                float(self.risk_adjustment),
                float(1.0 if self.bootstrap_mode else 0.0),
                float(len(self.current_positions)),
                float(sum(abs(p) for p in self.current_positions.values())),
                float(self.performance_metrics["sharpe"]),
                float(self.performance_metrics["max_dd"]),
                float(self.performance_metrics["volatility"]),
                float(self.daily_risk_used / self._cfg.risk_budget_daily) if self._cfg.risk_budget_daily > 0 else 0.0,
                float(self.performance_metrics["risk_quality"]),
                float(1.0 if self.current_mode in [RiskMode.CRITICAL, RiskMode.EMERGENCY] else 0.0),
            ]
            return np.array(features, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"Risk observation generation failed: {e}")
            return np.array([0.0] * 12, dtype=np.float32)

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            "status": self._health_status,
            "last_check": self._last_health_check,
            "circuit_breaker": self.circuit_breaker["state"],
            "current_mode": self.current_mode.value,
            "risk_quality": self.performance_metrics["risk_quality"],
            "var_95": self.current_var,
            "max_correlation": self.max_correlation,
            "bootstrap_mode": self.bootstrap_mode,
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    def get_portfolio_risk_report(self) -> str:
        """Generate operator-friendly portfolio risk report"""

        if self.current_var > 0.05:
            var_status = "[ALERT] High Risk"
        elif self.current_var > 0.03:
            var_status = "[WARN] Elevated"
        else:
            var_status = "[OK] Normal"

        if self.max_correlation > 0.8:
            corr_status = "[ALERT] High"
        elif self.max_correlation > 0.6:
            corr_status = "[WARN] Moderate"
        else:
            corr_status = "[OK] Low"

        mode_emoji = {
            RiskMode.INITIALIZATION: "[RELOAD]",
            RiskMode.BOOTSTRAP: "🏗️",
            RiskMode.NORMAL: "[OK]",
            RiskMode.ELEVATED: "[WARN]",
            RiskMode.CRITICAL: "[ALERT]",
            RiskMode.EMERGENCY: "🆘",
        }

        mode_status = f"{mode_emoji.get(self.current_mode, '❓')} {self.current_mode.value.upper()}"

        health_emoji = "[OK]" if self._health_status == "healthy" else "[WARN]"
        cb_status = "[RED] OPEN" if self.circuit_breaker["state"] == "OPEN" else "[GREEN] CLOSED"

        return f"""
💼 ENHANCED PORTFOLIO RISK SYSTEM v4.0
═══════════════════════════════════════════════════
[TARGET] Risk Mode: {mode_status}
[STATS] VaR Status: {var_status} ({self.current_var:.2%})
🔗 Correlation: {corr_status} ({self.max_correlation:.2f})
🏗️ Bootstrap Mode: {'[OK] Active' if self.bootstrap_mode else '[FAIL] Inactive'}

[HEALTH] SYSTEM HEALTH
• Status: {health_emoji} {self._health_status.upper()}
• Circuit Breaker: {cb_status}
• Risk Quality: {self.performance_metrics['risk_quality']:.2f}

[STATS] PORTFOLIO METRICS
• Current VaR (95%): {self.current_var:.2%}
• Portfolio Volatility: {self.performance_metrics["volatility"]:.1%}
• Max Correlation: {self.max_correlation:.2f}
• Total Exposure: {self.performance_metrics["total_exposure"]:.1%}
• Sharpe Ratio: {self.performance_metrics["sharpe"]:.2f}
• Max Drawdown: {self.performance_metrics["max_dd"]:.1%}

[MONEY] RISK BUDGET
• Daily Budget: {self._cfg.risk_budget_daily:.1%}
• Used Today: {self.daily_risk_used:.1%}
• Available: {max(0, self._cfg.risk_budget_daily - self.daily_risk_used):.1%}
• Budget Violations: {self.risk_budget_violations}

[BALANCE] RISK ADJUSTMENT
• Current Factor: {self.risk_adjustment:.1%}
• Base Position Limit: {self._cfg.max_position_pct:.1%}
• Adjusted Limit Range: {self._cfg.min_position_pct:.1%} - {self._cfg.max_position_pct:.1%}

[TOOL] SYSTEM PERFORMANCE
• Trade Count: {self.trade_count}
• Limit Violations: {self.limit_violations}
• Correlation Alerts: {self.correlation_alerts}
• Recent Risk Events: {len(self.risk_events)}

[CHART] PORTFOLIO PERFORMANCE
• Total PnL: {self.performance_metrics["total_pnl"]:.2f}
• Recent PnL: {self.performance_metrics["recent_pnl"]:.2f}
• Win Rate: {self.performance_metrics["win_rate"]:.1%}

💡 CONFIGURATION
• Instruments: {len(self.instruments)} tracked
• VaR Window: {self._cfg.var_window} periods
• Correlation Window: {self._cfg.correlation_window} periods
• DD Limit: {self._cfg.dd_limit:.1%}
        """

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> Dict[str, Any]:
        """Legacy step interface for backward compatibility"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self.process(**kwargs))
            return result
        finally:
            loop.close()

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        for inst in self.instruments:
            self.returns_history[inst].clear()
        self.portfolio_returns.clear()
        self._last_prices.clear()

        self.current_positions.clear()
        self.position_history.clear()
        self.trade_count = 0
        self.bootstrap_mode = True

        self.performance_metrics = {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.5,
            "recent_pnl": 0.0,
            "total_pnl": 0.0,
            "volatility": 0.0,
            "var_95": 0.0,
            "total_exposure": 0.0,
            "risk_quality": 0.5,
        }

        self.risk_adjustment = 1.0
        self.current_var = 0.0
        self.correlation_matrix = None
        self.max_correlation = 0.0

        for inst in self.instruments:
            self.position_limits[inst] = float(self._cfg.max_position_pct)

        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"

        self.daily_risk_used = 0.0
        self.risk_budget_violations = 0

        self.portfolio_analytics.clear()
        self.regime_performance.clear()
        self.risk_events.clear()
        self.limit_violations = 0
        self.correlation_alerts = 0

        self.current_mode = RiskMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()

        self.circuit_breaker["failures"] = 0
        self.circuit_breaker["state"] = "CLOSED"
        self._health_status = "healthy"

        self._adaptive_params = {
            "dynamic_limit_scaling": 1.0,
            "correlation_sensitivity": 1.0,
            "volatility_tolerance": 1.0,
            "risk_adaptation_confidence": 0.5,
        }

        self.logger.info("[RELOAD] Enhanced Portfolio Risk System reset - all state cleared")

    async def calculate_confidence(self, action: Dict[str, Any], **kwargs) -> float:
        """Calculate confidence level based on portfolio risk metrics"""
        try:
            confidence_factors: List[float] = []

            if self.current_var > 0:
                var_confidence = max(0.0, float(1.0 - (self.current_var / 0.05)))
                confidence_factors.append(var_confidence * 0.3)

            correlation_confidence = max(0.0, 1.0 - self.max_correlation)
            confidence_factors.append(float(correlation_confidence) * 0.3)

            mode_confidence = {
                RiskMode.NORMAL: 0.9,
                RiskMode.ELEVATED: 0.7,
                RiskMode.CRITICAL: 0.5,
                RiskMode.EMERGENCY: 0.2,
                RiskMode.BOOTSTRAP: 0.6,
                RiskMode.INITIALIZATION: 0.3,
            }.get(self.current_mode, 0.5)
            confidence_factors.append(mode_confidence * 0.4)

            total_confidence = sum(confidence_factors)

            if self.risk_adjustment < 1.0:
                total_confidence *= self.risk_adjustment

            # Down-weight if bootstrap (less information)
            if self.bootstrap_mode:
                total_confidence *= 0.9

            # Down-weight if proposed jump is large
            try:
                target_adj = float(action.get("target_risk_adjustment", self.risk_adjustment))
                jump = abs(target_adj - self.risk_adjustment)
                if jump > 0.25:
                    total_confidence *= 0.9
            except Exception:
                pass

            return max(0.0, min(1.0, float(total_confidence)))

        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    async def propose_action(self, **kwargs) -> Dict[str, Any]:
        """Propose portfolio risk management actions based on current state (also written to bus by process)"""
        try:
            status = self.get_current_risk_status()
            actions: List[Dict[str, Any]] = []

            if self.current_mode == RiskMode.EMERGENCY:
                actions.append(
                    {
                        "type": "emergency_stop",
                        "priority": "critical",
                        "message": "Emergency mode active - recommend immediate position review",
                    }
                )

            if self.current_var > 0.03:
                actions.append(
                    {
                        "type": "risk_reduction",
                        "priority": "high",
                        "message": f"High VaR detected: {self.current_var:.3f}",
                        "recommendation": "Consider reducing position sizes",
                    }
                )

            if self.max_correlation > 0.8:
                actions.append(
                    {
                        "type": "diversification",
                        "priority": "medium",
                        "message": f"High correlation detected: {self.max_correlation:.3f}",
                        "recommendation": "Consider diversifying portfolio",
                    }
                )

            if self.risk_adjustment < 0.8:
                actions.append(
                    {
                        "type": "risk_scaling",
                        "priority": "medium",
                        "message": f"Risk scaling active: {self.risk_adjustment:.3f}",
                        "recommendation": "Position sizes are being scaled down",
                    }
                )

            if len(self.portfolio_returns) > 10:
                recent_returns = list(self.portfolio_returns)[-10:]
                recent_volatility = float(np.std(recent_returns)) if recent_returns else 0.0
                if recent_volatility > 0.02 and self.current_mode == RiskMode.NORMAL:
                    actions.append(
                        {
                            "type": "mode_change",
                            "priority": "medium",
                            "message": f"High volatility detected: {recent_volatility:.4f}",
                            "recommendation": "Consider switching to elevated risk mode",
                        }
                    )

            proposal = {
                "timestamp": time.time(),
                "confidence": await self.calculate_confidence({}, **kwargs),
                "actions": actions,
                "status": status,
                "risk_metrics": {
                    "var_95": self.current_var,
                    "max_correlation": self.max_correlation,
                    "risk_adjustment": self.risk_adjustment,
                },
            }

            # Keep this write for callers that invoke propose_action() directly
            self.smart_bus.set(
                "portfolio_risk_proposal",
                proposal,
                module="PortfolioRiskSystem",
                thesis=f"Portfolio risk analysis with {len(actions)} recommendations",
            )

            return proposal

        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {"timestamp": time.time(), "confidence": 0.0, "actions": [], "error": str(e)}

    def get_current_risk_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio risk status"""
        try:
            return {
                "mode": self.current_mode.value,
                "risk_adjustment": self.risk_adjustment,
                "current_var": self.current_var,
                "max_correlation": self.max_correlation,
                "bootstrap_mode": self.bootstrap_mode,
                "rebalance_trigger": float(self._cfg.correlation_threshold),
            }
        except Exception as e:
            self.logger.error(f"Risk status retrieval failed: {e}")
            return {}

    # ================== VOTING INTERFACE ==================

    async def vote(self) -> Dict[str, Any]:
        """
        Build a voting proposal for the committee.
        Outputs a compact payload with desired portfolio risk posture and confidence.

        Schema:
        {
            "member": "PortfolioRiskSystem",
            "type": "risk_posture",
            "posture": "reduce" | "increase" | "maintain" | "halt",
            "target_risk_adjustment": 0.92,
            "target_position_limit": 0.18,
            "bounds": {
                "risk_adjustment": [min_adj, max_adj],
                "position_limit": [min_limit, max_limit]
            },
            "rationale": "...",
            "confidence": 0.78,
            "context": {...},
            "timestamp": 1700000000.0
        }
        """
        try:
            now = time.time()
            cur_adj = float(self.risk_adjustment)
            base_limit = float(self._cfg.max_position_pct)

            # Decide posture
            if self.circuit_breaker["state"] == "OPEN" or self.current_mode == RiskMode.EMERGENCY:
                posture = "halt"
                target_adj = max(self.min_risk_adjustment, min(cur_adj, 0.6))
                rationale = "Circuit breaker/emergency posture."
            else:
                if self.current_mode in (RiskMode.CRITICAL,):
                    posture = "reduce"
                    target_adj = max(self.min_risk_adjustment, min(cur_adj, 0.75))
                    rationale = "Critical conditions (VaR/violations) — reduce risk."
                elif self.current_mode == RiskMode.ELEVATED or self.current_var > 0.03 or self.max_correlation > 0.8:
                    posture = "reduce"
                    target_adj = max(self.min_risk_adjustment, min(cur_adj, 0.85))
                    rationale = "Elevated risk—VaR/correlation above thresholds."
                elif self.current_mode == RiskMode.BOOTSTRAP:
                    posture = "maintain"
                    target_adj = cur_adj
                    rationale = "Bootstrap phase — maintain until more data."
                else:
                    # NORMAL
                    if self.current_var < 0.02 and self.max_correlation < 0.6 and self.daily_risk_used < self._cfg.risk_budget_daily * 0.6:
                        posture = "increase"
                        target_adj = min(self.max_risk_adjustment, max(cur_adj, 1.1))
                        rationale = "Favorable conditions — increase cautiously."
                    else:
                        posture = "maintain"
                        target_adj = cur_adj
                        rationale = "Risk acceptable — maintain posture."

            # Map target adjustment to a suggested position limit
            suggested_limit = float(np.clip(base_limit * target_adj, self._cfg.min_position_pct, self._cfg.max_position_pct))

            # Confidence
            conf_input = {
                "action_type": "risk_posture",
                "posture": posture,
                "target_risk_adjustment": target_adj,
                "mode": self.current_mode.value,
            }
            confidence = await self.calculate_confidence(conf_input)

            payload = {
                "member": "PortfolioRiskSystem",
                "type": "risk_posture",
                "posture": posture,
                "target_risk_adjustment": float(target_adj),
                "target_position_limit": float(suggested_limit),
                "bounds": {
                    "risk_adjustment": [float(self.min_risk_adjustment), float(self.max_risk_adjustment)],
                    "position_limit": [float(self._cfg.min_position_pct), float(self._cfg.max_position_pct)],
                },
                "rationale": rationale,
                "confidence": float(confidence),
                "context": {
                    "mode": self.current_mode.value,
                    "var_95": float(self.current_var),
                    "max_correlation": float(self.max_correlation),
                    "risk_budget_used": float(self.daily_risk_used),
                    "market_regime": self.market_regime,
                    "volatility_regime": self.volatility_regime,
                },
                "timestamp": now,
            }
            return payload

        except Exception as e:
            self.logger.warning(f"Vote generation failed: {e}")
            return {
                "member": "PortfolioRiskSystem",
                "type": "risk_posture",
                "posture": "maintain",
                "target_risk_adjustment": float(self.risk_adjustment),
                "target_position_limit": float(self._cfg.max_position_pct),
                "bounds": {
                    "risk_adjustment": [float(self.min_risk_adjustment), float(self.max_risk_adjustment)],
                    "position_limit": [float(self._cfg.min_position_pct), float(self._cfg.max_position_pct)],
                },
                "rationale": f"fallback: {e}",
                "confidence": 0.5,
                "context": {},
                "timestamp": time.time(),
            }
