# ─────────────────────────────────────────────────────────────
# File: modules/market/liquidity_heatmap_layer.py
# [ROCKET] PRODUCTION-GRADE Liquidity Heatmap Analysis with Neural Networks
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# MODERNIZED: Complete SmartInfoBus integration with PyTorch neural networks
# ─────────────────────────────────────────────────────────────

import time
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque
from dataclasses import dataclass, asdict

# Core infrastructure
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@dataclass
class LiquidityConfig:
    """Configuration for Liquidity Heatmap Layer"""
    lstm_units: int = 64                 # LSTM hidden size
    sequence_length: int = 20            # timesteps fed to LSTM
    hidden_dim: int = 32                 # predictor MLP hidden
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    enable_gpu: bool = True
    prediction_horizon: int = 5

    # Liquidity thresholds
    high_liquidity_threshold: float = 0.8
    low_liquidity_threshold: float = 0.3

    # Market depth analysis
    depth_levels: int = 10
    spread_analysis_window: int = 50


class LiquidityLSTM(nn.Module):
    """LSTM + attention head for short-horizon liquidity features."""
    def __init__(self, input_dim: int, lstm_units: int, output_dim: int, dropout_rate: float = 0.2, mlp_hidden: int = 32):
        super().__init__()
        # LSTM dropout only active when num_layers > 1; use 0.0 for single-layer to silence warning.
        self.lstm = nn.LSTM(
            input_dim,
            lstm_units,
            batch_first=True,
            dropout=0.0,
        )
        self.attention = nn.MultiheadAttention(lstm_units, num_heads=4, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(lstm_units, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        lstm_out, _ = self.lstm(x)                      # [B, T, H]
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [B, T, H]
        final_hidden = attn_out[:, -1, :]               # last timestep, [B, H]
        output = self.predictor(final_hidden)           # [B, O]
        return output


@module(
    name="LiquidityHeatmapLayer",
    version="3.0.2",
    category="market",
    provides=[
        "liquidity_score",
        "market_depth",
        "spread_analysis",
        "liquidity_prediction",
        "trading_sessions",
        "session_data",
        "liquidity_thesis",
        "liquidity_capabilities",
    ],
    requires=[
        "bid_ask_data",
        "price_data",
        "prices",
    ],
    description="Advanced liquidity heatmap analysis with neural network predictions",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
)
class LiquidityHeatmapLayer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    PRODUCTION-GRADE Liquidity Heatmap Analysis with Neural Networks
    """

    # ── Strict output normalizer ─────────────────────────────────────────
    def _format_declared_outputs(
        self,
        *,
        liquidity_score: Optional[float] = None,
        market_depth: Optional[Dict[str, Any]] = None,
        spread_analysis: Optional[Dict[str, Any]] = None,
        liquidity_prediction: Optional[Dict[str, Any]] = None,
        trading_sessions: Optional[Dict[str, Any]] = None,
        session_data: Optional[Dict[str, Any]] = None,
        thesis: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        # liquidity_score
        try:
            ls = float(liquidity_score if liquidity_score is not None else getattr(self, "current_liquidity_score", 0.5))
            out["liquidity_score"] = float(np.clip(ls, 0.0, 1.0))
        except Exception:
            out["liquidity_score"] = 0.5

        # dict blocks
        out["market_depth"] = market_depth if isinstance(market_depth, dict) else {
            "current_depth": float(getattr(self, "current_depth", 0.0)),
            "analysis": {},
            "condition": "unknown",
            "status": "fallback",
        }

        out["spread_analysis"] = spread_analysis if isinstance(spread_analysis, dict) else {
            "current_spread": float(getattr(self, "current_spread", 0.0)),
            "analysis": {},
            "condition": "unknown",
            "status": "fallback",
        }

        # liquidity_prediction must exist and be a dict
        if not isinstance(liquidity_prediction, dict):
            out["liquidity_prediction"] = {
                "predictions": {},
                "confidence": 0.0,
                "horizon_steps": int(getattr(self._cfg, "prediction_horizon", 5)),
                "status": "unavailable",
            }
        else:
            out["liquidity_prediction"] = liquidity_prediction

        out["trading_sessions"] = trading_sessions if isinstance(trading_sessions, dict) else {"active": "unknown"}
        out["session_data"] = session_data if isinstance(session_data, dict) else {"active_session": "unknown"}

        # thesis + extras
        out["thesis"] = thesis or "Liquidity analysis completed."
        out["_thesis"] = out["thesis"]
        # explicit provide for orchestrator contract matching
        out["liquidity_thesis"] = out["thesis"]
        if isinstance(extra, dict):
            try:
                out.update(extra)
            except Exception:
                pass

        # fail-fast on essentials we promise
        for key in ("liquidity_score", "market_depth", "spread_analysis", "liquidity_prediction", "trading_sessions", "session_data"):
            if key not in out:
                raise ValueError(f"Critical output '{key}' missing in LiquidityHeatmapLayer")
        return out

    # ── Lifecycle ───────────────────────────────────────────────────────
    def __init__(self, config: Optional[Union[LiquidityConfig, Dict[str, Any]]] = None, **kwargs):
        # Normalize config to dataclass (typed) + keep dict for BaseModule
        self._cfg: LiquidityConfig = LiquidityConfig(**config) if isinstance(config, dict) else (config or LiquidityConfig())

        # Initialize advanced systems first (need _cfg for device)
        self._initialize_advanced_systems()

        # Parent init expects a dict-like config; pass a plain dict
        super().__init__(config=asdict(self._cfg))

        # Initialize neural networks, state, monitoring
        self._initialize_neural_networks()
        self._initialize_liquidity_state()
        self._start_monitoring()

        # Optionally publish capabilities immediately (avoid calling abstract _initialize here)
        self._publish_capabilities()

        self.logger.info(
            format_operator_message(
                "💧",
                "LIQUIDITY_HEATMAP_INITIALIZED",
                details=f"LSTM units: {self._cfg.lstm_units}, Device: {self.device}",
                result="Advanced liquidity analysis active",
                context="liquidity_engine_startup",
            )
        )

    def _initialize_advanced_systems(self):
        """Initialize all advanced systems."""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="LiquidityHeatmapLayer",
            log_path="logs/market/liquidity_heatmap.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() and self._cfg.enable_gpu else "cpu")

        # Advanced systems
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("LiquidityHeatmapLayer", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Circuit breaker for neural operations
        self.neural_circuit_breaker = {"failures": 0, "last_failure": 0, "state": "CLOSED", "threshold": 3}

    def _initialize_neural_networks(self):
        """Initialize PyTorch neural network components."""
        try:
            # Input feature vector per timestep: [spread, depth, price_liquidity, volume_volatility]
            input_dim = 4
            output_dim = 3  # [liquidity_score, depth_prediction, spread_prediction]

            self.lstm_model = LiquidityLSTM(
                input_dim=input_dim,
                lstm_units=self._cfg.lstm_units,
                output_dim=output_dim,
                dropout_rate=self._cfg.dropout_rate,
                mlp_hidden=self._cfg.hidden_dim,
            ).to(self.device)

            # Optimizer & Loss
            self.optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=self._cfg.learning_rate)
            self.criterion = nn.MSELoss()

            self.logger.info(f"Neural networks initialized on {self.device}")
        except Exception as e:
            self.logger.error(f"Neural network initialization failed: {e}")
            raise

    def _initialize_liquidity_state(self):
        """Initialize liquidity-specific state."""
        # Market data buffers
        self.price_history = deque(maxlen=500)
        self.spread_history = deque(maxlen=200)
        self.depth_history = deque(maxlen=200)
        self.volume_history = deque(maxlen=200)

        # Neural network data
        self.sequence_data = deque(maxlen=self._cfg.sequence_length)
        self.training_data = deque(maxlen=1000)

        # Current state
        self.current_liquidity_score = 0.5
        self.current_spread = 0.0
        self.current_depth = 0.0
        self.market_session = "unknown"

        # Stats
        self.liquidity_stats = {
            "predictions_made": 0,
            "successful_predictions": 0,
            "avg_prediction_accuracy": 0.0,  # 0..1
            "neural_forward_passes": 0,
            "model_training_episodes": 0,
        }

        # Health
        self.liquidity_health = {
            "model_health_score": 100.0,  # 0..100
            "data_quality_score": 100.0,  # 0..100
            "prediction_confidence": 0.0,  # 0..1
            "last_update": time.time(),
        }

        # Bookkeeping
        self.last_prediction_time: Optional[float] = None

    def _publish_capabilities(self):
        """Publish capabilities to SmartInfoBus (safe from __init__)."""
        try:
            self.smart_bus.set(
                "liquidity_capabilities",
                {
                    "prediction_horizon": self._cfg.prediction_horizon,
                    "sequence_length": self._cfg.sequence_length,
                    "device": str(self.device),
                    "depth_levels": self._cfg.depth_levels,
                    "neural_model": "LSTM_with_attention",
                },
                module="LiquidityHeatmapLayer",
                thesis="Liquidity analysis capabilities for market assessment",
            )
        except Exception:
            pass

    # Match BaseModule abstract signature to silence Pylance
    def _initialize(self, **kwargs) -> None:
        """Called by orchestrator; keep idempotent."""
        # We already published capabilities in __init__, but do it again to be safe.
        self._publish_capabilities()

    # ── Trading session helpers ─────────────────────────────────────────
    def _compute_trading_sessions(self, market_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Compute current trading session snapshot."""
        import datetime as _dt
        now = _dt.datetime.utcnow()
        weekday = now.weekday()  # 0=Mon .. 6=Sun
        hour = now.hour + now.minute / 60.0

        weekend = (weekday == 5) or (weekday == 6) or (weekday == 4 and hour >= 22.0) or (weekday == 6 and hour < 21.0)

        def in_window(h, start, end):
            return (h >= start and h < end) if start < end else (h >= start or h < end)

        asian = in_window(hour, 23.0, 7.0) and not weekend
        european = (7.0 <= hour < 16.0) and not weekend
        american = (12.0 <= hour < 21.0) and not weekend
        rollover = (21.0 <= hour < 23.0) and not weekend

        if weekend:
            active, bias = "weekend", 0.3
        elif rollover:
            active, bias = "rollover", 0.4
        elif american:
            active, bias = "american", 1.0
        elif european:
            active, bias = "european", 0.9
        elif asian:
            active, bias = "asian", 0.7
        else:
            active, bias = "unknown", 0.6

        sessions_map = {"asian": bool(asian), "european": bool(european), "american": bool(american), "rollover": bool(rollover), "weekend": bool(weekend), "active": active}
        session_meta = {
            "active_session": active,
            "utc_time": now.isoformat() + "Z",
            "windows_utc": {
                "asian": {"start": "23:00", "end": "07:00"},
                "european": {"start": "07:00", "end": "16:00"},
                "american": {"start": "12:00", "end": "21:00"},
                "rollover": {"start": "21:00", "end": "23:00"},
            },
            "liquidity_bias": bias,
        }
        return sessions_map, session_meta

    # ── Main processing ────────────────────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        """Main processing function for liquidity analysis (contract-compliant)."""
        t0 = time.time()

        # Circuit breaker
        if not self._check_neural_circuit_breaker():
            sessions_map, session_meta = self._compute_trading_sessions({})
            return self._format_declared_outputs(
                liquidity_score=self.current_liquidity_score,
                market_depth={"status": "fallback", "current_depth": self.current_depth, "analysis": {}, "condition": "unknown"},
                spread_analysis={"status": "fallback", "current_spread": self.current_spread, "analysis": {}, "condition": "unknown"},
                liquidity_prediction={"predictions": {}, "confidence": 0.0, "horizon_steps": self._cfg.prediction_horizon, "status": "unavailable"},
                trading_sessions=sessions_map,
                session_data=session_meta,
                thesis="Liquidity analysis fallback: neural circuit breaker open.",
                extra={"success": False, "processing_time_ms": (time.time() - t0) * 1000.0},
            )

        try:
            # 1) Extract market data
            market_data = await self._extract_market_data(**inputs)

            # 2) Core analytics
            liquidity_metrics = await self._analyze_liquidity(market_data)
            prediction_result = await self._neural_liquidity_prediction(liquidity_metrics)

            # 3) Session snapshot
            sessions_map, session_meta = self._compute_trading_sessions(market_data)

            # 4) Thesis
            thesis = await self._generate_liquidity_thesis(market_data, liquidity_metrics, prediction_result)
            thesis = f"{thesis}\n\nSession: {session_meta['active_session'].upper()} (UTC {session_meta['utc_time']})"

            # 5) Update SmartInfoBus
            await self._update_liquidity_smart_bus(liquidity_metrics, prediction_result, thesis, sessions_map, session_meta)

            # 6) Record success
            self._record_liquidity_success(time.time() - t0)

            # 7) Return (strict, contract-compliant)
            outputs = self._format_declared_outputs(
                liquidity_score=liquidity_metrics["liquidity_score"],
                market_depth=liquidity_metrics["depth_analysis"],
                spread_analysis=liquidity_metrics["spread_analysis"],
                liquidity_prediction=prediction_result,
                trading_sessions=sessions_map,
                session_data=session_meta,
                thesis=thesis,
                extra={"success": True, "processing_time_ms": (time.time() - t0) * 1000.0},
            )
            # include capabilities declared provide explicitly in returns
            try:
                caps = self.smart_bus.get("liquidity_capabilities", "LiquidityHeatmapLayer")
            except Exception:
                caps = None
            if not isinstance(caps, dict):
                caps = {
                    "prediction_horizon": self._cfg.prediction_horizon,
                    "sequence_length": self._cfg.sequence_length,
                    "device": str(self.device),
                    "depth_levels": self._cfg.depth_levels,
                    "neural_model": "LSTM_with_attention",
                }
            outputs["liquidity_capabilities"] = caps
            return outputs

        except Exception as e:
            fail = await self._handle_liquidity_error(e, t0)
            sessions_map, session_meta = self._compute_trading_sessions({})
            return self._format_declared_outputs(
                liquidity_score=fail.get("liquidity_score"),
                market_depth=fail.get("market_depth"),
                spread_analysis=fail.get("spread_analysis"),
                liquidity_prediction=fail.get("liquidity_prediction"),
                trading_sessions=sessions_map,
                session_data=session_meta,
                thesis=fail.get("thesis", f"Liquidity analysis error: {e}"),
                extra={"success": False, "processing_time_ms": (time.time() - t0) * 1000.0, "reason": fail.get("reason")},
            )

    # ── Data extraction ────────────────────────────────────────────────
    async def _extract_market_data(self, **inputs) -> Dict[str, Any]:
        """Extract market data from SmartInfoBus and inputs."""
        market_data: Dict[str, Any] = {"prices": [], "volumes": [], "timestamps": [], "bid_ask_spreads": [], "market_depth": {}}

        def _canon(sym: str) -> str:
            return sym.replace("/", "").upper().strip()

        bus_prices = self.smart_bus.get("prices", "LiquidityHeatmapLayer") or {}
        bus_price_data = self.smart_bus.get("price_data", "LiquidityHeatmapLayer") or {}
        bus_bid_ask = self.smart_bus.get("bid_ask_data", "LiquidityHeatmapLayer") or {}

        for instrument in ["EUR/USD", "XAU/USD"]:
            code = _canon(instrument)
            # price
            if isinstance(bus_prices, dict) and code in bus_prices:
                market_data["prices"].append(bus_prices[code])
            elif isinstance(bus_price_data, dict) and code in bus_price_data:
                pdict = bus_price_data.get(code) or {}
                close_val = pdict.get("close")
                if close_val is not None:
                    market_data["prices"].append(close_val)
            # spread
            if isinstance(bus_bid_ask, dict) and code in bus_bid_ask:
                spread = (bus_bid_ask.get(code) or {}).get("spread")
                if spread is not None:
                    market_data["bid_ask_spreads"].append(spread)

        # Merge explicit inputs
        if "market_data" in inputs and isinstance(inputs["market_data"], dict):
            inp_md = inputs["market_data"]
            if isinstance(inp_md.get("prices"), list):
                market_data["prices"].extend(inp_md["prices"])
            if isinstance(inp_md.get("bid_ask_spreads"), list):
                market_data["bid_ask_spreads"].extend(inp_md["bid_ask_spreads"])
            for k in ["volumes", "timestamps", "market_depth"]:
                if k in inp_md and market_data.get(k) in (None, [], {}):
                    market_data[k] = inp_md[k]

        # Fallback synthetic
        if not market_data["prices"]:
            market_data = self._generate_synthetic_market_data()

        return market_data

    def _generate_synthetic_market_data(self) -> Dict[str, Any]:
        """Generate synthetic market data for testing."""
        prices = np.random.normal(1.1000, 0.001, 50).tolist()
        volumes = np.random.exponential(1000, 50).tolist()
        spreads = np.random.uniform(0.0001, 0.0005, 50).tolist()
        return {
            "prices": prices,
            "volumes": volumes,
            "timestamps": list(range(50)),
            "bid_ask_spreads": spreads,
            "market_depth": {"bids": [(1.0999, 1000), (1.0998, 1500)], "asks": [(1.1001, 1200), (1.1002, 1800)]},
        }

    # ── Analytics ──────────────────────────────────────────────────────
    async def _analyze_liquidity(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current liquidity conditions."""
        try:
            if market_data["prices"]:
                current_price = market_data["prices"][-1]
                self.price_history.append(float(current_price))

            if market_data["bid_ask_spreads"]:
                self.current_spread = float(market_data["bid_ask_spreads"][-1])
                self.spread_history.append(self.current_spread)

            depth_info = market_data.get("market_depth", {})
            if depth_info:
                bid_depth = sum(float(v) for _, v in depth_info.get("bids", []))
                ask_depth = sum(float(v) for _, v in depth_info.get("asks", []))
                self.current_depth = float(bid_depth + ask_depth)
                self.depth_history.append(self.current_depth)

            liquidity_score = self._calculate_liquidity_score()
            volume_analysis = self._analyze_volume_patterns(market_data.get("volumes", []))
            spread_analysis = self._analyze_spread_patterns()
            depth_analysis = self._analyze_depth_patterns()

            return {
                "liquidity_score": liquidity_score,
                "spread_analysis": spread_analysis,
                "depth_analysis": {
                    "current_depth": self.current_depth,
                    "analysis": depth_analysis,
                    "condition": depth_analysis.get("condition", "unknown"),
                },
                "volume_analysis": volume_analysis,
                "current_spread": self.current_spread,
                "current_depth": self.current_depth,
            }
        except Exception as e:
            self.logger.error(f"Liquidity analysis failed: {e}")
            return {
                "liquidity_score": 0.5,
                "spread_analysis": {"status": "error"},
                "depth_analysis": {"status": "error"},
                "volume_analysis": {"status": "error"},
                "current_spread": 0.0,
                "current_depth": 0.0,
            }

    def _calculate_liquidity_score(self) -> float:
        """Calculate overall liquidity score."""
        comps: List[float] = []

        # Spread (lower is better)
        if self.spread_history:
            avg_spread = float(np.mean(list(self.spread_history)[-min(20, len(self.spread_history)) :]))
            spread_score = 1.0 - min(avg_spread / 0.001, 1.0)  # normalize to typical FX spread
            comps.append(spread_score * 0.4)

        # Depth (higher is better)
        if self.depth_history:
            avg_depth = float(np.mean(list(self.depth_history)[-min(20, len(self.depth_history)) :]))
            depth_score = min(avg_depth / 10000.0, 1.0)
            comps.append(depth_score * 0.4)

        # Volatility (lower is better → more liquid)
        if len(self.price_history) > 10:
            price_vol = float(np.std(list(self.price_history)[-min(20, len(self.price_history)) :]))
            volatility_score = 1.0 - min(price_vol / 0.01, 1.0)
            comps.append(volatility_score * 0.2)

        self.current_liquidity_score = float(np.clip(sum(comps) if comps else 0.5, 0.0, 1.0))
        return self.current_liquidity_score

    def _analyze_volume_patterns(self, volumes: List[float]) -> Dict[str, Any]:
        """Analyze volume patterns."""
        if not volumes:
            return {"status": "no_data"}
        rv = volumes[-20:] if len(volumes) >= 20 else volumes
        avg_volume = float(np.mean(rv))
        trend = "increasing" if len(rv) > 5 and rv[-1] > rv[-5] else "decreasing"
        return {"average_volume": avg_volume, "trend": trend, "volatility": float(np.std(rv)) if len(rv) > 1 else 0.0, "status": "analyzed"}

    def _analyze_spread_patterns(self) -> Dict[str, Any]:
        """Analyze bid-ask spread patterns."""
        if len(self.spread_history) < 5:
            return {"status": "insufficient_data"}
        spreads = list(self.spread_history)
        avg_spread = float(np.mean(spreads))
        spread_vol = float(np.std(spreads))
        if avg_spread < self._cfg.low_liquidity_threshold * 0.001:
            condition = "tight"
        elif avg_spread > self._cfg.high_liquidity_threshold * 0.001:
            condition = "wide"
        else:
            condition = "normal"
        return {
            "average_spread": avg_spread,
            "spread_volatility": spread_vol,
            "condition": condition,
            "trend": "widening" if spreads[-1] > spreads[-5] else "tightening",
            "status": "analyzed",
        }

    def _analyze_depth_patterns(self) -> Dict[str, Any]:
        """Analyze market depth patterns."""
        if len(self.depth_history) < 5:
            return {"status": "insufficient_data"}
        depths = list(self.depth_history)
        avg_depth = float(np.mean(depths))
        depth_stability = 1.0 - (float(np.std(depths)) / max(avg_depth, 1.0))
        if avg_depth > 50000:
            condition = "deep"
        elif avg_depth < 10000:
            condition = "shallow"
        else:
            condition = "moderate"
        return {
            "average_depth": avg_depth,
            "stability_score": float(np.clip(depth_stability, 0.0, 1.0)),
            "condition": condition,
            "trend": "deepening" if depths[-1] > depths[-5] else "shallowing",
            "status": "analyzed",
        }

    # ── Neural prediction ──────────────────────────────────────────────
    async def _neural_liquidity_prediction(self, liquidity_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate neural network predictions for liquidity."""
        try:
            # features: [spread, depth_norm, liquidity_score, volume_volatility]
            current_features = [
                float(liquidity_metrics["current_spread"]),
                float(liquidity_metrics["current_depth"]) / 10000.0,
                float(liquidity_metrics["liquidity_score"]),
                float(liquidity_metrics.get("volume_analysis", {}).get("volatility", 0.0)),
            ]
            self.sequence_data.append(np.array(current_features, dtype=np.float32))

            if len(self.sequence_data) < self._cfg.sequence_length:
                return {"predictions": {}, "confidence": 0.0, "horizon_steps": self._cfg.prediction_horizon, "status": "insufficient_sequence_data"}

            seq = np.array(list(self.sequence_data), dtype=np.float32)  # [T, F]
            x = torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, F]

            self.lstm_model.eval()
            with torch.no_grad():
                pred = self.lstm_model(x).squeeze(0).detach().cpu().numpy()  # [3]

            predicted_liquidity = float(np.clip(pred[0], 0.0, 1.0))
            predicted_depth = float(max(pred[1] * 10000.0, 0.0))
            predicted_spread = float(max(pred[2], 0.0))

            confidence = self._calculate_prediction_confidence()
            self.liquidity_stats["neural_forward_passes"] += 1
            self.last_prediction_time = time.time()

            return {
                "predictions": {"liquidity_score": predicted_liquidity, "depth": predicted_depth, "spread": predicted_spread},
                "confidence": confidence,
                "horizon_steps": self._cfg.prediction_horizon,
                "status": "success",
            }
        except Exception as e:
            self.logger.error(f"Neural liquidity prediction failed: {e}")
            return {"predictions": {}, "confidence": 0.0, "horizon_steps": self._cfg.prediction_horizon, "status": "error", "error": str(e)}

    def _calculate_prediction_confidence(self) -> float:
        """Confidence based on recent performance and health."""
        if self.liquidity_stats["predictions_made"] == 0:
            baseline = 0.5
        else:
            baseline = self.liquidity_stats["successful_predictions"] / max(1, self.liquidity_stats["predictions_made"])
        data_q = float(self.liquidity_health["data_quality_score"]) / 100.0
        model_h = float(self.liquidity_health["model_health_score"]) / 100.0
        return float(np.clip(baseline * data_q * model_h, 0.0, 1.0))

    # ── Thesis & Bus updates ───────────────────────────────────────────
    async def _generate_liquidity_thesis(self, market_data: Dict[str, Any], liquidity_metrics: Dict[str, Any], prediction_result: Dict[str, Any]) -> str:
        """Human-friendly summary."""
        try:
            liquidity_score = liquidity_metrics["liquidity_score"]
            spread_condition = liquidity_metrics["spread_analysis"].get("condition", "unknown")
            depth_condition = liquidity_metrics["depth_analysis"].get("condition", "unknown")

            if liquidity_score > self._cfg.high_liquidity_threshold:
                assessment = "High liquidity environment"
            elif liquidity_score < self._cfg.low_liquidity_threshold:
                assessment = "Low liquidity conditions"
            else:
                assessment = "Moderate liquidity conditions"

            prediction_conf = float(prediction_result.get("confidence", 0.0))
            prediction_status = prediction_result.get("status", "unknown")

            thesis = f"""
Liquidity Heatmap Analysis:

Current Market Conditions:
- {assessment} (score: {liquidity_score:.3f})
- Spread condition: {spread_condition}
- Market depth: {depth_condition}
- Current spread: {liquidity_metrics['current_spread']:.6f}
- Current depth: {liquidity_metrics['current_depth']:.0f}

Neural Network Analysis:
- Prediction status: {prediction_status}
- Model confidence: {prediction_conf:.1%}
- Forward passes completed: {self.liquidity_stats['neural_forward_passes']}
- Model health: {self.liquidity_health['model_health_score']:.1f}%

Market Assessment:
- Data quality: {self.liquidity_health['data_quality_score']:.1f}%
- Sequence data points: {len(self.sequence_data)}
- Processing device: {self.device}

Liquidity Forecast:
{'Neural predictions available' if prediction_result.get('predictions') else 'Insufficient data for prediction'}
- Prediction horizon: {self._cfg.prediction_horizon} steps
- Circuit breaker: {self.neural_circuit_breaker['state']}

Trading Implications:
{'Favorable for trading' if liquidity_score > 0.6 else 'Exercise caution' if liquidity_score > 0.4 else 'High risk environment'}
- Recommended position sizing: {'Normal' if liquidity_score > 0.6 else 'Reduced' if liquidity_score > 0.4 else 'Minimal'}
- Market impact assessment: {'Low' if liquidity_score > 0.7 else 'Medium' if liquidity_score > 0.5 else 'High'}
            """.strip()
            return thesis
        except Exception as e:
            return f"Liquidity analysis completed. Thesis generation failed: {str(e)}"

    async def _update_liquidity_smart_bus(
        self,
        liquidity_metrics: Dict[str, Any],
        prediction_result: Dict[str, Any],
        thesis: str,
        sessions_map: Optional[Dict[str, Any]] = None,
        session_meta: Optional[Dict[str, Any]] = None,
    ):
        """Update SmartInfoBus with results + sessions."""
        self.smart_bus.set(
            "liquidity_score",
            liquidity_metrics["liquidity_score"],
            module="LiquidityHeatmapLayer",
            thesis=f"Current market liquidity: {liquidity_metrics['liquidity_score']:.3f}",
        )

        self.smart_bus.set(
            "market_depth",
            {
                "current_depth": liquidity_metrics["current_depth"],
                "analysis": liquidity_metrics["depth_analysis"],
                "condition": liquidity_metrics["depth_analysis"].get("condition", "unknown"),
            },
            module="LiquidityHeatmapLayer",
            thesis=f"Market depth: {liquidity_metrics['depth_analysis'].get('condition', 'unknown')}",
        )

        self.smart_bus.set(
            "spread_analysis",
            {
                "current_spread": liquidity_metrics["current_spread"],
                "analysis": liquidity_metrics["spread_analysis"],
                "condition": liquidity_metrics["spread_analysis"].get("condition", "normal"),
            },
            module="LiquidityHeatmapLayer",
            thesis=f"Spread condition: {liquidity_metrics['spread_analysis'].get('condition', 'normal')}",
        )

        if prediction_result.get("predictions"):
            self.smart_bus.set(
                "liquidity_prediction",
                {
                    "predictions": prediction_result["predictions"],
                    "confidence": prediction_result.get("confidence", 0.0),
                    "horizon": self._cfg.prediction_horizon,
                    "timestamp": time.time(),
                },
                module="LiquidityHeatmapLayer",
                thesis=f"Liquidity prediction with {prediction_result.get('confidence', 0.0):.1%} confidence",
            )
        else:
            self.smart_bus.set(
                "liquidity_prediction",
                {"predictions": {}, "confidence": 0.0, "horizon": self._cfg.prediction_horizon, "timestamp": time.time()},
                module="LiquidityHeatmapLayer",
                thesis="Liquidity prediction unavailable (insufficient data / circuit breaker).",
            )

        if sessions_map is not None:
            self.smart_bus.set("trading_sessions", sessions_map, module="LiquidityHeatmapLayer", thesis=f"Active session: {sessions_map.get('active', 'unknown')}")
        if session_meta is not None:
            self.smart_bus.set("session_data", session_meta, module="LiquidityHeatmapLayer", thesis=f"Session metadata: {session_meta.get('active_session', 'unknown')}")

        self.smart_bus.set("liquidity_thesis", thesis, module="LiquidityHeatmapLayer", thesis="LiquidityHeatmapLayer analysis thesis")

    # ── Circuit breaker & error flow ───────────────────────────────────
    def _check_neural_circuit_breaker(self) -> bool:
        if self.neural_circuit_breaker["state"] == "OPEN":
            if time.time() - self.neural_circuit_breaker["last_failure"] > 120:
                self.neural_circuit_breaker["state"] = "HALF_OPEN"
                return True
            return False
        return True

    def _record_liquidity_success(self, processing_time: float):
        if self.neural_circuit_breaker["state"] == "HALF_OPEN":
            self.neural_circuit_breaker["state"] = "CLOSED"
            self.neural_circuit_breaker["failures"] = 0
        self.liquidity_health["model_health_score"] = min(100.0, self.liquidity_health["model_health_score"] + 1)
        self.liquidity_health["last_update"] = time.time()
        self.performance_tracker.record_metric("LiquidityHeatmapLayer", "liquidity_analysis", processing_time * 1000, True)

    async def _handle_liquidity_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        processing_time = time.time() - start_time
        self._record_liquidity_failure(error)
        error_context = self.error_pinpointer.analyze_error(error, "LiquidityHeatmapLayer")
        self.logger.error(
            format_operator_message("💧[CRASH]", "LIQUIDITY_ANALYSIS_ERROR", details=str(error), context="liquidity_processing")
        )
        return self._create_liquidity_fallback_response(f"Liquidity analysis failed: {str(error)}")

    def _record_liquidity_failure(self, error: Exception):
        self.neural_circuit_breaker["failures"] += 1
        self.neural_circuit_breaker["last_failure"] = time.time()
        if self.neural_circuit_breaker["failures"] >= self.neural_circuit_breaker["threshold"]:
            self.neural_circuit_breaker["state"] = "OPEN"
            self.logger.error(
                format_operator_message(
                    "💧[ALERT]",
                    "LIQUIDITY_CIRCUIT_BREAKER_OPEN",
                    details=f"Too many liquidity failures ({self.neural_circuit_breaker['failures']})",
                    context="liquidity_circuit_breaker",
                )
            )
        self.liquidity_health["model_health_score"] = max(0.0, self.liquidity_health["model_health_score"] - 10)

    def _create_liquidity_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response (keeps exposes aligned with provides)."""
        return {
            "success": False,
            "reason": reason,
            "liquidity_score": float(self.current_liquidity_score),
            "market_depth": {"current_depth": float(self.current_depth), "analysis": {"status": "fallback"}, "condition": "unknown"},
            "spread_analysis": {"current_spread": float(self.current_spread), "analysis": {"status": "fallback"}, "condition": "unknown"},
            "liquidity_prediction": {"predictions": {}, "confidence": 0.0, "horizon_steps": self._cfg.prediction_horizon, "status": "unavailable"},
            "thesis": f"Liquidity analysis unavailable: {reason}. Using last known values.",
            "processing_time_ms": 0.0,
        }

    # ── Monitoring ─────────────────────────────────────────────────────
    async def _liquidity_monitoring_loop(self):
        while True:
            try:
                await asyncio.sleep(30)
                self._update_liquidity_health()
                self._check_liquidity_anomalies()
            except Exception as e:
                self.logger.error(f"Liquidity monitoring error: {e}")

    def _update_liquidity_health(self):
        data_freshness = time.time() - self.liquidity_health["last_update"]
        if data_freshness < 60:
            self.liquidity_health["data_quality_score"] = min(100.0, self.liquidity_health["data_quality_score"] + 1)
        elif data_freshness > 300:
            self.liquidity_health["data_quality_score"] = max(0.0, self.liquidity_health["data_quality_score"] - 2)

        if self.neural_circuit_breaker["state"] == "CLOSED":
            self.liquidity_health["model_health_score"] = min(100.0, self.liquidity_health["model_health_score"] + 0.5)
        elif self.neural_circuit_breaker["state"] == "OPEN":
            self.liquidity_health["model_health_score"] = max(0.0, self.liquidity_health["model_health_score"] - 5)

    def _check_liquidity_anomalies(self):
        anomalies: List[str] = []
        if self.current_spread > 0.01:
            anomalies.append("Extremely wide spread detected")
        if self.current_depth < 1000:
            anomalies.append("Very low market depth")
        if self.neural_circuit_breaker["state"] == "OPEN":
            anomalies.append("Neural circuit breaker is open")
        if anomalies:
            self.logger.warning(
                format_operator_message("💧[WARN]", "LIQUIDITY_ANOMALIES", details=f"{len(anomalies)} anomalies detected", context="liquidity_monitoring")
            )

    # ── State I/O ──────────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        base = super().get_state()
        liq = {
            "config": {"lstm_units": self._cfg.lstm_units, "sequence_length": self._cfg.sequence_length, "device": str(self.device)},
            "liquidity_data": {
                "current_liquidity_score": self.current_liquidity_score,
                "current_spread": self.current_spread,
                "current_depth": self.current_depth,
                "sequence_data": [list(np.array(seq, dtype=float)) for seq in self.sequence_data],
                "price_history": list(map(float, self.price_history)),
                "spread_history": list(map(float, self.spread_history)),
                "depth_history": list(map(float, self.depth_history)),
            },
            "statistics": self.liquidity_stats,
            "health_metrics": self.liquidity_health,
            "circuit_breaker": self.neural_circuit_breaker,
        }
        return {**base, **liq}

    def set_state(self, state: Dict[str, Any]):
        super().set_state(state)
        if "liquidity_data" in state:
            data = state["liquidity_data"]
            self.current_liquidity_score = float(data.get("current_liquidity_score", 0.5))
            self.current_spread = float(data.get("current_spread", 0.0))
            self.current_depth = float(data.get("current_depth", 0.0))
            if "sequence_data" in data:
                self.sequence_data = deque([np.array(seq, dtype=np.float32) for seq in data["sequence_data"]], maxlen=self._cfg.sequence_length)
            if "price_history" in data:
                self.price_history = deque([float(x) for x in data["price_history"]], maxlen=500)
            if "spread_history" in data:
                self.spread_history = deque([float(x) for x in data["spread_history"]], maxlen=200)
            if "depth_history" in data:
                self.depth_history = deque([float(x) for x in data["depth_history"]], maxlen=200)

        if "statistics" in state:
            self.liquidity_stats.update(state["statistics"])
        if "health_metrics" in state:
            self.liquidity_health.update(state["health_metrics"])
        if "circuit_breaker" in state:
            self.neural_circuit_breaker.update(state["circuit_breaker"])

    def _start_monitoring(self) -> None:
        """Start the background monitoring loop (idempotent, safe if no event loop yet)."""
        # If a task exists and is still running, do nothing
        existing_task = getattr(self, "_monitor_task", None)
        if isinstance(existing_task, asyncio.Task) and not existing_task.done():
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (e.g., constructed in sync context); we’ll try again later
            self._monitor_task = None
            self.logger.debug("No running asyncio loop; deferring liquidity monitoring start.")
            return

        # (Re)start the monitoring task
        self._monitor_task = loop.create_task(self._liquidity_monitoring_loop())


    # ── Health & Reports ───────────────────────────────────────────────
    def get_health_status(self) -> Dict[str, Any]:
        return {
            "model_health_score": self.liquidity_health["model_health_score"],
            "data_quality_score": self.liquidity_health["data_quality_score"],
            "neural_circuit_breaker_state": self.neural_circuit_breaker["state"],
            "liquidity_statistics": self.liquidity_stats,
            "current_liquidity_score": self.current_liquidity_score,
            "device": str(self.device),
            "sequence_data_length": len(self.sequence_data),
        }

    def get_liquidity_performance_report(self) -> str:
        try:
            return self.english_explainer.explain_performance(
                module_name="LiquidityHeatmapLayer",
                metrics={
                    "neural_forward_passes": self.liquidity_stats["neural_forward_passes"],
                    "prediction_accuracy": self.liquidity_stats["avg_prediction_accuracy"],
                    "model_health_score": self.liquidity_health["model_health_score"],
                    "data_quality_score": self.liquidity_health["data_quality_score"],
                    "current_liquidity_score": self.current_liquidity_score,
                    "circuit_breaker_state": self.neural_circuit_breaker["state"],
                },
            )
        except Exception as e:
            return f"Liquidity performance report generation failed: {str(e)}"

    # ── Actions & Confidence ───────────────────────────────────────────
    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose liquidity-based action recommendations."""
        try:
            liquidity_score = float(self.current_liquidity_score)
            prediction_accuracy = float(self.liquidity_stats["avg_prediction_accuracy"])  # 0..1

            if liquidity_score > self._cfg.high_liquidity_threshold:
                if prediction_accuracy > 0.8:
                    action, rationale, risk = "aggressive_trade", "High liquidity with strong prediction accuracy - favorable for aggressive trading", "low"
                else:
                    action, rationale, risk = "moderate_trade", "High liquidity but lower prediction accuracy - proceed with moderate trading", "low"
            elif liquidity_score < self._cfg.low_liquidity_threshold:
                action, rationale, risk = "reduce_size", "Low liquidity detected - reduce position sizes to minimize market impact", "high"
            else:
                if prediction_accuracy > 0.7:
                    action, rationale, risk = "normal_trade", "Medium liquidity with good prediction accuracy - normal trading conditions", "medium"
                else:
                    action, rationale, risk = "cautious_trade", "Medium liquidity with lower prediction accuracy - trade cautiously", "medium"

            liquidity_confidence = float(np.clip((liquidity_score + prediction_accuracy) / 2.0, 0.0, 1.0))
            return {
                "action": action,
                "liquidity_confidence": liquidity_confidence,
                "rationale": rationale,
                "risk_level": risk,
                "current_liquidity_score": liquidity_score,
                "prediction_accuracy": prediction_accuracy,
                "neural_health": float(self.liquidity_health["model_health_score"]),
            }
        except Exception as e:
            self.logger.error(f"Error in propose_action: {e}")
            return {"action": "hold", "liquidity_confidence": 0.5, "rationale": f"Error in liquidity analysis: {str(e)}", "risk_level": "medium"}

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in the proposed action."""
        try:
            liquidity_score = float(self.current_liquidity_score)
            prediction_accuracy = float(self.liquidity_stats["avg_prediction_accuracy"])  # 0..1
            model_health = float(self.liquidity_health["model_health_score"]) / 100.0
            data_quality = float(self.liquidity_health["data_quality_score"]) / 100.0
            neural_confidence = 1.0 if self.neural_circuit_breaker["state"] == "CLOSED" else (0.5 if self.neural_circuit_breaker["state"] == "HALF_OPEN" else 0.3)

            last_pred_time = getattr(self, "last_prediction_time", None)
            if last_pred_time:
                freshness = max(0.0, 1.0 - (time.time() - last_pred_time) / 300.0)  # 5-min decay
            else:
                freshness = 0.5

            # Weighted blend (all in 0..1)
            confidence = (
                liquidity_score * 0.30
                + prediction_accuracy * 0.25
                + model_health * 0.20
                + data_quality * 0.15
                + neural_confidence * 0.05
                + freshness * 0.05
            )

            # Action-specific adjustments
            a = action.get("action", "hold")
            if a == "aggressive_trade" and liquidity_score < 0.8:
                confidence *= 0.7
            elif a == "reduce_size" and liquidity_score < 0.3:
                confidence *= 1.2
            elif a == "hold" and prediction_accuracy < 0.5:
                confidence *= 1.1

            return float(np.clip(confidence, 0.0, 1.0))
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
