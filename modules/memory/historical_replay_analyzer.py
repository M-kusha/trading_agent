# ─────────────────────────────────────────────────────────────
# File: modules/memory/historical_replay_analyzer.py
# [ROCKET] PRODUCTION-READY Historical Replay Analysis System
# Advanced sequence analysis with SmartInfoBus integration
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
import threading
import math
from modules.contracts import module_args
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Deque, DefaultDict, Callable, cast
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from modules.core.module_base import BaseModule, module
from modules.core.mixins import (
    SmartInfoBusTradingMixin,
    SmartInfoBusRiskMixin,
    SmartInfoBusStateMixin,
)
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager, SmartInfoBus
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ReplayConfig:
    """Configuration for Historical Replay Analyzer"""
    # cadence / algorithm
    interval: int = 10
    bonus: float = 0.1
    sequence_len: int = 5
    profit_threshold: float = 10.0
    pattern_sensitivity: float = 1.0
    replay_decay: float = 0.9  # per-hour decay base (applied as base ** hours_since_event)

    # performance / safety
    max_processing_time_ms: float = 200
    circuit_breaker_threshold: int = 3
    min_sequence_quality: float = 0.6

    # analysis parameters
    max_sequences: int = 100
    lookback_episodes: int = 50
    pattern_confidence_threshold: float = 0.7

    # integration
    namespace: Optional[str] = None  # keep default None to remain backward-compatible with readers


# ═══════════════════════════════════════════════════════════════════
# MODULE
# ═══════════════════════════════════════════════════════════════════

@module(**module_args(
    "HistoricalReplayAnalyzer",
    description="Deterministic multi-window feature extraction with circuit breaker, monitoring, and explainability.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=120,
))
class HistoricalReplayAnalyzer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced historical replay analyzer with SmartInfoBus integration.
    Identifies profitable trading sequences and patterns for learning optimization.
    """

    # ──────────────────────────────────────────────────────────────
    # Construction
    # ──────────────────────────────────────────────────────────────
    def __init__(
        self,
        config: Optional[Dict[str, Any] | ReplayConfig] = None,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        # Normalize config to dataclass and also provide dict to BaseModule
        if isinstance(config, dict):
            filtered = {k: config[k] for k in ReplayConfig.__dataclass_fields__ if k in config}
            self._cfg: ReplayConfig = ReplayConfig(**filtered)
        elif isinstance(config, ReplayConfig) or config is None:
            self._cfg = config or ReplayConfig()
        else:
            self._cfg = ReplayConfig()

        # EARLY INITIALIZATION: objects referenced by BaseModule._initialize
        self.smart_bus: SmartInfoBus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="HistoricalReplayAnalyzer",
            log_path="logs/memory/replay_analysis.log",
            max_lines=3000,
            operator_mode=True,
            plain_english=True,
        )

        # Now initialize BaseModule (may call self._initialize). Pass dict config.
        from dataclasses import asdict
        super().__init__(config=asdict(self._cfg), **kwargs)

        # systems
        self._initialize_advanced_systems()
        self._initialize_genome_parameters(genome)
        self._initialize_replay_state()
        self._register_bus_validators()
        self._subscribe_bus_events()
        self._start_monitoring()

        self.logger.info(
            format_operator_message(
                "🎭",
                "HISTORICAL_REPLAY_ANALYZER_INITIALIZED",
                details=f"Sequence length: {self._cfg.sequence_len}, Profit threshold: {self._cfg.profit_threshold}",
                result="Historical pattern analysis ready",
                context="memory_analysis",
            )
        )

    # Small helper to support both dict and dataclass configs for namespace
    def _get_namespace(self) -> Optional[str]:
        try:
            # Prefer typed config; stay defensive if external code replaced it with dict
            if hasattr(self, "_cfg") and getattr(self._cfg, "namespace", None) is not None:
                return self._cfg.namespace
            cfg = getattr(self, "config", None)
            if isinstance(cfg, dict):
                return cfg.get("namespace")
            return getattr(cfg, "namespace", None)
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────
    # Systems
    # ──────────────────────────────────────────────────────────────
    def _initialize_advanced_systems(self) -> None:
        """Initialize advanced systems for replay analysis"""
        self.smart_bus: SmartInfoBus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="HistoricalReplayAnalyzer",
            log_path="logs/memory/replay_analysis.log",
            max_lines=3000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("HistoricalReplayAnalyzer", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Circuit breaker (simple, module-specific; we also report to SmartInfoBus breaker)
        self.circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",
            "threshold": int(getattr(self._cfg, "circuit_breaker_threshold", 3)),
        }

        # health state
        self._health_status: str = "healthy"
        self._last_health_check: float = time.time()

        # advertise capabilities (optional redundancy with @module)
        try:
            InfoBusManager.register_module_capabilities(
                "HistoricalReplayAnalyzer",
                provides=["replay_sequences", "pattern_analysis", "sequence_quality", "learning_progress"],
                requires=["trades", "actions", "market_data", "episode_data"],
            )
        except Exception:
            # never crash init if registration fails
            pass

    def _register_bus_validators(self) -> None:
        """Schema validators so downstream readers get guaranteed shapes."""
        def _is_float(x: Any) -> bool:
            return isinstance(x, (float, int)) and not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))

        def validate_replay_sequences(v: Any) -> bool:
            if not isinstance(v, dict): return False
            return (
                isinstance(v.get("total_sequences", 0), int) and
                _is_float(v.get("best_sequence_pnl", 0.0)) and
                _is_float(v.get("replay_bonus", 0.0)) and
                isinstance(v.get("sequences_analyzed", 0), int)
            )

        def validate_pattern_analysis(v: Any) -> bool:
            if not isinstance(v, dict): return False
            best_ok = (v.get("best_pattern") is None) or isinstance(v.get("best_pattern"), str)
            return (
                isinstance(v.get("total_patterns", 0), int) and
                isinstance(v.get("profitable_patterns", 0), int) and
                best_ok and
                _is_float(v.get("pattern_confidence_avg", 0.0))
            )

        def validate_sequence_quality(v: Any) -> bool:
            if not isinstance(v, dict): return False
            return (
                _is_float(v.get("current_quality", 0.0)) and
                _is_float(v.get("average_quality", 0.0)) and
                isinstance(v.get("quality_trend", "insufficient_data"), str) and
                isinstance(v.get("episodes_processed", 0), int)
            )

        def validate_learning_progress(v: Any) -> bool:
            if not isinstance(v, dict): return False
            return (
                isinstance(v.get("profitable_sequences_count", 0), int) and
                isinstance(v.get("total_episodes", 0), int) and
                _is_float(v.get("success_rate", 0.0)) and
                _is_float(v.get("learning_acceleration", 0.0))
            )

        # Register validators on both namespaced and non-namespaced keys for flexibility
        try:
            self.smart_bus.register_validator("replay_sequences", validate_replay_sequences)
            self.smart_bus.register_validator("pattern_analysis", validate_pattern_analysis)
            self.smart_bus.register_validator("sequence_quality", validate_sequence_quality)
            self.smart_bus.register_validator("learning_progress", validate_learning_progress)

            ns = self._get_namespace()
            if ns:
                self.smart_bus.register_validator(f"{ns}:replay_sequences", validate_replay_sequences)
                self.smart_bus.register_validator(f"{ns}:pattern_analysis", validate_pattern_analysis)
                self.smart_bus.register_validator(f"{ns}:sequence_quality", validate_sequence_quality)
                self.smart_bus.register_validator(f"{ns}:learning_progress", validate_learning_progress)
        except Exception:
            # validators are helpful but non-critical
            pass

    def _subscribe_bus_events(self) -> None:
        """Optional: observe SmartInfoBus events to enrich heuristics without coupling."""
        def on_performance_warning(evt: Dict[str, Any]) -> None:
            try:
                if evt.get("module") == "HistoricalReplayAnalyzer":
                    # could adapt thresholds dynamically; keep it simple for now
                    pass
            except Exception:
                pass

        try:
            self.smart_bus.subscribe("performance_warning", on_performance_warning)
        except Exception:
            pass

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]) -> None:
        """Initialize genome-based parameters"""
        if genome:
            self.genome: Dict[str, Any] = {
                "interval": int(genome.get("interval", self._cfg.interval)),
                "bonus": float(genome.get("bonus", self._cfg.bonus)),
                "sequence_len": int(genome.get("sequence_len", self._cfg.sequence_len)),
                "profit_threshold": float(genome.get("profit_threshold", self._cfg.profit_threshold)),
                "pattern_sensitivity": float(genome.get("pattern_sensitivity", self._cfg.pattern_sensitivity)),
                "replay_decay": float(genome.get("replay_decay", self._cfg.replay_decay)),
            }
        else:
            self.genome = {
                "interval": self._cfg.interval,
                "bonus": self._cfg.bonus,
                "sequence_len": self._cfg.sequence_len,
                "profit_threshold": self._cfg.profit_threshold,
                "pattern_sensitivity": self._cfg.pattern_sensitivity,
                "replay_decay": self._cfg.replay_decay,
            }

    def _initialize_replay_state(self) -> None:
        """Initialize replay analysis state"""
        ns_max = int(getattr(self._cfg, "lookback_episodes", 50))

        # Core replay data
        self.episode_buffer: Deque[Dict[str, Any]] = deque(maxlen=ns_max)
        self.profitable_sequences: List[Dict[str, Any]] = []
        self.sequence_patterns: Dict[str, Dict[str, Any]] = {}  # pattern -> stats
        self.current_sequence: List[Dict[str, Any]] = []
        self.replay_bonus: float = 0.0
        self.best_sequence_pnl: float = 0.0

        # Enhanced tracking
        self._episode_count: int = 0
        self._pattern_evolution: Deque[Dict[str, Any]] = deque(maxlen=50)
        self._sequence_quality_scores: Deque[float] = deque(maxlen=100)
        self._replay_effectiveness: Dict[int, Any] = {}
        self._learning_curve: Deque[float] = deque(maxlen=200)

        # Pattern analysis aids
        self._pattern_success_rates: Dict[str, float] = {}
        self._pattern_market_conditions: Dict[str, Dict[str, Any]] = {}
        self._adaptive_thresholds: Dict[str, Any] = {
            "min_profit": self.genome["profit_threshold"],
            "min_sequence_len": 3,
            "pattern_confidence": float(getattr(self._cfg, "pattern_confidence_threshold", 0.7)),
        }

        # Performance analytics
        self._analysis_performance: Dict[str, Any] = {
            "sequences_analyzed": 0,
            "patterns_identified": 0,
            "successful_replays": 0,
            "total_replay_bonus": 0.0,
        }

    def _start_monitoring(self) -> None:
        """Start background monitoring (lightweight)"""
        def monitoring_loop() -> None:
            while getattr(self, "_monitoring_active", True):
                try:
                    self._update_replay_health()
                    self._analyze_pattern_effectiveness()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")

        self._monitoring_active: bool = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True, name="ReplayAnalyzer-Monitor")
        monitor_thread.start()

    # ──────────────────────────────────────────────────────────────
    # BaseModule entrypoint
    # ──────────────────────────────────────────────────────────────
    def _initialize(self) -> None:
        """Initialize module (called by BaseModule)"""
        try:
            initial_status = {
                "sequences_stored": 0,
                "patterns_identified": 0,
                "best_sequence_pnl": 0.0,
                "replay_bonus": 0.0,
            }
            self.smart_bus.set(
                "replay_sequences",
                initial_status,
                module="HistoricalReplayAnalyzer",
                thesis="Initial historical replay analysis status",
                namespace=self._get_namespace(),
            )
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    # ──────────────────────────────────────────────────────────────
    # Processing
    # ──────────────────────────────────────────────────────────────
    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """Process historical replay analysis"""
        t0 = time.time()

        try:
            sequence_data = await self._extract_sequence_data(**inputs)
            if not sequence_data:
                out = await self._handle_no_data_fallback()
                self._record_success((time.time() - t0) * 1000.0)
                # still publish bus data (best-effort)
                await self._update_replay_smart_bus(out, out.get("_thesis", "fallback"))
                return out

            sequence_result = await self._process_trading_sequence(sequence_data)

            # Episode completion → analyze patterns
            if sequence_data.get("episode_completed", False):
                pattern_result = await self._analyze_episode_patterns(sequence_data)
                sequence_result.update(pattern_result)

            # Replay suggestions
            replay_result = await self._generate_replay_recommendations()
            sequence_result.update(replay_result)

            # Thesis
            thesis = await self._generate_replay_thesis(sequence_data, sequence_result)

            # Update SmartInfoBus
            await self._update_replay_smart_bus(sequence_result, thesis)

            # success accounting
            processing_time_ms = (time.time() - t0) * 1000.0
            self._record_success(processing_time_ms)
            try:
                # feed metrics to SmartInfoBus
                self.smart_bus.record_module_timing("HistoricalReplayAnalyzer", processing_time_ms)
            except Exception:
                pass

            # always return contract payload
            replay_data = {
                "total_sequences": len(self.profitable_sequences),
                "best_sequence_pnl": self.best_sequence_pnl,
                "replay_bonus": self.replay_bonus,
                "sequences_analyzed": self._analysis_performance["sequences_analyzed"],
            }
            pattern_summary = {
                "total_patterns": len(self.sequence_patterns),
                "profitable_patterns": sum(1 for p in self.sequence_patterns.values() if p.get("avg_pnl", 0.0) > 0.0),
                "best_pattern": (
                    max(self.sequence_patterns.items(), key=lambda x: x[1].get("avg_pnl", 0.0))[0]
                    if self.sequence_patterns else None
                ),
                "pattern_confidence_avg": float(np.mean([p.get("confidence", 0.0) for p in self.sequence_patterns.values()]))
                if self.sequence_patterns else 0.0,
            }
            quality_metrics = {
                "current_quality": sequence_result.get("sequence_quality", 0.0),
                "average_quality": float(np.mean(list(self._sequence_quality_scores))) if self._sequence_quality_scores else 0.0,
                "quality_trend": self._calculate_quality_trend(),
                "episodes_processed": self._episode_count,
            }
            learning_metrics = {
                "profitable_sequences_count": len(self.profitable_sequences),
                "total_episodes": self._episode_count,
                "success_rate": len(self.profitable_sequences) / max(self._episode_count, 1),
                "learning_acceleration": self._calculate_learning_acceleration(),
            }
            sequence_result.update(
                {
                    "replay_sequences": replay_data,
                    "pattern_analysis": pattern_summary,
                    "sequence_quality": quality_metrics,
                    "learning_progress": learning_metrics,
                    "_thesis": thesis,
                }
            )
            return sequence_result

        except Exception as e:
            # report into bus circuit breaker as well
            try:
                self.smart_bus.record_module_failure("HistoricalReplayAnalyzer", str(e))
            except Exception:
                pass
            return await self._handle_replay_error(e, t0)

    async def _extract_sequence_data(self, **inputs: Any) -> Optional[Dict[str, Any]]:
        """Extract sequence data from SmartInfoBus"""
        try:
            ns = self._get_namespace()
            trades = self.smart_bus.get("trades", "HistoricalReplayAnalyzer", default=[], namespace=ns) or []
            actions = self.smart_bus.get("actions", "HistoricalReplayAnalyzer", default=[], namespace=ns) or []
            market_data = self.smart_bus.get("market_data", "HistoricalReplayAnalyzer", default={}, namespace=ns) or {}
            episode_data = self.smart_bus.get("episode_data", "HistoricalReplayAnalyzer", default={}, namespace=ns) or {}

            # Current action (default shape)
            current_action = inputs.get("action")
            if current_action is None:
                current_action = np.zeros(2)
            # normalize to a plain list (avoid ndarray surprises)
            try:
                current_action = np.asarray(current_action, dtype=float).tolist()
            except Exception:
                current_action = [0.0, 0.0]

            return {
                "trades": trades,
                "actions": actions,
                "market_data": market_data,
                "episode_data": episode_data,
                "current_action": current_action,
                "timestamp": datetime.now().isoformat(),
                "episode_completed": bool(inputs.get("episode_completed", False)),
            }
        except Exception as e:
            self.logger.error(f"Failed to extract sequence data: {e}")
            return None

    async def _process_trading_sequence(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process current trading sequence"""
        try:
            # Append current step
            if sequence_data.get("current_action") is not None:
                current_step = {
                    "action": sequence_data["current_action"],
                    "timestamp": time.time(),
                    "market_context": sequence_data.get("market_data", {}),
                }
                self.current_sequence.append(current_step)

            # Trim sequence
            max_len = int(max(1, self.genome["sequence_len"]))
            if len(self.current_sequence) > max_len:
                self.current_sequence = self.current_sequence[-max_len:]

            # Quality
            sequence_quality = self._calculate_sequence_quality(self.current_sequence)
            self._sequence_quality_scores.append(float(sequence_quality))

            # accounting
            self._analysis_performance["sequences_analyzed"] += 1

            return {
                "current_sequence_length": len(self.current_sequence),
                "sequence_quality": float(sequence_quality),
                "sequences_processed": self._analysis_performance["sequences_analyzed"],
            }
        except Exception as e:
            self.logger.error(f"Sequence processing failed: {e}")
            return self._create_fallback_response("sequence processing failed")

    def _calculate_sequence_quality(self, sequence: List[Dict[str, Any]]) -> float:
        """Calculate quality score of a trading sequence (0..1)"""
        if not sequence or len(sequence) < 2:
            return 0.0
        try:
            # Action magnitude variance → consistency
            actions: List[List[float]] = [cast(List[float], step.get("action", [0.0, 0.0])) for step in sequence]
            magnitudes: List[float] = []
            for a in actions:
                try:
                    magnitudes.append(float(math.sqrt(sum(float(x) ** 2 for x in a))))
                except Exception:
                    magnitudes.append(0.0)
            action_variance = float(np.var(magnitudes)) if magnitudes else 0.0
            consistency_score = max(0.0, 1.0 - action_variance)  # lower variance → higher consistency

            # Temporal spacing regularity
            timestamps = [float(step.get("timestamp", 0.0)) for step in sequence]
            if len(timestamps) >= 2:
                diffs = np.diff(timestamps)
                temporal_score = max(0.0, 1.0 - float(np.std(diffs)) / 100.0)
            else:
                temporal_score = 1.0

            # Market context proxy (if volatility provided, prefer moderate regime)
            context_score = 0.8
            try:
                vols = []
                for step in sequence:
                    ctx = step.get("market_context") or {}
                    v = ctx.get("volatility")
                    if isinstance(v, (float, int)) and not np.isnan(v):
                        vols.append(float(v))
                if vols:
                    # reward moderate volatility; penalize extremes
                    v_std = float(np.std(vols))
                    context_score = max(0.2, 1.0 - min(v_std / 5.0, 0.8))
            except Exception:
                pass

            quality = (consistency_score * 0.4) + (temporal_score * 0.3) + (context_score * 0.3)
            return float(max(0.0, min(1.0, quality)))
        except Exception as e:
            self.logger.error(f"Quality calculation failed: {e}")
            return 0.5

    async def _analyze_episode_patterns(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in completed episode"""
        try:
            episode_data = sequence_data.get("episode_data", {}) or {}
            episode_pnl = float(episode_data.get("pnl", 0.0))

            # Buffer
            episode_record = {
                "sequence": list(self.current_sequence),
                "pnl": episode_pnl,
                "timestamp": time.time(),
                "market_conditions": sequence_data.get("market_data", {}) or {},
            }
            self.episode_buffer.append(episode_record)
            self._episode_count += 1

            # Profitable → deeper analysis
            if episode_pnl > float(self.genome["profit_threshold"]):
                await self._analyze_profitable_sequence(episode_record)

            # Pattern extraction
            pattern = self._extract_sequence_pattern(self.current_sequence)
            if pattern:
                self._update_pattern_data(pattern, episode_pnl)

            # reset current sequence
            self.current_sequence = []

            return {
                "episode_analyzed": True,
                "episode_pnl": episode_pnl,
                "pattern_extracted": pattern is not None,
                "profitable_sequence": episode_pnl > float(self.genome["profit_threshold"]),
            }
        except Exception as e:
            self.logger.error(f"Episode pattern analysis failed: {e}")
            return {"episode_analyzed": False, "error": str(e)}

    async def _analyze_profitable_sequence(self, episode_record: Dict[str, Any]) -> None:
        """Analyze profitable sequence for learning"""
        try:
            sequence = cast(List[Dict[str, Any]], episode_record["sequence"])
            pnl = float(episode_record["pnl"])

            entry = {
                "sequence": sequence,
                "pnl": pnl,
                "timestamp": float(episode_record["timestamp"]),
                "market_conditions": episode_record["market_conditions"],
                "quality_score": float(self._calculate_sequence_quality(sequence)),
            }
            self.profitable_sequences.append(entry)

            # Top-K pruning
            if len(self.profitable_sequences) > int(self._cfg.max_sequences):
                self.profitable_sequences.sort(key=lambda x: float(x.get("pnl", 0.0)), reverse=True)
                self.profitable_sequences = self.profitable_sequences[: int(self._cfg.max_sequences)]

            if pnl > self.best_sequence_pnl:
                self.best_sequence_pnl = pnl

            self.logger.info(
                format_operator_message(
                    "[MONEY]",
                    "PROFITABLE_SEQUENCE_IDENTIFIED",
                    pnl=f"{pnl:.2f}",
                    sequence_length=len(sequence),
                    quality_score=f"{entry['quality_score']:.3f}",
                    context="pattern_learning",
                )
            )
        except Exception as e:
            self.logger.error(f"Profitable sequence analysis failed: {e}")

    def _extract_sequence_pattern(self, sequence: List[Dict[str, Any]]) -> Optional[str]:
        """Extract discrete pattern signature like H/L/S from first action component"""
        if not sequence or len(sequence) < 3:
            return None
        try:
            directions: List[str] = []
            for step in sequence:
                action = step.get("action", [0.0, 0.0])
                a0 = 0.0
                try:
                    a0 = float(action[0]) if isinstance(action, (list, tuple)) and len(action) >= 1 else 0.0
                except Exception:
                    a0 = 0.0
                if a0 > 0.1:
                    directions.append("L")
                elif a0 < -0.1:
                    directions.append("S")
                else:
                    directions.append("H")

            if len(directions) >= 3:
                return "".join(directions)
            return None
        except Exception as e:
            self.logger.error(f"Pattern extraction failed: {e}")
            return None

    def _update_pattern_data(self, pattern: str, pnl: float) -> None:
        """Update pattern tracking data"""
        try:
            if pattern not in self.sequence_patterns:
                self.sequence_patterns[pattern] = {
                    "count": 0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "last_seen": time.time(),
                    "confidence": 0.0,
                }

            pd = self.sequence_patterns[pattern]
            pd["count"] = int(pd["count"]) + 1
            pd["total_pnl"] = float(pd["total_pnl"]) + float(pnl)
            pd["avg_pnl"] = float(pd["total_pnl"]) / max(pd["count"], 1)
            pd["last_seen"] = time.time()

            # confidence grows with count; adjust by consistency
            confidence = min(1.0, pd["count"] / 10.0)
            if pd["count"] > 1:
                # relative deviation penalty
                denom = max(abs(pd["avg_pnl"]), 1.0)
                consistency = 1.0 - abs(float(pnl) - float(pd["avg_pnl"])) / denom
                confidence *= max(0.0, min(1.0, consistency))
            pd["confidence"] = float(confidence)

            self._pattern_evolution.append(
                {
                    "pattern": pattern,
                    "pnl": float(pnl),
                    "avg_pnl": float(pd["avg_pnl"]),
                    "confidence": float(confidence),
                    "timestamp": time.time(),
                }
            )

            # maintain counter for quick stats
            self._analysis_performance["patterns_identified"] = len(self.sequence_patterns)
        except Exception as e:
            self.logger.error(f"Pattern data update failed: {e}")

    async def _generate_replay_recommendations(self) -> Dict[str, Any]:
        """Generate replay recommendations based on patterns"""
        try:
            # decide cadence; if interval <= 0, never trigger automatically
            interval = int(self.genome.get("interval", 0))
            should_replay = interval > 0 and (self._episode_count > 0) and (self._episode_count % interval == 0)

            if not should_replay or not self.profitable_sequences:
                return {
                    "replay_recommended": False,
                    "replay_bonus": 0.0,
                    "best_pattern": None,
                }

            # Best sequence by PnL
            best_sequence = max(self.profitable_sequences, key=lambda x: float(x.get("pnl", 0.0)))

            # Time-decayed bonus: base ** hours_since(best) * scaled PnL
            hours_since = max(0.0, (time.time() - float(best_sequence.get("timestamp", time.time()))) / 3600.0)
            decay_base = float(self.genome["replay_decay"])
            decay_factor = decay_base ** hours_since  # if base<1, naturally decays with time
            pnl_scaled = float(best_sequence.get("pnl", 0.0)) / 100.0
            replay_bonus = float(self.genome["bonus"]) * pnl_scaled * decay_factor
            self.replay_bonus = float(replay_bonus)

            # Best pattern by (avg_pnl * confidence)
            best_pattern: Optional[str] = None
            if self.sequence_patterns:
                best_pattern = max(
                    self.sequence_patterns.items(),
                    key=lambda kv: float(kv[1].get("avg_pnl", 0.0)) * float(kv[1].get("confidence", 0.0)),
                )[0]

            self._analysis_performance["total_replay_bonus"] = float(
                self._analysis_performance.get("total_replay_bonus", 0.0)
            ) + float(replay_bonus)
            self._analysis_performance["successful_replays"] = int(
                self._analysis_performance.get("successful_replays", 0)
            ) + 1

            return {
                "replay_recommended": True,
                "replay_bonus": float(replay_bonus),
                "best_sequence_pnl": float(best_sequence.get("pnl", 0.0)),
                "best_pattern": best_pattern,
                "total_patterns": len(self.sequence_patterns),
            }
        except Exception as e:
            self.logger.error(f"Replay recommendation generation failed: {e}")
            return {
                "replay_recommended": False,
                "replay_bonus": 0.0,
                "error": str(e),
            }

    async def _generate_replay_thesis(self, sequence_data: Dict[str, Any], replay_result: Dict[str, Any]) -> str:
        """Generate comprehensive replay analysis thesis"""
        try:
            sequences_analyzed = int(self._analysis_performance["sequences_analyzed"])
            total_patterns = int(len(self.sequence_patterns))
            profitable_patterns = int(sum(1 for p in self.sequence_patterns.values() if p.get("avg_pnl", 0.0) > 0.0))
            best_pnl = float(self.best_sequence_pnl)

            replay_recommended = bool(replay_result.get("replay_recommended", False))
            replay_bonus = float(replay_result.get("replay_bonus", 0.0))

            thesis_parts: List[str] = [
                f"Historical Replay Analysis: Processed {sequences_analyzed} sequences across {self._episode_count} episodes",
                f"Pattern Recognition: Identified {total_patterns} patterns, {profitable_patterns} profitable",
                f"Best sequence performance: {best_pnl:.2f} PnL (quality-weighted)",
                f"Learning optimization: {len(self.profitable_sequences)} profitable sequences stored",
            ]

            if replay_recommended:
                thesis_parts.append(f"Replay triggered with bonus {replay_bonus:.4f} based on decayed PnL")
                best_pattern = replay_result.get("best_pattern")
                if isinstance(best_pattern, str) and best_pattern in self.sequence_patterns:
                    pd = self.sequence_patterns[best_pattern]
                    thesis_parts.append(
                        f"Best pattern '{best_pattern}': avg PnL {pd['avg_pnl']:.2f}, confidence {pd['confidence']:.2f}"
                    )
            else:
                thesis_parts.append("No replay this cycle — accumulating patterns")

            if self._sequence_quality_scores:
                avg_q = float(np.mean(list(self._sequence_quality_scores)[-20:]))
                thesis_parts.append(f"Recent sequence quality: {avg_q:.2f} (target ≥ {self._cfg.min_sequence_quality})")

            if len(self.profitable_sequences) > 5:
                recent = float(np.mean([float(s.get("pnl", 0.0)) for s in self.profitable_sequences[-5:]]))
                thesis_parts.append(f"Learning trend: recent avg PnL {recent:.2f}")

            return " | ".join(thesis_parts)
        except Exception as e:
            return f"Replay thesis generation failed: {str(e)} — continuing with basic metrics"

    async def _update_replay_smart_bus(self, replay_result: Dict[str, Any], thesis: str) -> None:
        """Update SmartInfoBus with replay analysis results"""
        try:
            ns = self._get_namespace()

            # Replay sequences
            replay_data = {
                "total_sequences": len(self.profitable_sequences),
                "best_sequence_pnl": float(self.best_sequence_pnl),
                "replay_bonus": float(self.replay_bonus),
                "sequences_analyzed": int(self._analysis_performance["sequences_analyzed"]),
            }
            self.smart_bus.set(
                "replay_sequences",
                replay_data,
                module="HistoricalReplayAnalyzer",
                thesis=thesis,
                namespace=ns,
            )

            # Pattern analysis
            pattern_summary = {
                "total_patterns": len(self.sequence_patterns),
                "profitable_patterns": sum(1 for p in self.sequence_patterns.values() if p.get("avg_pnl", 0.0) > 0.0),
                "best_pattern": (
                    max(self.sequence_patterns.items(), key=lambda x: float(x[1].get("avg_pnl", 0.0)))[0]
                    if self.sequence_patterns
                    else None
                ),
                "pattern_confidence_avg": float(
                    np.mean([float(p.get("confidence", 0.0)) for p in self.sequence_patterns.values()])
                )
                if self.sequence_patterns
                else 0.0,
            }
            self.smart_bus.set(
                "pattern_analysis",
                pattern_summary,
                module="HistoricalReplayAnalyzer",
                thesis=f"Pattern analysis: {pattern_summary['total_patterns']} patterns identified",
                namespace=ns,
            )

            # Sequence quality
            quality_metrics = {
                "current_quality": float(replay_result.get("sequence_quality", 0.0)),
                "average_quality": float(np.mean(list(self._sequence_quality_scores))) if self._sequence_quality_scores else 0.0,
                "quality_trend": self._calculate_quality_trend(),
                "episodes_processed": int(self._episode_count),
            }
            self.smart_bus.set(
                "sequence_quality",
                quality_metrics,
                module="HistoricalReplayAnalyzer",
                thesis="Sequence quality assessment and learning progress tracking",
                namespace=ns,
            )

            # Learning progress
            learning_metrics = {
                "profitable_sequences_count": len(self.profitable_sequences),
                "total_episodes": int(self._episode_count),
                "success_rate": len(self.profitable_sequences) / max(self._episode_count, 1),
                "learning_acceleration": float(self._calculate_learning_acceleration()),
            }
            self.smart_bus.set(
                "learning_progress",
                learning_metrics,
                module="HistoricalReplayAnalyzer",
                thesis="Learning progress and pattern discovery effectiveness",
                namespace=ns,
            )
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    # ──────────────────────────────────────────────────────────────
    # Derived metrics / health
    # ──────────────────────────────────────────────────────────────
    def _calculate_quality_trend(self) -> str:
        """Calculate sequence quality trend"""
        if len(self._sequence_quality_scores) < 10:
            return "insufficient_data"
        recent_quality = float(np.mean(list(self._sequence_quality_scores)[-5:]))
        older_quality = float(np.mean(list(self._sequence_quality_scores)[-10:-5]))
        if recent_quality > older_quality * 1.1:
            return "improving"
        if recent_quality < older_quality * 0.9:
            return "declining"
        return "stable"

    def _calculate_learning_acceleration(self) -> float:
        """Calculate learning acceleration metric"""
        if self._episode_count < 20:
            return 0.0
        # Compare recent vs overall profitable discovery rate
        recent_window_episodes = max(10, self._episode_count // 4)
        cutoff = time.time() - (recent_window_episodes * 60.0)
        recent_profitable = sum(1 for s in self.profitable_sequences if float(s.get("timestamp", 0.0)) >= cutoff)
        recent_rate = recent_profitable / float(recent_window_episodes)
        overall_rate = len(self.profitable_sequences) / float(max(self._episode_count, 1))
        if overall_rate <= 0.0:
            return 0.0
        return float((recent_rate - overall_rate) / overall_rate)

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no sequence data is available. Must include provides + _thesis."""
        self.logger.warning("No sequence data available - using cached analysis")
        thesis = "No sequence data available – serving cached historical replay analysis state"
        replay_data = {
            "total_sequences": len(self.profitable_sequences),
            "best_sequence_pnl": float(self.best_sequence_pnl),
            "replay_bonus": float(getattr(self, "replay_bonus", 0.0)),
            "sequences_analyzed": int(self._analysis_performance["sequences_analyzed"]),
        }
        pattern_summary = {
            "total_patterns": len(self.sequence_patterns),
            "profitable_patterns": sum(1 for p in self.sequence_patterns.values() if p.get("avg_pnl", 0.0) > 0.0),
            "best_pattern": (
                max(self.sequence_patterns.items(), key=lambda x: float(x[1].get("avg_pnl", 0.0)))[0]
                if self.sequence_patterns
                else None
            ),
            "pattern_confidence_avg": float(
                np.mean([float(p.get("confidence", 0.0)) for p in self.sequence_patterns.values()])
            )
            if self.sequence_patterns
            else 0.0,
        }
        quality_metrics = {
            "current_quality": 0.0,
            "average_quality": float(np.mean(list(self._sequence_quality_scores))) if self._sequence_quality_scores else 0.0,
            "quality_trend": self._calculate_quality_trend() if self._sequence_quality_scores else "insufficient_data",
            "episodes_processed": int(self._episode_count),
        }
        learning_metrics = {
            "profitable_sequences_count": len(self.profitable_sequences),
            "total_episodes": int(self._episode_count),
            "success_rate": len(self.profitable_sequences) / max(self._episode_count, 1),
            "learning_acceleration": float(self._calculate_learning_acceleration()) if self._episode_count >= 20 else 0.0,
        }
        return {
            "replay_sequences": replay_data,
            "pattern_analysis": pattern_summary,
            "sequence_quality": quality_metrics,
            "learning_progress": learning_metrics,
            "_thesis": thesis,
            "fallback_reason": "no_sequence_data",
        }

    async def _handle_replay_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle replay analysis errors"""
        processing_time = (time.time() - start_time) * 1000.0

        # simple CB
        self.circuit_breaker["failures"] = int(self.circuit_breaker.get("failures", 0)) + 1
        self.circuit_breaker["last_failure"] = time.time()
        if self.circuit_breaker["failures"] >= int(self.circuit_breaker.get("threshold", 3)):
            self.circuit_breaker["state"] = "OPEN"

        # log
        explanation = self.english_explainer.explain_error("HistoricalReplayAnalyzer", str(error), "replay analysis")
        self.logger.error(
            format_operator_message(
                "[CRASH]",
                "REPLAY_ANALYSIS_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="replay_analysis",
            )
        )

        # failure metric
        self._record_failure(error)

        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases (include provides + _thesis)."""
        thesis = f"Historical replay encountered issues; serving last known state ({reason})"
        replay_data = {
            "total_sequences": len(self.profitable_sequences),
            "best_sequence_pnl": float(self.best_sequence_pnl),
            "replay_bonus": float(getattr(self, "replay_bonus", 0.0)),
            "sequences_analyzed": int(self._analysis_performance["sequences_analyzed"]),
        }
        pattern_summary = {
            "total_patterns": len(self.sequence_patterns),
            "profitable_patterns": sum(1 for p in self.sequence_patterns.values() if p.get("avg_pnl", 0.0) > 0.0),
            "best_pattern": (
                max(self.sequence_patterns.items(), key=lambda x: float(x[1].get("avg_pnl", 0.0)))[0]
                if self.sequence_patterns
                else None
            ),
            "pattern_confidence_avg": float(
                np.mean([float(p.get("confidence", 0.0)) for p in self.sequence_patterns.values()])
            )
            if self.sequence_patterns
            else 0.0,
        }
        quality_metrics = {
            "current_quality": 0.0,
            "average_quality": float(np.mean(list(self._sequence_quality_scores))) if self._sequence_quality_scores else 0.0,
            "quality_trend": self._calculate_quality_trend() if self._sequence_quality_scores else "insufficient_data",
            "episodes_processed": int(self._episode_count),
        }
        learning_metrics = {
            "profitable_sequences_count": len(self.profitable_sequences),
            "total_episodes": int(self._episode_count),
            "success_rate": len(self.profitable_sequences) / max(self._episode_count, 1),
            "learning_acceleration": float(self._calculate_learning_acceleration()) if self._episode_count >= 20 else 0.0,
        }
        return {
            "replay_sequences": replay_data,
            "pattern_analysis": pattern_summary,
            "sequence_quality": quality_metrics,
            "learning_progress": learning_metrics,
            "_thesis": thesis,
            "fallback_reason": reason,
            "circuit_breaker_state": self.circuit_breaker.get("state", "CLOSED"),
        }

    def _update_replay_health(self) -> None:
        """Update replay analysis health metrics"""
        try:
            # pattern discovery rate
            if self._episode_count > 0:
                pattern_rate = len(self.sequence_patterns) / float(self._episode_count)
                if pattern_rate < 0.1:
                    self._health_status = "warning"
                elif pattern_rate > 0.3:
                    self._health_status = "healthy"

            # average sequence quality vs threshold
            if self._sequence_quality_scores:
                avg_quality = float(np.mean(list(self._sequence_quality_scores)[-10:]))
                if avg_quality < float(self._cfg.min_sequence_quality):
                    self._health_status = "warning"

            self._last_health_check = time.time()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = "warning"

    def _analyze_pattern_effectiveness(self) -> None:
        """Analyze effectiveness of identified patterns"""
        try:
            for pattern, data in self.sequence_patterns.items():
                if int(data.get("count", 0)) >= 5:
                    effectiveness = float(data.get("avg_pnl", 0.0)) * float(data.get("confidence", 0.0))
                    if effectiveness > 10.0:  # arbitrary "strong" threshold
                        self.logger.info(
                            format_operator_message(
                                "[TARGET]",
                                "EFFECTIVE_PATTERN_IDENTIFIED",
                                pattern=pattern,
                                avg_pnl=f"{float(data['avg_pnl']):.2f}",
                                confidence=f"{float(data['confidence']):.2f}",
                                count=int(data["count"]),
                                context="pattern_effectiveness",
                            )
                        )
        except Exception as e:
            self.logger.error(f"Pattern effectiveness analysis failed: {e}")

    # ──────────────────────────────────────────────────────────────
    # Metrics plumbing
    # ──────────────────────────────────────────────────────────────
    def _record_success(self, processing_time_ms: float) -> None:
        """Record successful processing"""
        try:
            self.performance_tracker.record_metric(
                "HistoricalReplayAnalyzer", "analysis_cycle", processing_time_ms, True
            )
        except Exception:
            pass

        # close CB on steady success
        if self.circuit_breaker.get("state") == "OPEN":
            self.circuit_breaker["failures"] = 0
            self.circuit_breaker["state"] = "CLOSED"

    def _record_failure(self, error: Exception) -> None:
        """Record processing failure"""
        try:
            self.performance_tracker.record_metric(
                "HistoricalReplayAnalyzer", "analysis_cycle", 0.0, False
            )
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # Persist/restore state
    # ──────────────────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            "profitable_sequences": list(self.profitable_sequences),
            "sequence_patterns": dict(self.sequence_patterns),
            "genome": dict(self.genome),
            "episode_count": int(self._episode_count),
            "best_sequence_pnl": float(self.best_sequence_pnl),
            "replay_bonus": float(self.replay_bonus),
            "analysis_performance": dict(self._analysis_performance),
            "circuit_breaker": dict(self.circuit_breaker),
            "health_status": str(self._health_status),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state from persistence"""
        if "profitable_sequences" in state:
            self.profitable_sequences = list(state["profitable_sequences"])
        if "sequence_patterns" in state:
            self.sequence_patterns = dict(state["sequence_patterns"])
        if "genome" in state:
            self.genome.update(dict(state["genome"]))
        if "episode_count" in state:
            self._episode_count = int(state["episode_count"])
        if "best_sequence_pnl" in state:
            self.best_sequence_pnl = float(state["best_sequence_pnl"])
        if "replay_bonus" in state:
            self.replay_bonus = float(state["replay_bonus"])
        if "analysis_performance" in state:
            self._analysis_performance.update(dict(state["analysis_performance"]))
        if "circuit_breaker" in state:
            self.circuit_breaker.update(dict(state["circuit_breaker"]))
        if "health_status" in state:
            self._health_status = str(state["health_status"])

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            "status": self._health_status,
            "last_check": self._last_health_check,
            "circuit_breaker": self.circuit_breaker.get("state", "CLOSED"),
            "total_patterns": len(self.sequence_patterns),
            "profitable_sequences": len(self.profitable_sequences),
            "episodes_processed": int(self._episode_count),
        }

    def stop_monitoring(self) -> None:
        """Stop background monitoring"""
        self._monitoring_active = False

    # ──────────────────────────────────────────────────────────────
    # Legacy compatibility
    # ──────────────────────────────────────────────────────────────
    async def propose_action(self, **inputs: Any) -> Dict[str, Any]:
        """Legacy compatibility for action proposal"""
        return {"action": [0.0, 0.0], "confidence": 0.5, "thesis": "Historical replay analyzer does not propose trading actions"}

    def confidence(self, obs: Any = None, **kwargs: Any) -> float:
        """Legacy compatibility for confidence"""
        if self._sequence_quality_scores:
            return float(np.mean(list(self._sequence_quality_scores)[-5:]))
        return 0.5

    async def calculate_confidence(self, action: Dict[str, Any], **inputs: Any) -> float:
        """
        Confidence in the proposed action based on recent sequence quality
        and pattern confidence if the augmented sequence matches a known pattern.
        """
        if self._sequence_quality_scores:
            base_conf = float(np.mean(list(self._sequence_quality_scores)[-5:]))
        else:
            base_conf = 0.5

        # Construct a hypothetical sequence ending with the proposed action
        try:
            augmented = list(self.current_sequence) + [{"action": np.asarray(list(action), dtype=float).tolist()}]
        except Exception:
            augmented = list(self.current_sequence) + [{"action": [0.0, 0.0]}]

        pattern = self._extract_sequence_pattern(augmented)
        if pattern and pattern in self.sequence_patterns:
            pattern_conf = float(self.sequence_patterns[pattern].get("confidence", 0.0))
            # 70% sequence quality, 30% pattern confidence
            conf = (0.7 * base_conf) + (0.3 * pattern_conf)
        else:
            conf = base_conf

        return float(max(0.0, min(1.0, conf)))
