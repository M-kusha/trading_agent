"""
🏛️ Enhanced Strategy Arbiter with SmartInfoBus Integration v3.1
Advanced multi-expert coordination, robust gating, and production-grade telemetry.
"""

from __future__ import annotations

import asyncio
import time
from modules.contracts import module_args
import numpy as np
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict

# ═══════════════════════════════════════════════════════════════════
# MODERN SMARTINFOBUS IMPORTS
# ═══════════════════════════════════════════════════════════════════
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker
from utils.get_dir import _BASE_GATE, _smart_gate


@module(**module_args(
    "StrategyArbiter",
    description="Advanced multi-expert coordination and sophisticated voting mechanisms",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class StrategyArbiter(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🏛️ PRODUCTION-GRADE Strategy Arbiter v3.1

    Capabilities
    ────────────
    • Sophisticated multi-expert blending with confidence & adaptive weights
    • Regime-aware, criteria-weighted gating with bootstrap handling
    • Per-instrument signal publishing for downstream execution layers
    • Member performance analytics (contribution, reliability, specialization)
    • Learning loop (REINFORCE-style) with baseline and telemetry
    • Full SmartInfoBus integration + circuit breaker, health & performance metrics
    """

    # Learning knobs
    REINFORCE_LR: float = 0.001
    REINFORCE_LAMBDA: float = 0.95
    PRIOR_BLEND: float = 0.30

    # ────────────────────────────
    # INIT
    # ────────────────────────────
    def _initialize(self) -> None:
        # Mixins & systems
        self._initialize_trading_state()
        self._initialize_state_management()
        self._init_systems()

        # Config
        self.members: List[Any] = list(self.config.get("members", []))
        init_weights = self.config.get("init_weights", [1.0] * max(1, len(self.members)))
        self.action_dim: int = int(self.config.get("action_dim", 4))
        self.adapt_rate: float = float(self.config.get("adapt_rate", 0.01))
        self.min_confidence: float = float(self.config.get("min_confidence", 0.3))
        self.bootstrap_steps: int = int(self.config.get("bootstrap_steps", 50))
        self.debug: bool = bool(self.config.get("debug", True))

        # Validation
        if len(init_weights) != max(1, len(self.members)):
            raise ValueError(
                f"init_weights ({len(init_weights)}) must match members ({len(self.members) or 1})"
            )

        # Core state
        self._ensure_instruments()
        self.weights: np.ndarray = np.asarray(init_weights, dtype=np.float32)
        self.weights = self._normalize_weights(self.weights)
        self.last_alpha: Optional[np.ndarray] = None

        # Market state
        self.curr_vol: float = 0.01
        self.market_regime: str = "unknown"
        self.market_session: str = "unknown"
        self.market_context: Dict[str, Any] = {}

        # Learning state
        self._baseline: float = 0.0
        self._baseline_beta: float = 0.98
        self.learning_history: deque = deque(maxlen=100)

        # Decision & audit
        self._trace: List[Dict[str, Any]] = []
        self._log_size: int = int(self.config.get("audit_log_size", 100))
        self.decision_history: deque = deque(maxlen=200)
        self.proposal_history: deque = deque(maxlen=100)
        self.gate_decisions: deque = deque(maxlen=150)

        # Counters & intelligence
        self._gate_passes: int = 0
        self._gate_attempts: int = 0
        self._step_count: int = 0
        self.gate_intelligence: Dict[str, Any] = {
            "adaptive_threshold": True,
            "criteria_weights": [0.25, 0.20, 0.20, 0.20, 0.15],  # strength, consensus, reliability, risk, novelty
            "bootstrap_factor": 0.5,
            "regime_adjustments": {"volatile": 0.8, "trending": 1.2, "ranging": 1.0, "noise": 0.7},
        }

        # Member tracking
        self.member_performance: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "proposals_made": 0,
                "successful_proposals": 0,
                "avg_confidence": 0.5,
                "recent_performance": deque(maxlen=20),
                "weight_evolution": deque(maxlen=50),
                "quality_scores": deque(maxlen=30),
                "contribution_score": 0.5,
                "reliability_index": 0.5,
                "specialization_score": 0.5,
            }
        )

        # Metrics & analytics
        self.voting_quality_metrics: Dict[str, float] = {"overall_quality_score": 0.5}
        self.member_analytics: Dict[str, float] = {"performance_consistency": 0.5}
        self.regime_analytics: Dict[str, float] = {"current_regime_fit": 0.5}
        self.coordination_analytics: Dict[str, float] = {"coordination_effectiveness": 0.5}

        self.voting_quality: Dict[str, float] = {
            "avg_consensus": 0.5,
            "collusion_risk": 0.0,
            "gate_effectiveness": 0.5,
            "member_diversity": 0.5,
            "decision_confidence": 0.5,
            "proposal_quality": 0.5,
            "learning_efficiency": 0.5,
            "adaptation_rate": 0.0,
        }

        self.arbiter_stats: Dict[str, Any] = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "weight_adaptations": 0,
            "consensus_failures": 0,
            "collusion_detected": 0,
            "gate_pass_rate": 0.0,
            "avg_proposal_quality": 0.5,
            "learning_convergence": 0.0,
            "member_coordination": 0.5,
            "decision_latency": 0.0,
            "session_start": dt.datetime.now().isoformat(),
        }

        self.decision_intelligence: Dict[str, Any] = {
            "quality_threshold": 0.7,
            "adaptation_sensitivity": 0.15,
            "member_learning_rate": 0.05,
            "consensus_weight": 0.3,
            "performance_memory": 0.9,
            "regime_adaptation": True,
            "dynamic_weighting": True,
        }

        self.market_adaptation: Dict[str, Any] = {
            "regime_multipliers": {
                "trending": {"confidence_boost": 1.1, "gate_adjustment": 1.2},
                "volatile": {"confidence_boost": 0.9, "gate_adjustment": 0.8},
                "ranging": {"confidence_boost": 1.0, "gate_adjustment": 1.0},
                "noise": {"confidence_boost": 0.8, "gate_adjustment": 0.7},
                "unknown": {"confidence_boost": 1.0, "gate_adjustment": 1.0},
            },
            "session_adjustments": {"american": 1.0, "european": 0.95, "asian": 0.9, "rollover": 0.6},
        }

        # Circuit breaker
        self.error_count: int = 0
        self.circuit_breaker_threshold: int = 5
        self.is_disabled: bool = False

        # Concurrency guard
        self._process_lock: asyncio.Lock = asyncio.Lock()

        # Thesis & early BUS pubs
        self._generate_initialization_thesis()
        version = getattr(self.metadata, "version", "3.1.0") if self.metadata else "3.1.0"
        self.logger.info(
            format_operator_message(
                icon="🏛️",
                message=f"Strategy Arbiter v{version} initialized",
                members=len(self.members),
                action_dim=self.action_dim,
                bootstrap_steps=self.bootstrap_steps,
                adaptive_learning=True,
            )
        )
        try:
            # Publish instrument universe under common aliases to satisfy downstream readers
            inst_list = list(self.instruments)
            self.smart_bus.set("instruments", inst_list, module="StrategyArbiter", thesis="Universe of instruments")

            self.smart_bus.set("alpha_weights", self.weights.tolist(), module="StrategyArbiter", thesis="Initial alpha weights")
            self.smart_bus.set(
                "blended_action",
                [0.0] * max(1, int(self.action_dim)),
                module="StrategyArbiter",
                thesis="Initial blended action placeholder",
            )
        except Exception:
            pass

    def _init_systems(self) -> None:
        """Initialize logging, error handling, telemetry, health."""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="StrategyArbiter",
            log_path="logs/voting/strategy_arbiter.log",
            max_lines=10000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("StrategyArbiter", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _generate_initialization_thesis(self) -> None:
        thesis = f"""
Strategy Arbiter v3.1 Initialization:
- Members={len(self.members)}, ActionDim={self.action_dim}, Bootstrap={self.bootstrap_steps}
- Gate criteria weights={self.gate_intelligence['criteria_weights']}, Adaptive={self.gate_intelligence['adaptive_threshold']}
- Learning: baseline_beta={self._baseline_beta:.3f}, adapt_rate={self.adapt_rate:.4f}
"""
        # keep a local copy so validate_outputs can include it in process() results
        self._init_payload = {
            "status": "initialized",
            "thesis": thesis,
            "timestamp": dt.datetime.now().isoformat(),
            "configuration": {
                "members": len(self.members),
                "action_dim": self.action_dim,
                "intelligence_parameters": self.decision_intelligence,
                "gate_parameters": self.gate_intelligence,
            },
        }
        self.smart_bus.set(
            "strategy_arbiter_initialization",
            dict(self._init_payload),
            module="StrategyArbiter",
            thesis=thesis,
        )

    # ────────────────────────────
    # PROCESS
    # ────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive strategy arbitration.
        Computes & publishes per-instrument signals, updates telemetry and quality metrics.
        """
        async with self._process_lock:
            start_time = time.time()
            try:
                if self.is_disabled:
                    return self._generate_disabled_response()

                # FIX #2: Read kernel's decision_id for coordination
                decision_id = self.smart_bus.get('kernel_decision_id', 'StrategyArbiter')
                
                # 1) Data & market state
                market_data = await self._get_comprehensive_market_data()
                await self._update_market_state_comprehensive(market_data)

                # Debug: Print received proposals
                # proposal_vectors = market_data.get('proposal_vectors', [])
                # member_proposals = market_data.get('member_proposals', [])
                # member_confidences = market_data.get('member_confidences', [])

                # print(f"\n{'='*80}")
                # print(f"STRATEGY ARBITER - Received Proposals")
                # print(f"{'='*80}")
                # print(f"Proposal vectors received: {len(proposal_vectors)}")
                # print(f"Member proposals received: {len(member_proposals)}")
                # print(f"Member confidences received: {len(member_confidences)}")
                # if proposal_vectors:
                #     print(f"\nProposal vectors content:")
                #     for i, pv in enumerate(proposal_vectors[:10], 1):  # Show first 10
                #         print(f"  {i}. {pv}")
                # if member_confidences:
                #     print(f"\nMember confidences:")
                #     for i, mc in enumerate(member_confidences[:10], 1):  # Show first 10
                #         print(f"  {i}. {mc:.3f}")
                # print(f"{'='*80}\n")

                # 2) Blended proposal + signals
                blended_proposal = self._compute_blended_proposal(market_data)
                signals = self._map_action_to_instrument_signals(blended_proposal)
                self._publish_instrument_signals(signals)
                self.smart_bus.set(
                    "instrument_signals",
                    signals,
                    module="StrategyArbiter",
                    thesis=f"Per-instrument intensities for {len(signals)} instruments",
                )

                # 3) Analytics & recommendations
                performance_analysis = await self._analyze_member_performance_comprehensive(market_data)
                quality_updates = await self._update_voting_quality_metrics_comprehensive()
                recommendations = await self._generate_intelligent_arbitration_recommendations(
                    performance_analysis, quality_updates
                )
                thesis = await self._generate_comprehensive_arbitration_thesis(performance_analysis, recommendations)

                # 4) Compose & publish
                # Build required outputs for contract validation
                init_payload = getattr(self, "_init_payload", None)
                if not init_payload:
                    try:
                        init_payload = self.smart_bus.get("strategy_arbiter_initialization", "StrategyArbiter") or {}
                    except Exception:
                        init_payload = {}

                # strategy_weights payload: map names -> weights
                try:
                    member_names: List[str] = []
                    for i, m in enumerate(self.members):
                        nm = None
                        try:
                            nm = getattr(m, "name", None) or getattr(m, "module_name", None)
                        except Exception:
                            nm = None
                        member_names.append(nm if isinstance(nm, str) and nm else f"member_{i}")
                    weights_list = self.weights.tolist()
                    strategy_weights = {
                        "by_member": {member_names[i]: float(weights_list[i]) for i in range(min(len(member_names), len(weights_list)))},
                        "members": member_names,
                        "weights": weights_list,
                        "timestamp": dt.datetime.utcnow().isoformat(),
                    }
                except Exception:
                    strategy_weights = {"by_member": {}, "members": [], "weights": [], "timestamp": dt.datetime.utcnow().isoformat()}

                # Generate expert_performance output required by contract
                expert_performance = self._generate_expert_performance_output()

                inst_list = list(getattr(self, "instruments", []))
                results: Dict[str, Any] = {
                    "blended_action": blended_proposal.tolist(),
                    "alpha_weights": self.last_alpha.tolist() if self.last_alpha is not None else [],
                    "member_weights": self.weights.tolist(),
                    "strategy_weights": strategy_weights,
                    "gate_decision": self._get_recent_gate_decision(),
                    "voting_quality": dict(self.voting_quality),
                    "member_performance": self._get_member_performance_summary(),
                    "decision_statistics": self._get_comprehensive_arbiter_stats(),
                    "proposal_analysis": self._get_recent_proposal_analysis(),
                    "arbiter_recommendations": list(recommendations),
                    "health_metrics": self._get_health_metrics(),
                    "instrument_signals": signals,
                    "instruments": inst_list,
                    "universe": inst_list,
                    "watched_instruments": inst_list,
                    "decision_id": decision_id or market_data.get("decision_id"),  # FIX #2: Use kernel's decision_id
                    "arbiter_decision_id": decision_id or market_data.get("decision_id"),  # FIX: Contract-required namespaced decision_id
                    "tick_ts": market_data.get("tick_ts") or dt.datetime.utcnow().isoformat(),
                    "_thesis": thesis,
                    "strategy_arbiter_initialization": init_payload,
                    "expert_performance": expert_performance,
                }
                await self._update_smartinfobus_comprehensive(results, thesis)

                # 5) Perf
                self.performance_tracker.record_metric(
                    "StrategyArbiter", "process_time_ms", (time.time() - start_time) * 1000.0, True
                )
                self.error_count = 0
                return results

            except Exception as e:
                return await self._handle_processing_error(e, start_time)

    # ────────────────────────────
    # BUS IO & MARKET STATE
    # ────────────────────────────
    async def _get_comprehensive_market_data(self) -> Dict[str, Any]:
        try:
            g = self.smart_bus.get
            # Use proposal_vectors (numeric) when available, fall back to member_proposals (rich dicts)
            proposal_vectors = g("proposal_vectors", "StrategyArbiter") or []
            member_proposals = g("member_proposals", "StrategyArbiter") or []

            return {
                "market_context": g("market_context", "StrategyArbiter") or {},
                "recent_trades": g("recent_trades", "StrategyArbiter") or [],
                "current_positions": g("current_positions", "StrategyArbiter") or [],
                "member_proposals": proposal_vectors if proposal_vectors else member_proposals,
                "proposal_vectors": proposal_vectors,  # Keep separate for clarity
                "member_confidences": g("member_confidences", "StrategyArbiter") or [],
                "consensus_score": g("consensus_score", "StrategyArbiter") or 0.5,
                "collusion_score": g("collusion_score", "StrategyArbiter") or 0.0,
                "horizon_alignment": g("horizon_alignment", "StrategyArbiter") or {},
                "volatility_data": g("volatility_data", "StrategyArbiter") or {},
                "market_regime": g("market_regime", "StrategyArbiter") or "unknown",
                "session_data": g("session_data", "StrategyArbiter") or {},
                "instruments": g("instruments", "StrategyArbiter") or list(getattr(self, "instruments", [])),
                "decision_id": g("decision_id", "StrategyArbiter"),
                "tick_ts": g("tick_ts", "StrategyArbiter"),
            }
        except Exception as e:
            ctx = self.error_pinpointer.analyze_error(e, "StrategyArbiter")
            self.logger.warning(f"Market data retrieval incomplete: {ctx}")
            return self._get_safe_market_defaults()

    async def _update_market_state_comprehensive(self, market_data: Dict[str, Any]) -> None:
        try:
            old_regime = self.market_regime
            self.market_regime = str(market_data.get("market_regime", "unknown"))
            self.market_session = str(market_data.get("session_data", {}).get("current_session", "unknown"))

            # Volatility extraction (supports various shapes)
            volatility_data = market_data.get("volatility_data", {})
            self.curr_vol = self._extract_avg_volatility(volatility_data)

            # Instruments sync (if changed upstream)
            if isinstance(market_data.get("instruments"), (list, tuple)) and market_data["instruments"]:
                self.instruments = list(market_data["instruments"])

            self.market_context = dict(market_data.get("market_context", {}))

            if old_regime != self.market_regime and old_regime != "unknown":
                self.logger.info(
                    format_operator_message(
                        icon="[STATS]",
                        message="Market regime transition detected",
                        old_regime=old_regime,
                        new_regime=self.market_regime,
                        volatility=f"{self.curr_vol:.3f}",
                        session=self.market_session,
                        impact="Strategy weights will adapt",
                    )
                )
                await self._apply_regime_adaptations()
        except Exception as e:
            ctx = self.error_pinpointer.analyze_error(e, "market_state_update")
            self.logger.warning(f"Market state update failed: {ctx}")

    def _extract_avg_volatility(self, vol: Any) -> float:
        try:
            if isinstance(vol, dict) and vol:
                vals: List[float] = []
                for v in vol.values():
                    if isinstance(v, (int, float)):
                        vals.append(float(v))
                    elif isinstance(v, dict):
                        for k in ("volatility", "atr", "sigma", "value", "vol"):
                            if isinstance(v.get(k), (int, float)):
                                vals.append(float(v[k]))
                                break
                if vals:
                    return max(0.001, float(np.mean(vals)))
            elif isinstance(vol, (int, float)):
                return max(0.001, float(vol))
            return 0.01
        except Exception:
            return 0.01

    async def _apply_regime_adaptations(self) -> None:
        try:
            rm = self.market_adaptation["regime_multipliers"].get(self.market_regime, {})
            if self.market_regime == "volatile":
                self.gate_intelligence["criteria_weights"] = [0.30, 0.25, 0.15, 0.15, 0.15]
            elif self.market_regime == "trending":
                self.gate_intelligence["criteria_weights"] = [0.20, 0.15, 0.25, 0.25, 0.15]
            elif self.market_regime == "ranging":
                self.gate_intelligence["criteria_weights"] = [0.25, 0.20, 0.20, 0.20, 0.15]
            else:
                self.gate_intelligence["criteria_weights"] = [0.35, 0.20, 0.15, 0.15, 0.15]

            boost = float(rm.get("confidence_boost", 1.0))
            self.decision_intelligence["adaptation_sensitivity"] *= boost
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "regime_adaptations")

    # ────────────────────────────
    # PROPOSAL / BLENDING / SIGNALS
    # ────────────────────────────
    def _ensure_instruments(self) -> None:
        """Ensure instruments universe is available."""
        if getattr(self, "instruments", None):
            return
        try:
            cfg_inst = self.config.get("instruments") if isinstance(self.config, dict) else None
        except Exception:
            cfg_inst = None
        if isinstance(cfg_inst, (list, tuple)) and cfg_inst:
            self.instruments = list(cfg_inst)
            return

        # Try bus
        for key in ("instruments", "watched_instruments", "universe"):
            vals = InfoBusManager.get_instance().get(key, "StrategyArbiter")
            if isinstance(vals, (list, tuple)) and vals:
                self.instruments = list(vals)
                return

        # Heuristic from vol keys
        vol = InfoBusManager.get_instance().get("volatility_data", "StrategyArbiter") or {}
        if isinstance(vol, dict) and vol:
            cands = [k for k in vol.keys() if isinstance(k, str)]
            if cands:
                self.instruments = cands
                return

        self.instruments = ["XAU_USD", "EUR_USD"]

    def _map_action_to_instrument_signals(self, action: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Map a vector to {instrument: {'intensity','confidence'}}.

        Robust to shapes:
        - N: [i0, i1, ..., i{n-1}]
        - 2N interleaved: [i0, c0, i1, c1, ...]
        Chooses the variant with non-zero informative content.
        """
        self._ensure_instruments()
        action = np.asarray(action, dtype=np.float32).flatten()
        n = len(self.instruments)

        def _conf_default() -> float:
            try:
                return float(self.config.get("default_signal_confidence", 0.6))
            except Exception:
                return 0.6

        out: Dict[str, Dict[str, float]] = {}
        if action.size >= 2 * n:
            interleaved = action[0 : 2 * n : 2]
            contiguous = action[:n]
            nz_inter = int(np.count_nonzero(interleaved))
            nz_contig = int(np.count_nonzero(contiguous))
            # Heuristic: prefer the variant with more non-zero content; tie-break on mean abs
            if (nz_contig > nz_inter) or (
                nz_contig == nz_inter and float(np.mean(np.abs(contiguous))) > float(np.mean(np.abs(interleaved)))
            ):
                chosen = contiguous
                chosen_mode = "contiguous"
            else:
                chosen = interleaved
                chosen_mode = "interleaved"
            for i, inst in enumerate(self.instruments):
                out[inst] = {"intensity": float(np.clip(chosen[i], -1.0, 1.0)), "confidence": _conf_default()}
            try:
                if getattr(self, "debug", False):
                    self.logger.debug(
                        format_operator_message(
                            icon="[MAP]",
                            message="Action mapping",
                            mode=chosen_mode,
                            nz_inter=nz_inter,
                            nz_contig=nz_contig,
                            mean_inter=f"{float(np.mean(np.abs(interleaved))):.3f}",
                            mean_contig=f"{float(np.mean(np.abs(contiguous))):.3f}",
                        )
                    )
            except Exception:
                pass
        elif action.size >= n:
            for i, inst in enumerate(self.instruments):
                out[inst] = {"intensity": float(np.clip(action[i], -1.0, 1.0)), "confidence": _conf_default()}
        else:
            for i, inst in enumerate(self.instruments):
                v = float(action[i]) if i < action.size else 0.0
                out[inst] = {"intensity": float(np.clip(v, -1.0, 1.0)), "confidence": _conf_default()}
        return out

    def _publish_instrument_signals(self, signals: Dict[str, Dict[str, float]]) -> None:
        """Publish signals to multiple key variants for consumer robustness."""
        now = dt.datetime.utcnow().isoformat()
        for inst, payload in signals.items():
            data = {
                "intensity": float(payload.get("intensity", 0.0)),
                "confidence": float(payload.get("confidence", 0.5)),
                "timestamp": now,
                "source": "StrategyArbiter",
            }
            keys = (
                f"signal_{inst}",
                f"signal_{inst.replace('/', '')}",
                f"signal_{inst.replace('/', '_')}",
                f"signal_{inst.upper()}",
                f"signal_{inst.replace('/', '').upper()}",
            )
            for k in keys:
                self.smart_bus.set(k, data, module="StrategyArbiter", thesis=f"Arbiter signal {inst}: {data['intensity']:.3f}")

    def _compute_blended_proposal(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Produce blended proposal vector using committee bus data when available,
        falling back to legacy member-based blending. Always returns an action_dim-sized array,
        and records a gate decision.
        """
        self._ensure_instruments()
        try:
            if not isinstance(self.action_dim, int) or self.action_dim <= 0:
                self.action_dim = max(1, 2 * len(self.instruments))
        except Exception:
            self.action_dim = max(1, 2 * len(self.instruments))

        # 1) Prefer committee-provided proposals from SmartInfoBus
        try:
            proposals_in = list(market_data.get("proposal_vectors") or [])
            confidences_in = list(market_data.get("member_confidences") or [])
        except Exception:
            proposals_in, confidences_in = [], []

        if proposals_in:
            try:
                props: List[np.ndarray] = []
                for p in proposals_in:
                    pa = np.asarray(p, dtype=np.float32).flatten()
                    if pa.size < self.action_dim:
                        pa = np.pad(pa, (0, self.action_dim - pa.size))
                    elif pa.size > self.action_dim:
                        pa = pa[: self.action_dim]
                    props.append(pa)

                # Weights: use member_confidences when provided; else equal weights
                if confidences_in and len(confidences_in) >= len(props):
                    c = np.asarray(confidences_in[: len(props)], dtype=np.float32)
                    c = np.maximum(c, 1e-6)
                    w = c / float(c.sum())
                else:
                    w = np.ones(len(props), dtype=np.float32) / max(1, len(props))

                blended = np.zeros(self.action_dim, dtype=np.float32)
                for wi, pi in zip(w, props):
                    blended += float(wi) * pi

                # Debug: summarize blending stats (guarded by debug)
                try:
                    if self.debug:
                        preview_w = (
                            w[: min(5, len(w))].tolist() if hasattr(w, "tolist") else list(w)[:5]
                        )
                        self.logger.debug(
                            format_operator_message(
                                icon="[ARB]",
                                message="Committee blend",
                                members=len(props),
                                weights_preview=[float(x) for x in preview_w],
                                mean_abs=f"{float(np.mean(np.abs(blended))):.3f}",
                                max_abs=f"{float(np.max(np.abs(blended))):.3f}",
                            )
                        )
                except Exception:
                    pass

                # Gate using consensus/collusion from BUS
                cons = float(self.smart_bus.get("consensus_score", "StrategyArbiter") or market_data.get("consensus_score", 0.5) or 0.5)
                coll = float(self.smart_bus.get("collusion_score", "StrategyArbiter") or market_data.get("collusion_score", 0.0) or 0.0)
                passed, _thr, crit = self._evaluate_gate(blended, cons, coll)
                final_action = blended if passed else np.zeros_like(blended)
                self._record_gate_decision(passed, crit, final_action)
                return final_action
            except Exception:
                # Fall back to legacy path if committee fusion fails
                pass

        # 2) Legacy path: build proposal from local members
        try:
            obs = self.get_observation_components()
            if not isinstance(obs, np.ndarray):
                obs = np.zeros(self.action_dim, dtype=np.float32)
        except Exception:
            obs = np.zeros(self.action_dim, dtype=np.float32)

        proposal = self.propose(obs)  # legacy compatibility path (includes gate + record)
        proposal = np.asarray(proposal, dtype=np.float32).flatten()
        if proposal.size < self.action_dim:
            proposal = np.pad(proposal, (0, self.action_dim - proposal.size))
        elif proposal.size > self.action_dim:
            proposal = proposal[: self.action_dim]
        return proposal

    # ═══════════════════════════════════════════════════════════════════
    # LEGACY PATHS (kept robust, productionized)
    # ═══════════════════════════════════════════════════════════════════
    def propose(self, obs: Any) -> np.ndarray:
        """Legacy proposal interface for backward compatibility."""
        try:
            return self._simple_proposal_fallback(obs)
        except Exception as e:
            ctx = self.error_pinpointer.analyze_error(e, "legacy_proposal")
            self.logger.error(f"Legacy proposal failed: {ctx}")
            return np.zeros(self.action_dim, dtype=np.float32)

    def _normalize_weights(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=np.float32)
        w = np.maximum(w, 1e-6)
        s = float(w.sum())
        return w / s if s > 0 else np.ones_like(w) / len(w)

    def _evaluate_gate(self, action: np.ndarray, consensus: float, collusion: float) -> Tuple[bool, float, Dict[str, float]]:
        """
        Multi-criteria gate: blend absolute strength, consensus support, reliability proxy,
        risk (volatility/collusion), novelty (change vs. previous).
        """
        try:
            weights = self.gate_intelligence.get("criteria_weights", [0.25, 0.20, 0.20, 0.20, 0.15])
            strength = float(np.clip(np.abs(action).mean(), 0.0, 1.0))
            reliability = 1.0 - float(np.std(self.weights)) if len(self.weights) > 1 else 0.5
            risk = float(np.clip(1.0 - (self.curr_vol * 5.0 + collusion), 0.0, 1.0))
            novelty = 0.5
            if self.decision_history:
                prev = np.asarray(self.decision_history[-1].get("action", np.zeros_like(action)), dtype=np.float32)
                novelty = float(np.clip(1.0 - min(1.0, float(np.linalg.norm(action - prev))), 0.0, 1.0))

            crit = [strength, float(np.clip(consensus, 0.0, 1.0)), reliability, risk, novelty]
            score = float(np.dot(np.array(crit, dtype=np.float32), np.array(weights, dtype=np.float32)))
            # Adaptive baseline threshold
            base_thr = _smart_gate(float(self.curr_vol), 0) if self._step_count >= self.bootstrap_steps else _BASE_GATE * 0.5
            adj = self.market_adaptation["regime_multipliers"].get(self.market_regime, {}).get("gate_adjustment", 1.0)
            threshold = float(np.clip(base_thr / adj, 0.05, 0.95))
            return (score >= threshold), threshold, {
                "strength": strength,
                "consensus": float(np.clip(consensus, 0.0, 1.0)),
                "reliability": reliability,
                "risk": risk,
                "novelty": novelty,
                "gate_score": score,
                "threshold": threshold,
            }
        except Exception:
            # fallback to simple strength gating
            s = float(np.abs(action).mean())
            thr = _BASE_GATE * 0.5
            return (s >= thr), thr, {"strength": s, "threshold": thr}

    def _record_gate_decision(self, passed: bool, details: Dict[str, float], action: np.ndarray) -> None:
        try:
            self._gate_attempts += 1
            if passed:
                self._gate_passes += 1

            # Derive aggregate direction/size from the action vector
            try:
                n = len(getattr(self, "instruments", []))
            except Exception:
                n = 0
            a = np.asarray(action, dtype=np.float32).flatten()
            intensities: np.ndarray
            if n and a.size >= 2 * n:
                intensities = a[0 : 2 * n : 2]
            elif n and a.size >= n:
                intensities = a[:n]
            else:
                intensities = a

            avg_dir = float(np.mean(intensities)) if intensities.size else 0.0
            strength = float(np.clip(np.mean(np.abs(intensities)) if intensities.size else 0.0, 0.0, 1.0))

            # Map to discrete action label
            dir_eps = 0.02
            if passed:
                if avg_dir > dir_eps:
                    action_label = "buy"
                elif avg_dir < -dir_eps:
                    action_label = "sell"
                else:
                    action_label = "abstain"
            else:
                action_label = "abstain"

            # Confidence blends gate score and signal strength (bounded [0,1])
            gate_score = float(details.get("gate_score", 0.0))
            confidence = float(np.clip(0.6 * gate_score + 0.4 * strength, 0.0, 1.0)) if passed else 0.0

            record = {
                "decision": "pass" if passed else "block",
                "action": action_label,
                "size": strength if passed else 0.0,
                "confidence": confidence,
                "criteria": details,
                "timestamp": dt.datetime.now().isoformat(),
                "reason": "gate_passed" if passed else "gating_blocked",
            }

            self.gate_decisions.append(record)
            self.decision_history.append(
                {
                    "timestamp": dt.datetime.now().isoformat(),
                    "action": a.tolist(),
                    "signal_strength": float(np.abs(a).mean()),
                    "passed": passed,
                }
            )
            self.arbiter_stats["total_decisions"] += 1
            if passed:
                self.arbiter_stats["successful_decisions"] = self.arbiter_stats.get("successful_decisions", 0) + 1
            self.arbiter_stats["gate_pass_rate"] = self._gate_passes / max(self._gate_attempts, 1)
            # Debug: concise gate breakdown (guarded by debug)
            try:
                if self.debug:
                    self.logger.debug(
                        format_operator_message(
                            icon="[GATE]",
                            message="Gate PASS" if passed else "Gate BLOCK",
                            decision=action_label,
                            size=f"{strength:.3f}",
                            confidence=f"{confidence:.3f}",
                            score=f"{float(details.get('gate_score', 0.0)):.3f}",
                            threshold=f"{float(details.get('threshold', 0.0)):.3f}",
                            strength=f"{float(details.get('strength', 0.0)):.3f}",
                            consensus=f"{float(details.get('consensus', 0.0)):.3f}",
                            risk=f"{float(details.get('risk', 0.0)):.3f}",
                            novelty=f"{float(details.get('novelty', 0.0)):.3f}",
                        )
                    )
            except Exception:
                pass
        except Exception:
            pass

    def _simple_proposal_fallback(self, obs: Any) -> np.ndarray:
        """Simple, safe blending & gating with confidence × weight alphas."""
        try:
            self._step_count += 1

            proposals: List[np.ndarray] = []
            confidences: List[float] = []

            for i, member in enumerate(self.members):
                try:
                    if hasattr(member, "propose_action"):
                        prop = member.propose_action(obs)
                    elif hasattr(member, "propose"):
                        prop = member.propose(obs)
                    else:
                        prop = np.zeros(self.action_dim, dtype=np.float32)

                    prop = np.asarray(prop, dtype=np.float32).flatten()
                    if prop.size < self.action_dim:
                        prop = np.pad(prop, (0, self.action_dim - prop.size))
                    elif prop.size > self.action_dim:
                        prop = prop[: self.action_dim]
                    proposals.append(prop)

                    if hasattr(member, "confidence"):
                        conf = float(member.confidence(obs))
                    else:
                        conf = 0.5
                    confidences.append(max(conf, self.min_confidence))
                except Exception:
                    proposals.append(np.zeros(self.action_dim, dtype=np.float32))
                    confidences.append(self.min_confidence)

            if not proposals:
                return np.zeros(self.action_dim, dtype=np.float32)

            # Blend (weights × confidences)
            w_norm = self._normalize_weights(self.weights)
            c = np.asarray(confidences, dtype=np.float32)
            c_norm = c / (float(c.sum()) + 1e-12)
            alpha = w_norm * c_norm
            alpha = alpha / (float(alpha.sum()) + 1e-12)
            self.last_alpha = alpha.copy()

            action = np.zeros(self.action_dim, dtype=np.float32)
            for prop, a in zip(proposals, alpha):
                action += a * prop

            # Gate: include consensus & collusion from BUS if present
            cons = self.smart_bus.get("consensus_score", "StrategyArbiter") or 0.5
            coll = self.smart_bus.get("collusion_score", "StrategyArbiter") or 0.0
            passed, threshold, crit = self._evaluate_gate(action, float(cons), float(coll))
            if passed:
                final_action = action
            else:
                final_action = np.zeros_like(action)

            self._record_gate_decision(passed, crit, final_action)
            return final_action

        except Exception as e:
            self.logger.error(f"Simple proposal fallback failed: {e}")
            return np.zeros(self.action_dim, dtype=np.float32)

    # ────────────────────────────
    # ANALYTICS & ADAPTATION
    # ────────────────────────────
    async def _analyze_member_performance_comprehensive(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            performance_analysis: Dict[str, Any] = {
                "member_updates": {},
                "weight_changes": {},
                "quality_assessments": {},
                "specialization_analysis": {},
                "coordination_effectiveness": 0.0,
            }

            recent_trades = market_data.get("recent_trades", [])
            member_proposals = market_data.get("member_proposals", [])
            member_confidences = market_data.get("member_confidences", [])

            recent_pnl = [float(t.get("pnl", 0.0)) for t in recent_trades[-10:]] if recent_trades else []
            recent_success_rate = (sum(1 for v in recent_pnl if v > 0) / len(recent_pnl)) if recent_pnl else 0.5

            for i, _member in enumerate(self.members):
                if i >= len(self.weights):
                    continue
                info = await self._analyze_individual_member_performance(
                    i, member_proposals, member_confidences, recent_success_rate
                )
                performance_analysis["member_updates"][i] = info

                # Adaptive weight update
                dw = await self._calculate_adaptive_weight_update(i, info)
                if abs(dw) > 0.05:
                    performance_analysis["weight_changes"][i] = {
                        "old_weight": float(self.weights[i]),
                        "weight_change": float(dw),
                        "reason": info.get("primary_factor", "performance"),
                    }
                    self.weights[i] = max(0.01, float(self.weights[i] + dw))

            self.weights = self._normalize_weights(self.weights)
            performance_analysis["coordination_effectiveness"] = await self._calculate_coordination_effectiveness(
                performance_analysis
            )
            return performance_analysis
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "member_performance_analysis")
            return {"member_updates": {}, "coordination_effectiveness": 0.5}

    def _handle_proposal_dimension_mismatch(self, proposal: Any, target_dim: Optional[int] = None) -> np.ndarray:
        """Handle dimension mismatches in proposals - pad, trim, or extract numeric features"""
        try:
            target_dim = target_dim or self.action_dim

            # If it's already a numeric array
            if isinstance(proposal, (list, np.ndarray)):
                arr = np.asarray(proposal, dtype=np.float32).flatten()
                if arr.size == target_dim:
                    return arr
                elif arr.size < target_dim:
                    # Pad with zeros
                    return np.pad(arr, (0, target_dim - arr.size))
                else:
                    # Trim to target size
                    return arr[:target_dim]

            # If it's a dict (rich proposal), extract numeric features
            elif isinstance(proposal, dict):
                vote = proposal.get('vote', {})
                action = str(vote.get('action', 'abstain')).lower()
                signal_strength = float(vote.get('signal_strength', 0.0))
                confidence = float(proposal.get('confidence', 0.0))

                # Create a minimal numeric vector
                numeric_features = [
                    self._action_to_numeric(action) * signal_strength,
                    confidence
                ]

                # Pad or trim to target dimension
                arr = np.array(numeric_features, dtype=np.float32)
                if arr.size < target_dim:
                    arr = np.pad(arr, (0, target_dim - arr.size))
                elif arr.size > target_dim:
                    arr = arr[:target_dim]
                return arr

            # Fallback: return zeros
            return np.zeros(target_dim, dtype=np.float32)

        except Exception:
            return np.zeros(target_dim or self.action_dim, dtype=np.float32)

    def _action_to_numeric(self, action: str) -> float:
        """Convert action string to numeric value"""
        action = action.lower()
        if action in ('long', 'buy'): return 1.0
        elif action in ('short', 'sell'): return -1.0
        else: return 0.0  # abstain, hold, etc.

    async def _analyze_individual_member_performance(
        self, member_idx: int, proposals: List[Any], confidences: List[Any], recent_success_rate: float
    ) -> Dict[str, Any]:
        try:
            perf = self.member_performance[member_idx]
            perf["proposals_made"] = int(perf.get("proposals_made", 0)) + 1

            cur_conf = float(np.clip(confidences[member_idx], 0.1, 1.0)) if member_idx < len(confidences) else 0.5
            old_conf = float(perf.get("avg_confidence", 0.5))
            perf["avg_confidence"] = float(old_conf * 0.9 + cur_conf * 0.1)

            # Proposal quality with dimension handling
            pq = 0.5
            if member_idx < len(proposals):
                proposal = proposals[member_idx]
                normalized_proposal = self._handle_proposal_dimension_mismatch(proposal)
                pq = await self._assess_proposal_quality(normalized_proposal, cur_conf)

            try:
                perf["quality_scores"].append(pq)
            except Exception:
                perf["quality_scores"] = deque([pq], maxlen=30)

            contribution = float((pq + cur_conf + recent_success_rate) / 3.0)
            perf["contribution_score"] = contribution

            if isinstance(perf.get("quality_scores"), deque) and len(perf["quality_scores"]) >= 5:
                recent = list(perf["quality_scores"])[-5:]
                consistency = float(np.clip(1.0 - float(np.std(recent)), 0.0, 1.0))
                perf["reliability_index"] = float((perf["avg_confidence"] + consistency) / 2.0)

            specialization = await self._calculate_member_specialization(member_idx, proposals)
            perf["specialization_score"] = specialization

            return {
                "contribution_score": contribution,
                "proposal_quality": pq,
                "confidence": cur_conf,
                "reliability": perf.get("reliability_index", 0.5),
                "specialization": specialization,
                "primary_factor": "contribution" if contribution > 0.7 else "reliability",
            }
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "individual_member_analysis")
            return {"contribution_score": 0.5, "proposal_quality": 0.5, "confidence": 0.5}

    async def _assess_proposal_quality(self, proposal: np.ndarray, confidence: float) -> float:
        try:
            strength = float(min(1.0, float(np.linalg.norm(proposal)) / 2.0))
            consistency = 0.5
            if self.proposal_history:
                prev = np.asarray(self.proposal_history[-1].get("proposal", proposal), dtype=np.float32)
                consistency = float(max(0.0, 1.0 - float(np.linalg.norm(proposal - prev)) / 2.0))

            # Regime appropriateness
            regime_adj = 0.5
            if self.market_regime == "volatile" and strength < 0.5:
                regime_adj = 0.8
            elif self.market_regime == "trending" and strength > 0.3:
                regime_adj = 0.8

            w = np.array([0.3, 0.3, 0.2, 0.2], dtype=np.float32)
            vals = np.array([strength, float(confidence), consistency, regime_adj], dtype=np.float32)
            return float(np.clip(float(np.dot(w[: len(vals)], vals[: len(vals)])), 0.0, 1.0))
        except Exception:
            return 0.5

    async def _calculate_member_specialization(self, member_idx: int, proposals: List[Any]) -> float:
        try:
            if member_idx >= len(proposals) or len(proposals) < 2:
                return 0.5
            mp = np.asarray(proposals[member_idx], dtype=np.float32)
            other = [np.asarray(p, dtype=np.float32) for i, p in enumerate(proposals) if i != member_idx and isinstance(p, (list, np.ndarray))]
            if not other:
                return 0.5
            dists: List[float] = []
            for q in other:
                if q.size == mp.size:
                    dists.append(float(np.linalg.norm(mp - q)))
            if dists:
                return float(min(1.0, float(np.mean(dists)) / 2.0))
            return 0.5
        except Exception:
            return 0.5

    async def _calculate_adaptive_weight_update(self, member_idx: int, analysis: Dict[str, Any]) -> float:
        try:
            contribution = float(analysis.get("contribution_score", 0.5))
            reliability = float(analysis.get("reliability", 0.5))
            specialization = float(analysis.get("specialization", 0.5))

            performance_factor = (contribution + reliability) / 2.0
            ideal = performance_factor / max(1, len(self.members))

            if specialization > 0.7:
                ideal *= 1.2

            lr = float(self.decision_intelligence.get("member_learning_rate", 0.05))
            delta = (ideal - float(self.weights[member_idx])) * lr

            # Regime bonus
            if self.market_regime == "volatile" and contribution > 0.8:
                delta *= 1.3
            elif self.market_regime == "trending" and specialization > 0.6:
                delta *= 1.2

            return float(np.clip(delta, -0.1, 0.1))
        except Exception:
            return 0.0

    async def _calculate_coordination_effectiveness(self, perf: Dict[str, Any]) -> float:
        try:
            updates = perf.get("member_updates", {})
            if not updates:
                return 0.5
            contrib = np.array([u.get("contribution_score", 0.5) for u in updates.values()], dtype=np.float32)
            spec = np.array([u.get("specialization", 0.5) for u in updates.values()], dtype=np.float32)
            avg_contrib = float(np.mean(contrib)) if contrib.size else 0.5
            div = float(min(1.0, float(np.std(contrib)) * 2.0)) if contrib.size > 1 else 0.0
            avg_spec = float(np.mean(spec)) if spec.size else 0.5
            score = 0.4 * avg_contrib + 0.3 * div + 0.3 * avg_spec
            self.coordination_analytics["coordination_effectiveness"] = float(np.clip(score, 0.0, 1.0))
            return self.coordination_analytics["coordination_effectiveness"]
        except Exception:
            return 0.5

    async def _update_voting_quality_metrics_comprehensive(self) -> Dict[str, Any]:
        try:
            updates: Dict[str, Any] = {"metric_changes": {}, "trend_analysis": {}, "quality_drivers": {}}

            # Gate effectiveness
            old_ge = float(self.voting_quality.get("gate_effectiveness", 0.5))
            new_ge = self._gate_passes / max(self._gate_attempts, 1)
            self.voting_quality["gate_effectiveness"] = float(new_ge)
            if abs(new_ge - old_ge) > 0.1:
                updates["metric_changes"]["gate_effectiveness"] = {
                    "old_value": old_ge,
                    "new_value": float(new_ge),
                    "trend": "improving" if new_ge > old_ge else "declining",
                }

            # Decision confidence (recent)
            if len(self.decision_history) >= 5:
                confs = [float(d.get("signal_strength", 0.5)) for d in list(self.decision_history)[-5:]]
                self.voting_quality["decision_confidence"] = float(np.mean(confs))

            # Member diversity
            if self.member_performance:
                contribs = [float(p.get("contribution_score", 0.5)) for p in self.member_performance.values()]
                if len(contribs) > 1:
                    self.voting_quality["member_diversity"] = float(min(1.0, float(np.std(contribs)) * 2.0))

            # Learning efficiency
            if len(self.learning_history) >= 10:
                rewards = [float(e.get("reward", 0.0)) for e in list(self.learning_history)[-10:]]
                if len(rewards) > 1:
                    try:
                        slope = float(np.polyfit(range(len(rewards)), rewards, 1)[0])
                    except Exception:
                        slope = 0.0
                    self.voting_quality["learning_efficiency"] = float(0.5 + np.tanh(slope * 10) * 0.5)

            # Adaptation rate (simple proxy)
            if self._step_count > 0:
                self.voting_quality["adaptation_rate"] = float(
                    self.arbiter_stats.get("weight_adaptations", 0) / self._step_count
                )
            return updates
        except Exception:
            return {"metric_changes": {}}

    async def _generate_intelligent_arbitration_recommendations(
        self, perf: Dict[str, Any], quality_updates: Dict[str, Any]
    ) -> List[str]:
        try:
            recs: List[str] = []
            coord = float(perf.get("coordination_effectiveness", 0.5))
            gate_eff = float(self.voting_quality.get("gate_effectiveness", 0.5))
            diversity = float(self.voting_quality.get("member_diversity", 0.5))
            learn_eff = float(self.voting_quality.get("learning_efficiency", 0.5))

            if coord < 0.4:
                recs.append("LOW COORDINATION: Rebalance members or run sync workshop")
            elif coord > 0.8:
                recs.append("HIGH COORDINATION: Maintain current approach")

            if gate_eff < 0.3:
                recs.append("GATE RESTRICTIVE: Loosen criteria or recalibrate thresholds")
            elif gate_eff > 0.8:
                recs.append("GATE PERMISSIVE: Tighten criteria to manage risk")

            if diversity < 0.3:
                recs.append("LOW DIVERSITY: Encourage specialization or diversify models")
            elif diversity > 0.8:
                recs.append("HIGH DIVERSITY: Balance with coordination checks")

            if learn_eff < 0.3:
                recs.append("LEARNING ISSUES: Review reward signals and LR")
            elif learn_eff > 0.8:
                recs.append("LEARNING EFFECTIVE: Consider advanced strategies")

            if self.market_regime == "volatile":
                recs.append("VOLATILE REGIME: Emphasize risk management and shorter horizons")
            elif self.market_regime == "trending":
                recs.append("TRENDING REGIME: Favor momentum and sized entries")
            elif self.market_regime == "noise":
                recs.append("NOISE REGIME: Reduce sizes and raise quality thresholds")

            if self._step_count < self.bootstrap_steps:
                remaining = self.bootstrap_steps - self._step_count
                recs.append(f"BOOTSTRAP MODE: {remaining} steps remaining for stabilization")

            return recs[:6] if recs else ["SYSTEM OPTIMAL: Arbitration within normal parameters"]
        except Exception as e:
            ctx = self.error_pinpointer.analyze_error(e, "arbitration_recommendations")
            return [f"Recommendation generation failed: {ctx}"]

    async def _generate_comprehensive_arbitration_thesis(
        self, perf: Dict[str, Any], recommendations: List[str]
    ) -> str:
        try:
            coord = float(perf.get("coordination_effectiveness", 0.5))
            gate_eff = float(self.voting_quality.get("gate_effectiveness", 0.5))
            learning_eff = float(self.voting_quality.get("learning_efficiency", 0.5))
            coordination_level = "HIGH" if coord > 0.7 else "MODERATE" if coord > 0.4 else "LOW"

            parts = [
                f"ARBITRATION: {coordination_level} coordination ({coord:.1%})",
                f"MEMBERS: {len(self.members)} experts; diversity {self.voting_quality.get('member_diversity', 0.5):.1%}",
                f"GATE: {'EFFECTIVE' if gate_eff > 0.6 else 'RESTRICTIVE' if gate_eff < 0.4 else 'MODERATE'} ({gate_eff:.1%} pass)",
                f"LEARNING: {'STRONG' if learning_eff > 0.7 else 'STABLE' if learning_eff > 0.4 else 'WEAK'} ({learning_eff:.1%})",
                f"MARKET: {self.market_regime.upper()} regime, vol {self.curr_vol:.2%}",
                f"WEIGHTS: {self.arbiter_stats.get('weight_adaptations', 0)} adaptations over {self._step_count} steps",
            ]
            td = self.arbiter_stats.get("total_decisions", 0)
            sd = self.arbiter_stats.get("successful_decisions", 0)
            sr = (sd / td) if td > 0 else 0.0
            parts.append(f"PERF: {sr:.1%} success across {td} decisions")

            pri = [r for r in recommendations if any(k in r for k in ("LOW", "HIGH", "CRITICAL", "URGENT"))]
            if pri:
                parts.append(f"ACTION ITEMS: {len(pri)} priority recs")
            return " | ".join(parts)
        except Exception as e:
            ctx = self.error_pinpointer.analyze_error(e, "arbitration_thesis_generation")
            return f"Arbitration thesis generation failed: {ctx}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str) -> None:
        try:
            s = self.smart_bus.set
            s("blended_action", results["blended_action"], module="StrategyArbiter", thesis=thesis)
            s(
                "alpha_weights",
                results["alpha_weights"],
                module="StrategyArbiter",
                thesis=f"Alpha weights: {len(results['alpha_weights'])} member allocations",
            )
            s(
                "member_weights",
                results["member_weights"],
                module="StrategyArbiter",
                thesis=f"Member weights: {len(results['member_weights'])} experts balanced",
            )
            s(
                "gate_decision",
                results["gate_decision"],
                module="StrategyArbiter",
                thesis=f"Gate decision: {results['gate_decision'].get('decision', 'unknown')}",
            )
            s(
                "voting_quality",
                results["voting_quality"],
                module="StrategyArbiter",
                thesis=f"Voting quality: {len(results['voting_quality'])} metrics",
            )
            s(
                "member_performance",
                results["member_performance"],
                module="StrategyArbiter",
                thesis=f"Member performance: {len(results['member_performance'])} profiles",
            )
            s(
                "decision_statistics",
                results["decision_statistics"],
                module="StrategyArbiter",
                thesis=f"Decision stats: {results['decision_statistics'].get('total_decisions', 0)} decisions",
            )
            s(
                "proposal_analysis",
                results["proposal_analysis"],
                module="StrategyArbiter",
                thesis="Recent member proposals evaluated",
            )
            s(
                "arbiter_recommendations",
                results["arbiter_recommendations"],
                module="StrategyArbiter",
                thesis=f"Recommendations: {len(results['arbiter_recommendations'])}",
            )
            s(
                "expert_performance",
                results["expert_performance"],
                module="StrategyArbiter",
                thesis=f"Expert performance: {len(results['expert_performance'])} experts tracked",
            )

            # Keep universe aliases fresh each cycle
            try:
                inst_list = list(results.get("instruments", list(getattr(self, "instruments", []))))
                s("instruments", inst_list, module="StrategyArbiter", thesis="Instrument universe (canonical)")

            except Exception:
                pass

            # Publish a convenient weights map
            try:
                names: List[str] = []
                for i, m in enumerate(self.members):
                    nm = None
                    try:
                        nm = getattr(m, "name", None) or getattr(m, "module_name", None)
                    except Exception:
                        nm = None
                    names.append(nm if isinstance(nm, str) and nm else f"member_{i}")
                weights_list = results.get("member_weights", self.weights.tolist())
                weights_map = {names[i]: float(weights_list[i]) for i in range(min(len(names), len(weights_list)))}
                s(
                    "strategy_weights",
                    {
                        "by_member": weights_map,
                        "members": names,
                        "weights": weights_list,
                        "timestamp": dt.datetime.utcnow().isoformat(),
                    },
                    module="StrategyArbiter",
                    thesis="Current strategy/member weights",
                )
            except Exception:
                pass
            # FIX #2: Publish decision_id for kernel coordination
            if results.get("decision_id"):
                s("arbiter_decision_id", results["decision_id"], module="StrategyArbiter",
                  thesis=f"Arbiter decision ID: {results['decision_id']}")
            
            # FIX: Publish namespaced keys for VotingKernel coordination (REAL DATA, NO FALLBACKS)
            s("arbiter_instrument_signals", results.get("instrument_signals", {}), module="StrategyArbiter",
              thesis=f"Namespaced instrument signals for VotingKernel: {len(results.get('instrument_signals', {}))} instruments")
            
            gate_decision_data = results.get("gate_decision", {})
            s("arbiter_gate_decision", gate_decision_data, module="StrategyArbiter",
              thesis=f"Namespaced gate decision for VotingKernel: {gate_decision_data.get('decision', 'unknown')}")
            
            # Extract gate breakdown from gate decision
            gate_breakdown = {
                "decision": gate_decision_data.get("decision", "unknown"),
                "criteria_met": gate_decision_data.get("criteria_met", 0),
                "total_criteria": gate_decision_data.get("total_criteria", 5),
                "pass_rate": gate_decision_data.get("criteria_met", 0) / max(gate_decision_data.get("total_criteria", 5), 1)
            }
            s("arbiter_gate_breakdown", gate_breakdown, module="StrategyArbiter",
              thesis=f"Gate decision breakdown for VotingKernel: {gate_breakdown['criteria_met']}/{gate_breakdown['total_criteria']} criteria met")
        except Exception as e:
            ctx = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {ctx}")

    # ────────────────────────────
    # PUBLIC INTERFACE
    # ────────────────────────────
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        try:
            q = float(self.voting_quality_metrics.get("overall_quality_score", 0.5))
            cons = float(self.member_analytics.get("performance_consistency", 0.5))
            regime = float(self.regime_analytics.get("current_regime_fit", 0.5))
            gate = self._gate_passes / max(self._gate_attempts, 1)
            consensus_strength = 1.0 - float(np.std(self.weights)) if len(self.weights) > 1 else 0.5
            conf = 0.3 * q + 0.25 * cons + 0.2 * regime + 0.15 * gate + 0.1 * consensus_strength
            return float(np.clip(conf, 0.1, 0.95))
        except Exception:
            return 0.4

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """
        Propose arbitration action and publish per-instrument intensities.
        Mirrors process() path for signal publication; returns friendly diagnostics.
        """
        try:
            market_data = await self._get_comprehensive_market_data()
            await self._update_market_state_comprehensive(market_data)

            blended_proposal = self._compute_blended_proposal(market_data)
            signals = self._map_action_to_instrument_signals(blended_proposal)
            self._publish_instrument_signals(signals)
            self.smart_bus.set(
                "instrument_signals",
                signals,
                module="StrategyArbiter",
                thesis=f"Per-instrument intensities for {len(signals)} instruments",
            )

            voting_quality = self.voting_quality_metrics.get("overall_quality_score", 0.5)
            member_coord = self.coordination_analytics.get("coordination_effectiveness", 0.5)
            proposal_strength = float(np.linalg.norm(blended_proposal))

            if voting_quality < 0.3:
                action_type, signal_strength, reasoning = (
                    "rebalance_members",
                    0.8,
                    f"Poor voting quality ({voting_quality:.3f}) requires member rebalancing",
                )
            elif member_coord < 0.4:
                action_type, signal_strength, reasoning = (
                    "improve_coordination",
                    0.6,
                    f"Low member coordination ({member_coord:.3f}) needs attention",
                )
            elif proposal_strength > 0.7:
                action_type, signal_strength, reasoning = (
                    "execute_proposal",
                    min(proposal_strength, 0.9),
                    f"Strong blended proposal (strength: {proposal_strength:.3f})",
                )
            else:
                action_type, signal_strength, reasoning = ("monitor", 0.3, "Normal arbitration state — continue monitoring")

            return {
                "action": action_type,
                "signal_strength": float(signal_strength),
                "reasoning": reasoning,
                "arbitration_metrics": {
                    "voting_quality": float(voting_quality),
                    "member_coordination": float(member_coord),
                    "proposal_strength": float(proposal_strength),
                    "member_count": len(self.members),
                    "weight_distribution": self.weights.tolist() if hasattr(self.weights, "tolist") else [],
                },
                "blended_proposal": blended_proposal.tolist(),
                "published_signals": signals,
                "confidence": await self.calculate_confidence({}, **inputs),
            }
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {"action": "abstain", "signal_strength": 0.0, "reasoning": f"Arbitration error: {str(e)}", "confidence": 0.1}

    # ────────────────────────────
    # REPORTING & HEALTH
    # ────────────────────────────
    def get_observation_components(self) -> np.ndarray:
        try:
            features: List[float] = [
                float(self._gate_passes / max(self._gate_attempts, 1)),
                float(self.curr_vol),
                float(self._baseline),
                float(self.voting_quality["avg_consensus"]),
                float(self.voting_quality["decision_confidence"]),
                float(self.voting_quality["member_diversity"]),
                float(len(self.decision_history) / 200),
                float(self.arbiter_stats["weight_adaptations"] / max(self._step_count, 1)),
            ]
            features.extend(self._normalize_weights(self.weights).tolist())
            obs = np.array(features, dtype=np.float32)
            if np.any(~np.isfinite(obs)):
                self.logger.error(f"Invalid arbitration observation: {obs}")
                obs = np.nan_to_num(obs, nan=0.5)
            return obs
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "observation_generation")
            default_features = [0.5, 0.02, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0]
            default_features.extend([1.0 / max(1, len(self.members))] * max(1, len(self.members)))
            return np.array(default_features, dtype=np.float32)

    def get_health_metrics(self) -> Dict[str, Any]:
        return {
            "module_name": "StrategyArbiter",
            "status": "disabled" if self.is_disabled else "healthy",
            "error_count": int(self.error_count),
            "circuit_breaker_threshold": int(self.circuit_breaker_threshold),
            "total_decisions": int(self.arbiter_stats.get("total_decisions", 0)),
            "successful_decisions": int(self.arbiter_stats.get("successful_decisions", 0)),
            "gate_pass_rate": float(self._gate_passes / max(self._gate_attempts, 1)),
            "weight_adaptations": int(self.arbiter_stats.get("weight_adaptations", 0)),
            "learning_baseline": float(self._baseline),
            "coordination_effectiveness": float(self.voting_quality.get("member_diversity", 0.5)),
            "members_count": int(len(self.members)),
            "step_count": int(self._step_count),
            "bootstrap_complete": bool(self._step_count >= self.bootstrap_steps),
            "market_regime": str(self.market_regime),
            "consensus_score": float(self.smart_bus.get("consensus_score", "StrategyArbiter") or 0.5),
            "session_duration": (
                dt.datetime.now() - dt.datetime.fromisoformat(self.arbiter_stats["session_start"])
            ).total_seconds()
            / 3600.0,
        }

    def _get_health_metrics(self) -> Dict[str, Any]:
        return self.get_health_metrics()

    def get_arbiter_report(self) -> str:
        decision_conf = self.voting_quality["decision_confidence"]
        if decision_conf > 0.8:
            quality_status = "[OK] EXCELLENT"
        elif decision_conf > 0.6:
            quality_status = "[FAST] GOOD"
        elif decision_conf > 0.4:
            quality_status = "[WARN] FAIR"
        else:
            quality_status = "[ALERT] POOR"

        gate_rate = self.voting_quality["gate_effectiveness"]
        if gate_rate > 0.7:
            gate_status = "[GREEN] EFFECTIVE"
        elif gate_rate > 0.4:
            gate_status = "[YELLOW] MODERATE"
        else:
            gate_status = "[RED] RESTRICTIVE"

        member_lines: List[str] = []
        for member_idx, perf in list(self.member_performance.items())[:5]:
            if member_idx < len(self.weights):
                weight = float(self.weights[member_idx])
                contribution = float(perf.get("contribution_score", 0.5))
                reliability = float(perf.get("reliability_index", 0.5))
                emoji = "🌟" if contribution > 0.7 else "[FAST]" if contribution > 0.5 else "[WARN]"
                member_lines.append(
                    f"  {emoji} Member {member_idx}: Weight {weight:.3f}, Contrib {contribution:.1%}, Rel {reliability:.1%}"
                )

        learning_eff = float(self.voting_quality.get("learning_efficiency", 0.5))
        learning_status = "[CHART] Strong" if learning_eff > 0.7 else "→ Stable" if learning_eff > 0.4 else "📉 Weak"

        return f"""
🏛️ STRATEGY ARBITER v3.1
═══════════════════════════════════════════════════════════════
[TARGET] Decision Quality: {quality_status} ({decision_conf:.1%})
🚪 Gate Status: {gate_status} ({gate_rate:.1%})
[STATS] Consensus Level: {self.voting_quality['avg_consensus']:.1%}
[CHART] Learning Status: {learning_status} ({learning_eff:.1%})

[BALANCE] Committee Overview:
• Total Members: {len(self.members)}
• Action Dimensions: {self.action_dim}
• Current Step: {self._step_count}
• Bootstrap Mode: {'[OK] Active' if self._step_count < self.bootstrap_steps else '[FAIL] Complete'}
• Learning Baseline: {self._baseline:.3f}

[STATS] Market Context:
• Regime: {self.market_regime.title()}
• Session: {self.market_session.title()}
• Volatility: {self.curr_vol:.2%}

[TARGET] Voting Quality:
• Decision Confidence: {self.voting_quality['decision_confidence']:.1%}
• Average Consensus: {self.voting_quality['avg_consensus']:.1%}
• Member Diversity: {self.voting_quality['member_diversity']:.1%}
• Collusion Risk: {self.voting_quality['collusion_risk']:.1%}
• Gate Effectiveness: {self.voting_quality['gate_effectiveness']:.1%}
• Proposal Quality: {self.voting_quality['proposal_quality']:.1%}
• Learning Efficiency: {self.voting_quality['learning_efficiency']:.1%}

[CHART] Performance:
• Total Decisions: {self.arbiter_stats['total_decisions']}
• Successful Decisions: {self.arbiter_stats['successful_decisions']}
• Success Rate: {(self.arbiter_stats['successful_decisions'] / max(self.arbiter_stats['total_decisions'], 1)):.1%}
• Weight Adaptations: {self.arbiter_stats['weight_adaptations']}
• Collusion Events: {self.arbiter_stats['collusion_detected']}
• Learning Trend: {self._calculate_learning_trend().title()}

🚪 Gate Intelligence:
• Attempts: {self._gate_attempts} | Passes: {self._gate_passes} | Pass Rate: {(self._gate_passes / max(self._gate_attempts, 1)):.1%}
• Criteria Weights: {', '.join([f'{w:.2f}' for w in self.gate_intelligence['criteria_weights']])}
• Adaptive Threshold: {'[OK] Enabled' if self.gate_intelligence['adaptive_threshold'] else '[FAIL] Disabled'}

👥 Top Performing Members:
{chr(10).join(member_lines) if member_lines else "  📭 No member performance data available"}

[TOOL] Config:
• Adapt Rate: {self.adapt_rate:.4f} | Min Confidence: {self.min_confidence:.2f} | Bootstrap: {self.bootstrap_steps}
• REINFORCE Beta: {self._baseline_beta:.3f} | LR: {self.REINFORCE_LR:.4f}

[TOOL] System Health:
• Errors: {self.error_count}/{self.circuit_breaker_threshold} | Status: {'[ALERT] DISABLED' if self.is_disabled else '[OK] OPERATIONAL'}
• Session Duration: {(dt.datetime.now() - dt.datetime.fromisoformat(self.arbiter_stats['session_start'])).total_seconds() / 3600:.1f}h
"""

    def _get_comprehensive_arbiter_stats(self) -> Dict[str, Any]:
        return {
            **self.arbiter_stats,
            "current_weights": self.weights.tolist(),
            "last_alpha": self.last_alpha.tolist() if self.last_alpha is not None else None,
            "step_count": self._step_count,
            "gate_passes": self._gate_passes,
            "gate_attempts": self._gate_attempts,
            "baseline_estimate": self._baseline,
            "market_regime": self.market_regime,
            "market_session": self.market_session,
            "volatility": self.curr_vol,
            "bootstrap_complete": self._step_count >= self.bootstrap_steps,
            "learning_trend": self._calculate_learning_trend(),
        }

    def _calculate_learning_trend(self) -> str:
        try:
            if len(self.learning_history) < 5:
                return "insufficient_data"
            rewards = [float(e.get("reward", 0.0)) for e in list(self.learning_history)[-5:]]
            try:
                slope = float(np.polyfit(range(len(rewards)), rewards, 1)[0])
            except Exception:
                slope = 0.0
            if slope > 0.01:
                return "improving"
            if slope < -0.01:
                return "declining"
            return "stable"
        except Exception:
            return "unknown"

    def _get_recent_gate_decision(self) -> Dict[str, Any]:
        try:
            if self.gate_decisions:
                return dict(self.gate_decisions[-1])
            return {
                "decision": "unknown",
                "criteria_met": 0,
                "total_criteria": 5,
                "timestamp": dt.datetime.now().isoformat(),
            }
        except Exception:
            return {"decision": "unknown"}

    def _get_member_performance_summary(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {}
            for idx, perf in self.member_performance.items():
                qs = perf.get("quality_scores", [0.5])
                recent_quality = float(np.mean(list(qs)[-5:])) if isinstance(qs, deque) and len(qs) > 0 else 0.5
                out[f"member_{int(idx)}"] = {
                    "contribution_score": float(perf.get("contribution_score", 0.5)),
                    "reliability_index": float(perf.get("reliability_index", 0.5)),
                    "specialization_score": float(perf.get("specialization_score", 0.5)),
                    "avg_confidence": float(perf.get("avg_confidence", 0.5)),
                    "proposals_made": int(perf.get("proposals_made", 0)),
                    "recent_quality": float(recent_quality),
                }
            return out
        except Exception:
            return {}

    def _generate_expert_performance_output(self) -> Dict[str, Any]:
        """
        Generate expert_performance output required by module contract.
        Maps member names to their normalized performance scores (0.0-1.0).
        """
        try:
            expert_performance: Dict[str, Any] = {}

            # Generate member name to performance mapping
            for idx, perf in self.member_performance.items():
                # Get member name
                member_name = f"member_{int(idx)}"
                try:
                    if idx < len(self.members):
                        member = self.members[idx]
                        nm = getattr(member, "name", None) or getattr(member, "module_name", None)
                        if isinstance(nm, str) and nm:
                            member_name = nm
                except Exception:
                    pass

                # Calculate normalized performance score (0.0-1.0)
                contribution = float(perf.get("contribution_score", 0.5))
                reliability = float(perf.get("reliability_index", 0.5))
                specialization = float(perf.get("specialization_score", 0.5))

                # Weighted performance score
                performance_score = (
                    0.4 * contribution +
                    0.4 * reliability +
                    0.2 * specialization
                )

                expert_performance[member_name] = float(np.clip(performance_score, 0.0, 1.0))

            # Add aggregate metrics
            expert_performance["_metadata"] = {
                "total_experts": len(self.members),
                "active_experts": len([p for p in self.member_performance.values() if p.get("proposals_made", 0) > 0]),
                "avg_performance": float(np.mean(list(expert_performance.values()))) if expert_performance else 0.5,
                "performance_std": float(np.std(list(expert_performance.values()))) if len(expert_performance) > 1 else 0.0,
                "timestamp": dt.datetime.utcnow().isoformat(),
                "arbiter_version": "3.1.0"
            }

            return expert_performance

        except Exception as e:
            # Return safe fallback expert_performance
            self.logger.warning(f"Failed to generate expert_performance: {e}")
            return {
                "_metadata": {
                    "total_experts": len(getattr(self, "members", [])),
                    "active_experts": 0,
                    "avg_performance": 0.5,
                    "performance_std": 0.0,
                    "timestamp": dt.datetime.utcnow().isoformat(),
                    "arbiter_version": "3.1.0",
                    "error": str(e)
                }
            }

    def _get_recent_proposal_analysis(self) -> Dict[str, Any]:
        try:
            if not self.proposal_history:
                return {"status": "no_proposals"}
            recent = list(self.proposal_history)[-5:]
            qualities = [float(p.get("quality", 0.5)) for p in recent]
            analysis = {
                "proposal_count": len(recent),
                "avg_quality": float(np.mean(qualities)) if qualities else 0.5,
                "quality_trend": "stable",
                "diversity_score": 0.5,
            }
            if len(qualities) >= 3:
                try:
                    slope = float(np.polyfit(range(len(qualities)), qualities, 1)[0])
                except Exception:
                    slope = 0.0
                analysis["quality_trend"] = "improving" if slope > 0.05 else "declining" if slope < -0.05 else "stable"
            return analysis
        except Exception:
            return {"status": "analysis_error"}

    # ────────────────────────────
    # LEARNING LOOP
    # ────────────────────────────
    def update_weights(self, reward: float) -> None:
        """Enhanced REINFORCE weight update with comprehensive tracking."""
        if self.last_alpha is None:
            return
        try:
            # Baseline & advantage
            self._baseline = float(self._baseline_beta * self._baseline + (1 - self._baseline_beta) * reward)
            advantage = float(reward - self._baseline)

            rm = self.market_adaptation["regime_multipliers"].get(self.market_regime, {})
            effective_lr = float(self.adapt_rate * rm.get("confidence_boost", 1.0))

            grad = advantage * (self.last_alpha - self.weights)
            old = self.weights.copy()
            self.weights = self._normalize_weights(np.maximum(self.weights + effective_lr * grad, 0.01))

            # Learning telemetry
            change = float(np.linalg.norm(self.weights - old))
            self.learning_history.append(
                {
                    "timestamp": dt.datetime.now().isoformat(),
                    "reward": float(reward),
                    "advantage": float(advantage),
                    "baseline": float(self._baseline),
                    "weight_change": change,
                    "regime": self.market_regime,
                }
            )
            if change > 0.05:
                self.arbiter_stats["weight_adaptations"] += 1
                self.logger.info(
                    format_operator_message(
                        icon="[BALANCE]",
                        message="Significant weight adaptation",
                        reward=f"{reward:+.3f}",
                        advantage=f"{advantage:+.3f}",
                        change=f"{change:.3f}",
                        regime=self.market_regime,
                    )
                )
            if reward > 0:
                self.arbiter_stats["successful_decisions"] += 1

            # Perf metrics
            self._update_performance_metric("weight_adaptation_magnitude", change)
            self._update_performance_metric("learning_advantage", advantage)
            self._update_performance_metric("baseline_estimate", float(self._baseline))
        except Exception as e:
            ctx = self.error_pinpointer.analyze_error(e, "weight_update")
            self.logger.error(f"Weight update failed: {ctx}")

    def _update_performance_metric(self, metric_name: str, value: float) -> None:
        try:
            if hasattr(self, "performance_tracker") and self.performance_tracker:
                self.performance_tracker.record_metric("StrategyArbiter", metric_name, float(value), True)
            if not hasattr(self, "_metric_history"):
                self._metric_history = defaultdict(lambda: deque(maxlen=100))
            self._metric_history[metric_name].append({"timestamp": dt.datetime.now().isoformat(), "value": float(value)})
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.warning(f"Performance metric update failed for {metric_name}: {e}")

    # ────────────────────────────
    # HEALTH & ERRORS
    # ────────────────────────────
    def get_health_status(self) -> Dict[str, Any]:
        return {
            "module_name": "StrategyArbiter",
            "status": "disabled" if self.is_disabled else "healthy",
            "metrics": self._get_health_metrics(),
            "alerts": self._generate_health_alerts(),
            "recommendations": self._generate_health_recommendations(),
        }

    def _generate_health_alerts(self) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        if self.is_disabled:
            alerts.append(
                {"severity": "critical", "message": "StrategyArbiter disabled due to errors", "action": "Inspect logs and restart"}
            )
        if self.error_count > 2:
            alerts.append({"severity": "warning", "message": f"Elevated error count: {self.error_count}", "action": "Monitor stability"})
        gate_rate = self._gate_passes / max(self._gate_attempts, 1)
        if gate_rate < 0.2:
            alerts.append({"severity": "warning", "message": f"Very low gate pass rate: {gate_rate:.1%}", "action": "Review gate criteria"})
        elif gate_rate > 0.9:
            alerts.append({"severity": "info", "message": f"Very high gate pass rate: {gate_rate:.1%}", "action": "Tighten criteria"})
        learning_efficiency = self.voting_quality.get("learning_efficiency", 0.5)
        if learning_efficiency < 0.3:
            alerts.append(
                {"severity": "warning", "message": f"Poor learning efficiency: {learning_efficiency:.1%}", "action": "Review rewards & LR"}
            )
        member_diversity = self.voting_quality.get("member_diversity", 0.5)
        if member_diversity < 0.2:
            alerts.append({"severity": "info", "message": f"Low member diversity: {member_diversity:.1%}", "action": "Encourage specialization"})
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        recs: List[str] = []
        if self.is_disabled:
            recs.append("Restart StrategyArbiter module after investigating errors")
        if self._step_count < self.bootstrap_steps:
            recs.append(f"Bootstrap: {self.bootstrap_steps - self._step_count} steps to stabilization")
        gate_rate = self._gate_passes / max(self._gate_attempts, 1)
        if gate_rate < 0.3:
            recs.append("Gate too restrictive — consider loosening criteria")
        elif gate_rate > 0.8:
            recs.append("Gate too permissive — consider tightening criteria")
        if len(getattr(self, "learning_history", [])) < 10:
            recs.append("Limited learning history — continue operations to build patterns")
        if self._calculate_learning_trend() == "declining":
            recs.append("Learning trend declining — review reward signal quality")
        adaptation_rate = self.arbiter_stats.get("weight_adaptations", 0) / max(self._step_count, 1)
        if adaptation_rate > 0.2:
            recs.append("High weight adaptation frequency — ensure stability")
        elif adaptation_rate < 0.05 and self._step_count > self.bootstrap_steps:
            recs.append("Low weight adaptation — consider increasing sensitivity")
        return recs or ["StrategyArbiter operating within normal parameters"]

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        self.error_count += 1
        ctx = self.error_pinpointer.analyze_error(error, "StrategyArbiter")
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(
                format_operator_message(
                    icon="[ALERT]",
                    message="Strategy Arbiter disabled due to repeated errors",
                    error_count=self.error_count,
                    threshold=self.circuit_breaker_threshold,
                )
            )
        self.performance_tracker.record_metric(
            "StrategyArbiter", "process_time_ms", (time.time() - start_time) * 1000.0, False
        )
        # Ensure all contract-required outputs are present even on error
        try:
            init_payload = getattr(self, "_init_payload", None)
            if not init_payload and hasattr(self, "smart_bus"):
                init_payload = self.smart_bus.get("strategy_arbiter_initialization", "StrategyArbiter") or {}
        except Exception:
            init_payload = {}

        try:
            member_names: List[str] = []
            for i, m in enumerate(self.members):
                nm = None
                try:
                    nm = getattr(m, "name", None) or getattr(m, "module_name", None)
                except Exception:
                    nm = None
                member_names.append(nm if isinstance(nm, str) and nm else f"member_{i}")
            weights_list = self.weights.tolist()
            strategy_weights = {
                "by_member": {member_names[i]: float(weights_list[i]) for i in range(min(len(member_names), len(weights_list)))},
                "members": member_names,
                "weights": weights_list,
                "timestamp": dt.datetime.utcnow().isoformat(),
            }
        except Exception:
            strategy_weights = {"by_member": {}, "members": [], "weights": [], "timestamp": dt.datetime.utcnow().isoformat()}

        try:
            arbiter_decision_id = None
            if hasattr(self, "smart_bus"):
                arbiter_decision_id = self.smart_bus.get('kernel_decision_id', 'StrategyArbiter')
        except Exception:
            arbiter_decision_id = None

        return {
            "blended_action": [],
            "alpha_weights": [],
            "member_weights": self.weights.tolist(),
            "gate_decision": {"decision": "error", "error_context": str(ctx)},
            "voting_quality": {"error": str(ctx)},
            "member_performance": {},
            "decision_statistics": {"error": str(ctx)},
            "proposal_analysis": {"error": str(ctx)},
            "arbiter_recommendations": ["Investigate strategy arbiter errors"],
            "health_metrics": {"status": "error", "error_context": str(ctx)},
            "instrument_signals": {},
            "instruments": list(getattr(self, "instruments", [])),
            "strategy_weights": strategy_weights,
            "strategy_arbiter_initialization": init_payload,
            "arbiter_decision_id": arbiter_decision_id,
            "expert_performance": {"_metadata": {"error": str(ctx), "timestamp": dt.datetime.utcnow().isoformat()}},
            "_thesis": f"StrategyArbiter error: {ctx}",
        }

    def _get_safe_market_defaults(self) -> Dict[str, Any]:
        return {
            "market_context": {},
            "recent_trades": [],
            "current_positions": [],
            "member_proposals": [],
            "proposal_vectors": [],
            "member_confidences": [],
            "consensus_score": 0.5,
            "collusion_score": 0.0,
            "horizon_alignment": {},
            "volatility_data": {},
            "market_regime": "unknown",
            "session_data": {},
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        # Ensure all contract-required outputs are present even when disabled
        try:
            init_payload = getattr(self, "_init_payload", None)
            if not init_payload and hasattr(self, "smart_bus"):
                init_payload = self.smart_bus.get("strategy_arbiter_initialization", "StrategyArbiter") or {}
        except Exception:
            init_payload = {}

        try:
            member_names: List[str] = []
            for i, m in enumerate(self.members):
                nm = None
                try:
                    nm = getattr(m, "name", None) or getattr(m, "module_name", None)
                except Exception:
                    nm = None
                member_names.append(nm if isinstance(nm, str) and nm else f"member_{i}")
            weights_list = self.weights.tolist()
            strategy_weights = {
                "by_member": {member_names[i]: float(weights_list[i]) for i in range(min(len(member_names), len(weights_list)))},
                "members": member_names,
                "weights": weights_list,
                "timestamp": dt.datetime.utcnow().isoformat(),
            }
        except Exception:
            strategy_weights = {"by_member": {}, "members": [], "weights": [], "timestamp": dt.datetime.utcnow().isoformat()}

        try:
            arbiter_decision_id = None
            if hasattr(self, "smart_bus"):
                arbiter_decision_id = self.smart_bus.get('kernel_decision_id', 'StrategyArbiter')
        except Exception:
            arbiter_decision_id = None

        return {
            "blended_action": [],
            "alpha_weights": [],
            "member_weights": self.weights.tolist(),
            "gate_decision": {"decision": "disabled"},
            "voting_quality": {"status": "disabled"},
            "member_performance": {},
            "decision_statistics": {"status": "disabled"},
            "proposal_analysis": {"status": "disabled"},
            "arbiter_recommendations": ["Restart strategy arbiter system"],
            "health_metrics": {"status": "disabled", "reason": "circuit_breaker_triggered"},
            "instrument_signals": {},
            "instruments": list(getattr(self, "instruments", [])),
            "universe": list(getattr(self, "instruments", [])),
            "watched_instruments": list(getattr(self, "instruments", [])),
            "strategy_weights": strategy_weights,
            "strategy_arbiter_initialization": init_payload,
            "arbiter_decision_id": arbiter_decision_id,
            "expert_performance": {"_metadata": {"status": "disabled", "timestamp": dt.datetime.utcnow().isoformat()}},
            "_thesis": "StrategyArbiter disabled due to circuit breaker",
        }

    # ────────────────────────────
    # STATE / HOT-RELOAD
    # ────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        return {
            "module_info": {"name": "StrategyArbiter", "version": "3.1.0", "last_updated": dt.datetime.now().isoformat()},
            "configuration": {
                "action_dim": int(self.action_dim),
                "adapt_rate": float(self.adapt_rate),
                "min_confidence": float(self.min_confidence),
                "bootstrap_steps": int(self.bootstrap_steps),
                "debug": bool(self.debug),
            },
            "arbitration_state": {
                "weights": self.weights.tolist(),
                "last_alpha": self.last_alpha.tolist() if self.last_alpha is not None else None,
                "baseline": float(self._baseline),
                "step_count": int(self._step_count),
                "gate_passes": int(self._gate_passes),
                "gate_attempts": int(self._gate_attempts),
            },
            "market_state": {
                "curr_vol": float(self.curr_vol),
                "market_regime": str(self.market_regime),
                "market_session": str(self.market_session),
                "market_context": dict(self.market_context),
            },
            "intelligence_state": {
                "decision_intelligence": dict(self.decision_intelligence),
                "gate_intelligence": dict(self.gate_intelligence),
                "market_adaptation": dict(self.market_adaptation),
                "voting_quality": dict(self.voting_quality),
            },
            "performance_state": {
                "member_performance": {
                    k: {
                        "contribution_score": float(v.get("contribution_score", 0.5)),
                        "reliability_index": float(v.get("reliability_index", 0.5)),
                        "specialization_score": float(v.get("specialization_score", 0.5)),
                        "avg_confidence": float(v.get("avg_confidence", 0.5)),
                        "proposals_made": int(v.get("proposals_made", 0)),
                        "quality_scores": [],  # keep small; consumers shouldn't restore long deques
                    }
                    for k, v in self.member_performance.items()
                },
                "arbiter_stats": dict(self.arbiter_stats),
            },
            "history_state": {
                "decision_history": list(self.decision_history)[-50:],
                "learning_history": list(self.learning_history)[-30:],
                "gate_decisions": list(self.gate_decisions)[-20:],
                "proposal_history": list(self.proposal_history)[-20:],
                "trace": self._trace[-20:] if self._trace else [],
            },
            "error_state": {"error_count": int(self.error_count), "is_disabled": bool(self.is_disabled)},
            "performance_metrics": self.get_health_metrics(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        try:
            cfg = state.get("configuration", {})
            self.action_dim = int(cfg.get("action_dim", self.action_dim))
            self.adapt_rate = float(cfg.get("adapt_rate", self.adapt_rate))
            self.min_confidence = float(cfg.get("min_confidence", self.min_confidence))
            self.bootstrap_steps = int(cfg.get("bootstrap_steps", self.bootstrap_steps))
            self.debug = bool(cfg.get("debug", self.debug))

            arb = state.get("arbitration_state", {})
            self.weights = np.asarray(arb.get("weights", self.weights.tolist()), dtype=np.float32)
            la = arb.get("last_alpha")
            if la is not None:
                self.last_alpha = np.asarray(la, dtype=np.float32)
            self._baseline = float(arb.get("baseline", self._baseline))
            self._step_count = int(arb.get("step_count", self._step_count))
            self._gate_passes = int(arb.get("gate_passes", self._gate_passes))
            self._gate_attempts = int(arb.get("gate_attempts", self._gate_attempts))

            mkt = state.get("market_state", {})
            self.curr_vol = float(mkt.get("curr_vol", self.curr_vol))
            self.market_regime = str(mkt.get("market_regime", self.market_regime))
            self.market_session = str(mkt.get("market_session", self.market_session))
            self.market_context = dict(mkt.get("market_context", self.market_context))

            intel = state.get("intelligence_state", {})
            self.decision_intelligence.update(intel.get("decision_intelligence", {}))
            self.gate_intelligence.update(intel.get("gate_intelligence", {}))
            self.market_adaptation.update(intel.get("market_adaptation", {}))
            self.voting_quality.update(intel.get("voting_quality", {}))

            perf = state.get("performance_state", {})
            mp = perf.get("member_performance", {})
            self.member_performance.clear()
            for k, v in mp.items():
                idx = int(k)
                self.member_performance[idx] = {
                    "proposals_made": int(v.get("proposals_made", 0)),
                    "successful_proposals": int(v.get("successful_proposals", 0)),
                    "avg_confidence": float(v.get("avg_confidence", 0.5)),
                    "contribution_score": float(v.get("contribution_score", 0.5)),
                    "reliability_index": float(v.get("reliability_index", 0.5)),
                    "specialization_score": float(v.get("specialization_score", 0.5)),
                    "recent_performance": deque(maxlen=20),
                    "weight_evolution": deque(maxlen=50),
                    "quality_scores": deque(v.get("quality_scores", []), maxlen=30),
                }

            self.arbiter_stats.update(perf.get("arbiter_stats", {}))

            hist = state.get("history_state", {})
            self.decision_history = deque(hist.get("decision_history", []), maxlen=200)
            self.learning_history = deque(hist.get("learning_history", []), maxlen=100)
            self.gate_decisions = deque(hist.get("gate_decisions", []), maxlen=150)
            self.proposal_history = deque(hist.get("proposal_history", []), maxlen=100)
            self._trace = list(hist.get("trace", []))

            err = state.get("error_state", {})
            self.error_count = int(err.get("error_count", self.error_count))
            self.is_disabled = bool(err.get("is_disabled", self.is_disabled))

            self.logger.info(
                format_operator_message(
                    icon="[RELOAD]",
                    message="Strategy Arbiter state restored",
                    members=len(self.members),
                    action_dim=self.action_dim,
                    step_count=self._step_count,
                    total_decisions=self.arbiter_stats.get("total_decisions", 0),
                )
            )
        except Exception as e:
            ctx = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {ctx}")

    # ────────────────────────────
    # RESET & TEARDOWN
    # ────────────────────────────
    def reset(self) -> None:
        super().reset()
        self._baseline = 0.0
        self.last_alpha = None
        self._gate_passes = 0
        self._gate_attempts = 0
        self._step_count = 0
        self._trace.clear()
        self.decision_history.clear()
        self.proposal_history.clear()
        self.gate_decisions.clear()
        self.learning_history.clear()
        self.member_performance.clear()
        self.voting_quality.update(
            {
                "avg_consensus": 0.5,
                "collusion_risk": 0.0,
                "gate_effectiveness": 0.5,
                "member_diversity": 0.5,
                "decision_confidence": 0.5,
                "proposal_quality": 0.5,
                "learning_efficiency": 0.5,
                "adaptation_rate": 0.0,
            }
        )
        self.arbiter_stats = {
            "total_decisions": 0,
            "successful_decisions": 0,
            "weight_adaptations": 0,
            "consensus_failures": 0,
            "collusion_detected": 0,
            "gate_pass_rate": 0.0,
            "avg_proposal_quality": 0.5,
            "learning_convergence": 0.0,
            "member_coordination": 0.5,
            "decision_latency": 0.0,
            "session_start": dt.datetime.now().isoformat(),
        }
        self.error_count = 0
        self.is_disabled = False
        self.logger.info(
            format_operator_message(icon="[RELOAD]", message="Strategy Arbiter reset completed", status="All state cleared")
        )

    def __del__(self) -> None:
        try:
            if hasattr(self, "logger") and self.logger:
                self.logger.info(
                    format_operator_message(
                        icon="👋",
                        message="Strategy Arbiter shutting down",
                        total_decisions=self.arbiter_stats.get("total_decisions", 0),
                        weight_adaptations=self.arbiter_stats.get("weight_adaptations", 0),
                        gate_pass_rate=f"{(self._gate_passes / max(self._gate_attempts, 1)):.1%}",
                    )
                )
        except Exception:
            pass
