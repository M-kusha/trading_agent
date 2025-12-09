"""
Horizon Aligner
===============
Time-based weight scaling for voting proposals.
Adjusts weights based on trading horizons, sessions, and market regimes.

Refactored from time_horizon_aligner.py (~1302 lines).
~350+ lines focused on core alignment logic.

v2: Instrument-aware alignment
------------------------------
- Still produces global alignment (backward compatible)
- Additionally computes per-instrument alignment when instruments are available
- Uses optional per-instrument regime/session/volatility if published on the bus
"""

from __future__ import annotations

import datetime
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase
from modules.voting.core.constants import VotingBusKeys


@module(**module_args("HorizonAligner"))
class HorizonAligner(VotingModuleBase):
    """
    Time horizon alignment for voting proposals.

    Aligns proposal weights based on:
    - Trading session (American, European, Asian, rollover, closed)
    - Market regime (trending, volatile, ranging, breakout, reversal)
    - Volatility level (very_low → extreme)
    - Performance/quality feedback (basic diagnostics)

    v2: Instrument-aware
    --------------------
    - Computes a global alignment (for legacy consumers)
    - Computes per-instrument alignments when instruments are known
      and publishes them under:
        - horizon_alignment['per_instrument']
        - 'horizon_alignment_by_instrument' bus key
        - 'horizon_weights_by_instrument' bus key
    """

    # ====================================================================== #
    # Initialization
    # ====================================================================== #

    def _module_specific_init(self) -> None:
        """Initialize horizon alignment state."""
        # Horizon grid (in minutes). Used to shape regime/session patterns.
        default_horizons = [1, 3, 5, 10, 15, 30, 60, 120, 240]
        self.horizons = np.array(
            self.config.get("horizons", default_horizons),
            dtype=np.float32,
        )

        # Behavior configuration
        self.adaptive_scaling = bool(self.config.get("adaptive_scaling", True))
        self.regime_awareness = bool(self.config.get("regime_awareness", True))

        # Global state
        self.current_regime: str = "unknown"
        self.current_session: str = "unknown"
        # Numeric volatility (approx. ATR/price) as a rough proxy
        self.current_volatility: float = 0.02

        # Multipliers / patterns
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.regime_multipliers = self._init_regime_multipliers()
        self.session_patterns = self._init_session_patterns()

        # Quality tracking (global)
        self.alignment_quality: Dict[str, float] = {
            "effectiveness": 0.5,
            "consistency": 0.5,
            "adaptability": 0.5,
            "overall_quality": 0.5,
        }

        # History & stats (global)
        self.alignment_history: deque = deque(maxlen=200)
        self.volatility_history: deque = deque(maxlen=50)
        self.alignment_stats: Dict[str, Any] = {
            "total_alignments": 0,
            "regime_switches": 0,
            "session_transitions": 0,
        }

        self.logger.info(
            f"[HORIZON] HorizonAligner initialized | "
            f"horizons={len(self.horizons)} | adaptive={self.adaptive_scaling}"
        )

        # Publish baseline to avoid stale values
        self._publish_horizon_baseline()

    def _init_regime_multipliers(self) -> Dict[str, np.ndarray]:
        """
        Initialize regime-specific horizon multipliers.

        We use simple hand-tuned curves:
        - trending: prefer longer horizons
        - volatile: prefer shorter horizons
        - ranging: neutral
        - breakout: slightly favor medium/long
        - reversal: favor shorter/medium horizons
        """
        base = np.ones_like(self.horizons)
        n = len(base)
        
        def shape(v: List[float]) -> np.ndarray:
            """
            Normalize pattern vector to match horizons length.
            
            Handles both cases:
            - Pattern shorter than horizons: pad with 1.0 (neutral)
            - Pattern longer than horizons: truncate to horizons length
            
            This ensures broadcast compatibility regardless of config.
            """
            arr = np.array(v, dtype=np.float32)
            if len(arr) < n:
                # Pad shorter patterns with neutral 1.0
                arr = np.pad(arr, (0, n - len(arr)), constant_values=1.0)
            elif len(arr) > n:
                # Truncate longer patterns to match horizons
                arr = arr[:n]
            return base * arr

        return {
            "trending": shape([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]),
            "volatile": shape([1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
            "ranging": shape([1.0] * len(self.horizons)),
            "breakout": shape([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]),
            "reversal": shape([1.2, 1.1, 1.0, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5]),
            "unknown": shape([1.0] * len(self.horizons)),
        }

    def _init_session_patterns(self) -> Dict[str, np.ndarray]:
        """
        Initialize session-specific horizon patterns.

        These are coarse: they bias toward horizons that historically work
        better in each session (e.g., Asia often slower, US more impulsive).
        """
        base = np.ones_like(self.horizons)
        n = len(base)

        def shape(v: List[float]) -> np.ndarray:
            """
            Normalize pattern vector to match horizons length.
            
            Handles both cases:
            - Pattern shorter than horizons: pad with 1.0 (neutral)
            - Pattern longer than horizons: truncate to horizons length
            """
            arr = np.array(v, dtype=np.float32)
            if len(arr) < n:
                arr = np.pad(arr, (0, n - len(arr)), constant_values=1.0)
            elif len(arr) > n:
                arr = arr[:n]
            return base * arr

        return {
            "american": shape([1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9]),
            "european": shape([1.0] * len(self.horizons)),
            "asian": shape([0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1]),
            "closed": shape([0.3] * len(self.horizons)),
            "rollover": shape([0.5] * len(self.horizons)),
            "unknown": shape([0.8] * len(self.horizons)),
        }

    def _publish_horizon_baseline(self) -> None:
        """Publish baseline horizon keys."""
        try:
            name = self.__class__.__name__
            self.smart_bus.set(
                "horizon_alignment",
                {
                    "status": "initialized",
                    "weights": [],
                    "multipliers": {},
                    "per_instrument": {},
                    "distances": {},
                    "adaptation_status": {"status": "initialized"},
                },
                module=name,
                thesis="Baseline horizon alignment",
            )
        except Exception:
            # Baseline is best-effort only
            pass

    # ====================================================================== #
    # Main process
    # ====================================================================== #

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """Align proposal weights by time horizon (global + per-instrument)."""
        start = time.time()
        name = self.__class__.__name__

        try:
            # Decision ID for coordination across voting stages
            decision_id = self.smart_bus.get("kernel_decision_id", name)

            # Get voting / context data
            data = await self._get_voting_data()

            # Update global market state (regime/session/volatility)
            await self._update_market_state(data)

            # Global alignment (backward compatible)
            aligned_global = await self._calculate_aligned_weights(data)

            # Per-instrument alignment (v2)
            per_instrument = await self._calculate_aligned_weights_per_instrument(
                data, aligned_global
            )
            if per_instrument:
                aligned_global["per_instrument"] = per_instrument

            # Generate human-readable thesis
            thesis = self._generate_thesis(aligned_global)

            # Publish to SmartInfoBus
            await self._update_bus(aligned_global, thesis)

            # Update stats
            self.alignment_stats["total_alignments"] += 1

            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(name, "process", elapsed_ms, True)

            return {
                "horizon_alignment": aligned_global,
                "aligned_weights": aligned_global.get("weights", []),
                "horizon_multipliers": aligned_global.get("multipliers", {}),
                "current_regime": self.current_regime,
                "current_session": self.current_session,
                "alignment_quality": dict(self.alignment_quality),
                "alignment_statistics": dict(self.alignment_stats),
                "decision_id": decision_id,
                "horizon_decision_id": decision_id,
                # Contract-expected keys
                "horizon_weights": aligned_global.get("weights", []),
                "horizon_thesis": thesis,
                "horizon_distances": aligned_global.get("distances", {}),
                "adaptation_status": aligned_global.get(
                    "adaptation_status", {"status": "normal"}
                ),
                "_thesis": thesis,
            }

        except Exception as e:
            err_pin = getattr(self, "error_pinpointer", None)
            if err_pin is not None:
                error_context = err_pin.analyze_error(e, "horizon_process")
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)

    # ====================================================================== #
    # Data collection & state updates
    # ====================================================================== #

    async def _get_voting_data(self) -> Dict[str, Any]:
        """Get voting and context data from SmartInfoBus."""
        name = self.__class__.__name__

        # Member-level base weights
        member_confidences = (
            self.smart_bus.get(VotingBusKeys.MEMBER_CONFIDENCES, name, default=None)
            or self.smart_bus.get("committee_member_confidences", name)
            or []
        )
        expert_weights = self.smart_bus.get("expert_weights", name) or {}

        # Global regime/session/volatility labels
        market_regime = self.smart_bus.get("market_regime", name) or "unknown"
        session = self.smart_bus.get("session_canonical", name) or "unknown"
        volatility = self.smart_bus.get("volatility_level", name) or "medium"

        # Instruments (prefer ACTIVE_INSTRUMENTS, else infer from committee decisions)
        instruments: Optional[List[str]] = self.smart_bus.get(
            VotingBusKeys.ACTIVE_INSTRUMENTS, name, default=None
        )
        if not instruments:
            inst_decisions = self.smart_bus.get(
                "committee_decisions_by_instrument", name, default={}
            ) or {}
            if isinstance(inst_decisions, dict):
                instruments = list(inst_decisions.keys())
            else:
                instruments = []

        # Optional per-instrument context
        regime_by_inst = self.smart_bus.get(
            "market_regime_by_instrument", name, default={}
        ) or {}
        session_by_inst = self.smart_bus.get(
            "session_canonical_by_instrument", name, default={}
        ) or {}
        vol_by_inst = self.smart_bus.get(
            "volatility_level_by_instrument", name, default={}
        ) or {}

        return {
            "member_confidences": member_confidences,
            "expert_weights": expert_weights,
            "market_regime": market_regime,
            "session": session,
            "volatility": volatility,
            "instruments": instruments,
            "market_regime_by_instrument": regime_by_inst,
            "session_by_instrument": session_by_inst,
            "volatility_by_instrument": vol_by_inst,
        }

    async def _update_market_state(self, data: Dict[str, Any]) -> None:
        """Update current global market state from bus data."""
        new_regime = str(data.get("market_regime", "unknown")).lower()
        new_session = str(data.get("session", "unknown")).lower()

        # Track regime changes
        if new_regime != self.current_regime:
            self.alignment_stats["regime_switches"] += 1
            self.current_regime = new_regime

        # Track session changes
        if new_session != self.current_session:
            self.alignment_stats["session_transitions"] += 1
            self.current_session = new_session

        # Update global volatility (numeric)
        vol_str = str(data.get("volatility", "medium")).lower()
        vol_map = {
            "very_low": 0.005,
            "low": 0.01,
            "medium": 0.02,
            "high": 0.04,
            "extreme": 0.08,
        }
        self.current_volatility = vol_map.get(vol_str, 0.02)
        self.volatility_history.append(self.current_volatility)

    # ====================================================================== #
    # Alignment (global + per-instrument)
    # ====================================================================== #

    async def _calculate_aligned_weights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate GLOBAL horizon-aligned weights (backward compatible).

        Uses:
        - member_confidences (preferred) or expert_weights as base
        - current_regime / current_session / volatility from global state
        """
        raw_confidences = data.get("member_confidences") or []

        # member_confidences may be a dict {member: conf} or a list; normalize to list of floats
        if isinstance(raw_confidences, dict):
            base_weights = list(raw_confidences.values())
        elif isinstance(raw_confidences, list):
            base_weights = raw_confidences
        else:
            base_weights = []

        if not base_weights:
            expert_weights = data.get("expert_weights") or {}
            if isinstance(expert_weights, dict) and expert_weights:
                base_weights = list(expert_weights.values())
            else:
                base_weights = [0.5]

        if not base_weights:
            return {
                "weights": [],
                "multipliers": {},
                "status": "no_weights",
                "distances": {},
                "adaptation_status": {"status": "no_weights"},
            }

        global_vol_label = str(data.get("volatility", "medium")).lower()

        aligned = self._compute_aligned_for_context(
            base_weights=base_weights,
            regime=self.current_regime if self.regime_awareness else "unknown",
            session=self.current_session,
            volatility_label=global_vol_label,
            record_history=True,  # only global alignment updates history/quality
        )

        return aligned

    async def _calculate_aligned_weights_per_instrument(
        self,
        data: Dict[str, Any],
        global_alignment: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate per-instrument horizon-aligned weights.

        - Uses same base_weights as global (same committee members)
        - Uses per-instrument regime/session/vol if available
        - Falls back to global context when missing
        - Does NOT touch global history/quality (purely contextual)
        """
        instruments: List[str] = data.get("instruments") or []
        if not instruments:
            return {}

        # Base weights identical per instrument (reweighting same committee)
        raw_confidences = data.get("member_confidences") or []
        if isinstance(raw_confidences, dict):
            base_weights = list(raw_confidences.values())
        elif isinstance(raw_confidences, list):
            base_weights = raw_confidences
        else:
            base_weights = []

        if not base_weights:
            expert_weights = data.get("expert_weights") or {}
            if isinstance(expert_weights, dict) and expert_weights:
                base_weights = list(expert_weights.values())
            else:
                return {}

        regime_by_inst: Dict[str, Any] = data.get("market_regime_by_instrument") or {}
        session_by_inst: Dict[str, Any] = data.get("session_by_instrument") or {}
        vol_by_inst: Dict[str, Any] = data.get("volatility_by_instrument") or {}
        global_vol_label = str(data.get("volatility", "medium")).lower()

        per_inst: Dict[str, Dict[str, Any]] = {}

        for inst in instruments:
            # Instrument-specific context with fallback to global
            inst_regime = str(regime_by_inst.get(inst, self.current_regime)).lower()
            if not self.regime_awareness:
                inst_regime = "unknown"

            inst_session = str(session_by_inst.get(inst, self.current_session)).lower()
            inst_vol_label = str(vol_by_inst.get(inst, global_vol_label)).lower()

            inst_alignment = self._compute_aligned_for_context(
                base_weights=base_weights,
                regime=inst_regime,
                session=inst_session,
                volatility_label=inst_vol_label,
                record_history=False,  # do not pollute global history
            )

            inst_alignment["instrument"] = inst
            per_inst[inst] = inst_alignment

        return per_inst

    def _compute_aligned_for_context(
        self,
        base_weights: List[float],
        regime: str,
        session: str,
        volatility_label: str,
        record_history: bool = False,
    ) -> Dict[str, Any]:
        """
        Core alignment logic for a given (regime, session, volatility) context.

        Produces:
        - weights: reweighted member confidences
        - multipliers: regime/session/combined scalar multipliers
        - distances: diagnostic distances from neutral multiplier (1.0)
        - adaptation_status: coarse description of how aggressive the adjustment is

        This method does not mutate global state except when record_history=True.
        """
        if not base_weights:
            return {
                "weights": [],
                "multipliers": {},
                "status": "no_weights",
                "distances": {},
                "adaptation_status": {"status": "no_weights"},
            }

        regime_key = str(regime or "unknown").lower()
        session_key = str(session or "unknown").lower()

        regime_mult = self.regime_multipliers.get(
            regime_key, np.ones_like(self.horizons)
        )
        session_mult = self.session_patterns.get(
            session_key, np.ones_like(self.horizons)
        )

        # Use the first horizon as a canonical scalar multiplier
        combined_mult = float(regime_mult[0] * session_mult[0])

        # Volatility-based adjustment
        vol_map = {
            "very_low": 0.005,
            "low": 0.01,
            "medium": 0.02,
            "high": 0.04,
            "extreme": 0.08,
        }
        vol_numeric = vol_map.get(str(volatility_label).lower(), self.current_volatility)

        if self.adaptive_scaling:
            vol_mult = self._get_volatility_multiplier(vol_numeric)
            combined_mult *= vol_mult

        # Apply scalar multiplier to base weights and clamp to [0, 1]
        aligned_weights = [
            max(0.0, min(1.0, float(w) * combined_mult)) for w in base_weights
        ]

        # Diagnostics: distances from neutral (1.0)
        delta = combined_mult - 1.0
        abs_delta = abs(delta)
        distances = {
            "multiplier": combined_mult,
            "delta_from_neutral": delta,
            "abs_delta_from_neutral": abs_delta,
            "volatility_numeric": vol_numeric,
        }

        # Adaptation status label for operators
        if abs_delta < 0.05:
            status = "neutral"
        elif abs_delta < 0.15:
            status = "mild_adjustment"
        else:
            status = "strong_adjustment"

        adaptation_status = {
            "status": status,
            "regime": regime_key,
            "session": session_key,
            "volatility_label": str(volatility_label).lower(),
            "multiplier": combined_mult,
        }

        # Optionally record history / quality for global alignment
        if record_history:
            self.alignment_history.append(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "regime": regime_key,
                    "session": session_key,
                    "multiplier": combined_mult,
                    "volatility": vol_numeric,
                }
            )
            self._update_quality_metrics(combined_mult, vol_numeric)

        return {
            "weights": aligned_weights,
            "multipliers": {
                "regime": float(regime_mult[0]),
                "session": float(session_mult[0]),
                "combined": combined_mult,
            },
            "regime": regime_key,
            "session": session_key,
            "status": "aligned",
            "distances": distances,
            "adaptation_status": adaptation_status,
        }

    def _get_volatility_multiplier(self, vol: Optional[float] = None) -> float:
        """
        Get volatility-based multiplier from numeric vol level.

        - Very low volatility → slightly reduce impact (avoid overreaction)
        - Very high volatility → increase impact (let horizon shaping matter more)
        """
        if vol is None:
            vol = self.current_volatility

        if vol < 0.01:
            return 0.9
        if vol < 0.02:
            return 1.0
        if vol < 0.04:
            return 1.1
        if vol < 0.07:
            return 1.2
        return 1.3

    # ====================================================================== #
    # Quality metrics & diagnostics
    # ====================================================================== #

    def _update_quality_metrics(self, multiplier: float, vol: float) -> None:
        """
        Update alignment quality metrics (global):

        - effectiveness: prefers moderate, not extreme, adjustments
        - consistency: how stable recent multipliers have been
        - adaptability: how responsive multipliers are to volatility shifts
        - overall_quality: simple weighted aggregate for diagnostics
        """
        # Effectiveness: maximal around multiplier ~1.0, decays as we move away
        delta = abs(multiplier - 1.0)
        effectiveness = 1.0 - min(1.0, delta * 2.0)
        self.alignment_quality["effectiveness"] = float(
            max(0.0, min(1.0, effectiveness))
        )

        # Consistency: low std of recent multipliers → high consistency
        if len(self.alignment_history) >= 5:
            recent_mults = [
                h["multiplier"] for h in list(self.alignment_history)[-10:]
            ]
            if len(recent_mults) >= 2:
                std = float(np.std(recent_mults))
                consistency = 1.0 - min(1.0, std * 2.0)
            else:
                consistency = 0.5
        else:
            consistency = 0.5
        self.alignment_quality["consistency"] = float(
            max(0.0, min(1.0, consistency))
        )

        # Adaptability: we want some reaction to volatility, but not wild swings.
        # Simple heuristic: higher when multiplier and volatility both deviate
        # moderately from baseline.
        vol_baseline = 0.02
        vol_delta = abs(vol - vol_baseline)
        # Map vol_delta into [0,1], saturating around 0.04
        vol_factor = min(1.0, vol_delta / 0.04)
        # Combine with magnitude of multiplier adjustment
        mult_factor = min(1.0, delta / 0.2)
        adaptability = 0.5 * vol_factor + 0.5 * mult_factor
        self.alignment_quality["adaptability"] = float(
            max(0.0, min(1.0, adaptability))
        )

        # Aggregate
        overall = (
            0.4 * self.alignment_quality["effectiveness"]
            + 0.3 * self.alignment_quality["consistency"]
            + 0.3 * self.alignment_quality["adaptability"]
        )
        self.alignment_quality["overall_quality"] = float(
            max(0.0, min(1.0, overall))
        )

    def _generate_thesis(self, aligned: Dict[str, Any]) -> str:
        """Generate human-readable alignment thesis (global)."""
        mults = aligned.get("multipliers", {})
        combined = float(mults.get("combined", 1.0))
        n_weights = len(aligned.get("weights", []))
        per_inst = aligned.get("per_instrument") or {}
        n_instruments = len(per_inst)
        status = aligned.get("adaptation_status", {}).get("status", "unknown")

        return (
            f"HORIZON: regime={self.current_regime} | session={self.current_session} | "
            f"multiplier={combined:.2f} | status={status} | weights={n_weights} | "
            f"instruments={n_instruments} | "
            f"quality={self.alignment_quality['overall_quality']:.2f}"
        )

    async def _update_bus(self, aligned: Dict[str, Any], thesis: str) -> None:
        """Update SmartInfoBus with results."""
        try:
            name = self.__class__.__name__

            # Global alignment (legacy + v2 structure)
            self.smart_bus.set(
                "horizon_alignment",
                aligned,
                module=name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "aligned_weights",
                aligned.get("weights", []),
                module=name,
                thesis=f'Aligned {len(aligned.get("weights", []))} global weights',
            )

            # Per-instrument convenience surfaces (v2)
            per_inst = aligned.get("per_instrument") or {}
            if per_inst:
                self.smart_bus.set(
                    "horizon_alignment_by_instrument",
                    per_inst,
                    module=name,
                    thesis=f"Horizon alignment for {len(per_inst)} instruments",
                )
                per_inst_weights = {
                    inst: ctx.get("weights", []) for inst, ctx in per_inst.items()
                }
                self.smart_bus.set(
                    "horizon_weights_by_instrument",
                    per_inst_weights,
                    module=name,
                    thesis="Per-instrument horizon-aligned member weights",
                )

        except Exception as e:
            self.logger.warning(f"[HORIZON] Bus update failed: {e}")

    # ====================================================================== #
    # Error handling
    # ====================================================================== #

    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        thesis = f"Horizon alignment error: {error}"
        return {
            "horizon_alignment": {
                "error": error,
                "status": "error",
                "per_instrument": {},
                "distances": {},
                "adaptation_status": {"status": "error", "error": error},
            },
            "aligned_weights": [],
            "horizon_multipliers": {},
            "current_regime": self.current_regime,
            "current_session": self.current_session,
            "alignment_quality": dict(self.alignment_quality),
            "alignment_statistics": dict(self.alignment_stats),
            "decision_id": None,
            "horizon_decision_id": None,
            # Contract-expected keys
            "horizon_weights": [],
            "horizon_thesis": thesis,
            "horizon_distances": {},
            "adaptation_status": {"status": "error", "error": error},
            "_thesis": thesis,
        }

    # ═══════════════════════════════════════════════════════════════════
    # STATE PERSISTENCE - Save/Load module state
    # ═══════════════════════════════════════════════════════════════════

    def _get_custom_state(self) -> Dict[str, Any]:
        """
        Get custom state for persistence.

        Saves:
        - Global regime/session
        - Volatility and alignment histories (truncated)
        - Alignment quality metrics
        - Alignment statistics
        """
        return {
            "current_regime": self.current_regime,
            "current_session": self.current_session,
            "current_volatility": self.current_volatility,
            "alignment_history": list(self.alignment_history)[-50:],
            "volatility_history": list(self.volatility_history)[-50:],
            "alignment_quality": dict(self.alignment_quality),
            "alignment_stats": dict(self.alignment_stats),
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        """
        Restore custom state from persistence.
        """
        if not state:
            return

        try:
            self.current_regime = str(state.get("current_regime", "unknown"))
            self.current_session = str(state.get("current_session", "unknown"))
            self.current_volatility = float(state.get("current_volatility", 0.02))
        except Exception:
            self.current_regime = "unknown"
            self.current_session = "unknown"
            self.current_volatility = 0.02

        hist = state.get("alignment_history", [])
        self.alignment_history = deque(hist, maxlen=200)

        vol_hist = state.get("volatility_history", [])
        self.volatility_history = deque(vol_hist, maxlen=50)

        quality = state.get("alignment_quality", {})
        self.alignment_quality.update(quality)

        stats = state.get("alignment_stats", {})
        self.alignment_stats.update(stats)

        self.logger.info(
            f"📂 HorizonAligner state restored | "
            f"regime={self.current_regime} | session={self.current_session} | "
            f"alignments={self.alignment_stats.get('total_alignments', 0)}"
        )
