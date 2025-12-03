"""
Horizon Aligner
===============
Time-based weight scaling for voting proposals.
Adjusts weights based on trading horizons, sessions, and market regimes.

Refactored from time_horizon_aligner.py (~1302 lines).
~350 lines focused on core alignment logic.

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
    - Trading session (American, European, Asian)
    - Market regime (trending, volatile, ranging)
    - Time horizons (short, medium, long term)
    - Performance feedback
    
    v2: Instrument-aware
    --------------------
    - Computes a global alignment (for legacy consumers)
    - Computes per-instrument alignments when instruments are known
      and publishes them under:
        - horizon_alignment['per_instrument']
        - 'horizon_alignment_by_instrument' bus key
    """

    def _module_specific_init(self) -> None:
        """Initialize horizon alignment state."""
        # Horizons (in minutes)
        default_horizons = [1, 3, 5, 10, 15, 30, 60, 120, 240]
        self.horizons = np.array(
            self.config.get('horizons', default_horizons),
            dtype=np.float32
        )

        # Configuration
        self.adaptive_scaling = bool(self.config.get('adaptive_scaling', True))
        self.regime_awareness = bool(self.config.get('regime_awareness', True))

        # State (global context)
        self.current_regime = 'unknown'
        self.current_session = 'unknown'
        self.current_volatility = 0.02

        # Alignment multipliers
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.regime_multipliers = self._init_regime_multipliers()
        self.session_patterns = self._init_session_patterns()

        # Quality tracking (global)
        self.alignment_quality: Dict[str, float] = {
            'effectiveness': 0.5,
            'consistency': 0.5,
            'adaptability': 0.5,
            'overall_quality': 0.5,
        }

        # History & stats (global)
        self.alignment_history: deque = deque(maxlen=200)
        self.volatility_history: deque = deque(maxlen=50)
        self.alignment_stats: Dict[str, Any] = {
            'total_alignments': 0,
            'regime_switches': 0,
            'session_transitions': 0,
        }

        self.logger.info(
            f"[HORIZON] HorizonAligner initialized | "
            f"horizons={len(self.horizons)} | adaptive={self.adaptive_scaling}"
        )

        # Publish baseline
        self._publish_horizon_baseline()

    def _init_regime_multipliers(self) -> Dict[str, np.ndarray]:
        """Initialize regime-specific multipliers."""
        base = np.ones_like(self.horizons)
        return {
            'trending': base * np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])[:len(base)],
            'volatile': base * np.array([1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])[:len(base)],
            'ranging': base * 1.0,
            'breakout': base * np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])[:len(base)],
            'reversal': base * np.array([1.2, 1.1, 1.0, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5])[:len(base)],
            'unknown': base * 1.0,
        }

    def _init_session_patterns(self) -> Dict[str, np.ndarray]:
        """Initialize session-specific patterns."""
        base = np.ones_like(self.horizons)
        return {
            'american': base * np.array([1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9, 0.9])[:len(base)],
            'european': base * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])[:len(base)],
            'asian': base * np.array([0.9, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.1, 1.1])[:len(base)],
            'closed': base * 0.3,
            'rollover': base * 0.5,
            'unknown': base * 0.8,
        }

    def _publish_horizon_baseline(self) -> None:
        """Publish baseline horizon keys."""
        try:
            self.smart_bus.set(
                'horizon_alignment',
                {'status': 'initialized', 'weights': [], 'multipliers': {}, 'per_instrument': {}},
                module='HorizonAligner',
                thesis='Baseline horizon alignment'
            )
        except Exception:
            pass

    async def process(self, **inputs) -> Dict[str, Any]:
        """Align proposal weights by time horizon (global + per-instrument)."""
        start = time.time()
        name = self.__class__.__name__

        try:
            # Decision ID for coordination
            decision_id = self.smart_bus.get('kernel_decision_id', name)

            # Get voting / context data
            data = await self._get_voting_data()

            # Update global market state (regime/session/volatility)
            await self._update_market_state(data)

            # Global alignment (backward compatible)
            aligned_global = await self._calculate_aligned_weights(data)

            # Per-instrument alignment (new)
            per_instrument = await self._calculate_aligned_weights_per_instrument(data, aligned_global)

            if per_instrument:
                aligned_global['per_instrument'] = per_instrument

            # Thesis
            thesis = self._generate_thesis(aligned_global)

            # Publish to bus
            await self._update_bus(aligned_global, thesis)

            # Stats
            self.alignment_stats['total_alignments'] += 1

            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(name, 'process', elapsed_ms, True)

            return {
                'horizon_alignment': aligned_global,
                'aligned_weights': aligned_global.get('weights', []),
                'horizon_multipliers': aligned_global.get('multipliers', {}),
                'current_regime': self.current_regime,
                'current_session': self.current_session,
                'alignment_quality': dict(self.alignment_quality),
                'alignment_statistics': dict(self.alignment_stats),
                'decision_id': decision_id,
                'horizon_decision_id': decision_id,
                # Contract-expected keys
                'horizon_weights': aligned_global.get('weights', []),
                'horizon_thesis': thesis,
                'horizon_distances': aligned_global.get('distances', {}),
                'adaptation_status': aligned_global.get('adaptation', {'status': 'normal'}),
                '_thesis': thesis,
            }

        except Exception as e:
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(e, 'horizon_process')
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)

    async def _get_voting_data(self) -> Dict[str, Any]:
        """Get voting and context data from SmartInfoBus."""
        # Global member-level stuff (same for all instruments)
        member_confidences = self.smart_bus.get(
            'committee_member_confidences', self.__class__.__name__
        ) or []
        expert_weights = self.smart_bus.get(
            'expert_weights', self.__class__.__name__
        ) or {}

        # Global regime/session/volatility
        market_regime = self.smart_bus.get(
            'market_regime', self.__class__.__name__
        ) or 'unknown'
        session = self.smart_bus.get(
            'session_canonical', self.__class__.__name__
        ) or 'unknown'
        volatility = self.smart_bus.get(
            'volatility_level', self.__class__.__name__
        ) or 'medium'

        # Instruments (prefer ACTIVE_INSTRUMENTS, fallback to committee decisions)
        instruments: Optional[List[str]] = self.bus_get(
            VotingBusKeys.ACTIVE_INSTRUMENTS, default=None
        )
        if not instruments:
            inst_decisions = self.smart_bus.get(
                'committee_decisions_by_instrument', self.__class__.__name__, default={}
            ) or {}
            if isinstance(inst_decisions, dict):
                instruments = list(inst_decisions.keys())
            else:
                instruments = []

        # Optional per-instrument state (if you start publishing these)
        regime_by_inst = self.smart_bus.get(
            'market_regime_by_instrument', self.__class__.__name__
        ) or {}
        session_by_inst = self.smart_bus.get(
            'session_canonical_by_instrument', self.__class__.__name__
        ) or {}
        vol_by_inst = self.smart_bus.get(
            'volatility_level_by_instrument', self.__class__.__name__
        ) or {}

        return {
            'member_confidences': member_confidences,
            'expert_weights': expert_weights,
            'market_regime': market_regime,
            'session': session,
            'volatility': volatility,
            'instruments': instruments,
            'market_regime_by_instrument': regime_by_inst,
            'session_by_instrument': session_by_inst,
            'volatility_by_instrument': vol_by_inst,
        }

    async def _update_market_state(self, data: Dict[str, Any]) -> None:
        """Update current global market state."""
        new_regime = str(data.get('market_regime', 'unknown')).lower()
        new_session = str(data.get('session', 'unknown')).lower()

        # Track regime changes
        if new_regime != self.current_regime:
            self.alignment_stats['regime_switches'] += 1
            self.current_regime = new_regime

        # Track session changes
        if new_session != self.current_session:
            self.alignment_stats['session_transitions'] += 1
            self.current_session = new_session

        # Update global volatility (numeric)
        vol_str = str(data.get('volatility', 'medium')).lower()
        vol_map = {
            'very_low': 0.005, 'low': 0.01, 'medium': 0.02,
            'high': 0.04, 'extreme': 0.08
        }
        self.current_volatility = vol_map.get(vol_str, 0.02)
        self.volatility_history.append(self.current_volatility)

    async def _calculate_aligned_weights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate GLOBAL horizon-aligned weights (backward compatible).
        Uses current_regime/current_session/current_volatility.
        """
        # Base weights: first try member_confidences, then expert_weights
        base_weights = data.get('member_confidences') or []
        if not base_weights:
            expert_weights = data.get('expert_weights') or {}
            base_weights = list(expert_weights.values()) if expert_weights else [0.5]

        n_members = len(base_weights)
        if n_members == 0:
            return {
                'weights': [],
                'multipliers': {},
                'status': 'no_weights',
            }

        # Use global context
        global_vol_label = str(data.get('volatility', 'medium')).lower()

        aligned = self._compute_aligned_for_context(
            base_weights=base_weights,
            regime=self.current_regime if self.regime_awareness else 'unknown',
            session=self.current_session,
            volatility_label=global_vol_label,
            record_history=True,   # only global alignment updates history/quality
        )

        return aligned

    async def _calculate_aligned_weights_per_instrument(
        self,
        data: Dict[str, Any],
        global_alignment: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate per-instrument horizon-aligned weights.

        - Uses same base_weights as global
        - Uses per-instrument regime/session/vol if available
        - Falls back to global context when missing
        - Does NOT touch global history/quality (purely contextual)
        """
        instruments: List[str] = data.get('instruments') or []
        if not instruments:
            return {}

        # Base weights identical per instrument (we are re-weighting the same committee)
        base_weights = data.get('member_confidences') or []
        if not base_weights:
            expert_weights = data.get('expert_weights') or {}
            base_weights = list(expert_weights.values()) if expert_weights else [0.5]

        if not base_weights:
            return {}

        regime_by_inst: Dict[str, Any] = data.get('market_regime_by_instrument') or {}
        session_by_inst: Dict[str, Any] = data.get('session_by_instrument') or {}
        vol_by_inst: Dict[str, Any] = data.get('volatility_by_instrument') or {}
        global_vol_label = str(data.get('volatility', 'medium')).lower()

        per_inst: Dict[str, Dict[str, Any]] = {}

        for inst in instruments:
            # Instrument-specific context with fallback to global
            inst_regime = str(regime_by_inst.get(inst, self.current_regime)).lower()
            if not self.regime_awareness:
                inst_regime = 'unknown'

            inst_session = str(session_by_inst.get(inst, self.current_session)).lower()
            inst_vol_label = str(vol_by_inst.get(inst, global_vol_label)).lower()

            inst_alignment = self._compute_aligned_for_context(
                base_weights=base_weights,
                regime=inst_regime,
                session=inst_session,
                volatility_label=inst_vol_label,
                record_history=False,  # do not pollute global history
            )

            inst_alignment['instrument'] = inst
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

        - Does not assume global state except horizons / multipliers tables.
        - Optionally records history & updates quality for global alignment.
        """
        if not base_weights:
            return {
                'weights': [],
                'multipliers': {},
                'status': 'no_weights',
            }

        # Regime/session multipliers
        regime_key = str(regime or 'unknown').lower()
        session_key = str(session or 'unknown').lower()

        regime_mult = self.regime_multipliers.get(regime_key, np.ones_like(self.horizons))
        session_mult = self.session_patterns.get(session_key, np.ones_like(self.horizons))

        # Combined (we still use the first horizon as canonical scalar)
        combined_mult = float(regime_mult[0] * session_mult[0])

        # Volatility-based adjustment
        if self.adaptive_scaling:
            vol_map = {
                'very_low': 0.005, 'low': 0.01, 'medium': 0.02,
                'high': 0.04, 'extreme': 0.08
            }
            vol_numeric = vol_map.get(str(volatility_label).lower(), self.current_volatility)
            vol_mult = self._get_volatility_multiplier(vol_numeric)
            combined_mult *= vol_mult

        # Apply scalar to base weights
        aligned_weights = [
            max(0.0, min(1.0, float(w) * combined_mult))
            for w in base_weights
        ]

        # Optionally record history / quality (for global alignment)
        if record_history:
            self.alignment_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'regime': regime_key,
                'session': session_key,
                'multiplier': combined_mult,
            })
            self._update_quality_metrics(combined_mult)

        return {
            'weights': aligned_weights,
            'multipliers': {
                'regime': float(regime_mult[0]),
                'session': float(session_mult[0]),
                'combined': combined_mult,
            },
            'regime': regime_key,
            'session': session_key,
            'status': 'aligned',
        }

    def _get_volatility_multiplier(self, vol: Optional[float] = None) -> float:
        """Get volatility-based multiplier from numeric vol level."""
        if vol is None:
            vol = self.current_volatility

        if vol < 0.01:
            return 0.8  # Low vol: less adjustment
        elif vol > 0.05:
            return 1.3  # High vol: more adjustment
        return 1.0

    def _update_quality_metrics(self, multiplier: float) -> None:
        """Update alignment quality metrics (global)."""
        # Effectiveness: how much we're adjusting (optimal is moderate)
        eff = 1.0 - abs(multiplier - 1.0)
        self.alignment_quality['effectiveness'] = float(max(0.0, min(1.0, eff)))

        # Consistency: how stable our adjustments are
        if len(self.alignment_history) >= 5:
            recent_mults = [h['multiplier'] for h in list(self.alignment_history)[-5:]]
            std = float(np.std(recent_mults))
            self.alignment_quality['consistency'] = float(max(0.0, min(1.0, 1.0 - std)))

        # Overall
        self.alignment_quality['overall_quality'] = float(
            self.alignment_quality['effectiveness'] * 0.4 +
            self.alignment_quality['consistency'] * 0.4 +
            self.alignment_quality['adaptability'] * 0.2
        )

    def _generate_thesis(self, aligned: Dict[str, Any]) -> str:
        """Generate alignment thesis (global)."""
        mults = aligned.get('multipliers', {})
        combined = mults.get('combined', 1.0)
        n_weights = len(aligned.get('weights', []))
        per_inst = aligned.get('per_instrument') or {}
        n_instruments = len(per_inst)

        return (
            f"HORIZON: regime={self.current_regime} | session={self.current_session} | "
            f"multiplier={combined:.2f} | weights={n_weights} | "
            f"instruments={n_instruments} | "
            f"quality={self.alignment_quality['overall_quality']:.2f}"
        )

    async def _update_bus(self, aligned: Dict[str, Any], thesis: str) -> None:
        """Update SmartInfoBus with results."""
        try:
            name = self.__class__.__name__

            # Global alignment (legacy + v2)
            self.smart_bus.set(
                'horizon_alignment',
                aligned,
                module=name,
                thesis=thesis
            )
            self.smart_bus.set(
                'aligned_weights',
                aligned.get('weights', []),
                module=name,
                thesis=f'Aligned {len(aligned.get("weights", []))} global weights'
            )

            # Per-instrument convenience surfaces (new)
            per_inst = aligned.get('per_instrument') or {}
            if per_inst:
                self.smart_bus.set(
                    'horizon_alignment_by_instrument',
                    per_inst,
                    module=name,
                    thesis=f'Horizon alignment for {len(per_inst)} instruments'
                )
                # Optionally publish just weights per instrument
                per_inst_weights = {
                    inst: ctx.get('weights', [])
                    for inst, ctx in per_inst.items()
                }
                self.smart_bus.set(
                    'horizon_weights_by_instrument',
                    per_inst_weights,
                    module=name,
                    thesis='Per-instrument horizon-aligned member weights'
                )

        except Exception as e:
            self.logger.warning(f"Bus update failed: {e}")

    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        return {
            'horizon_alignment': {'error': error, 'status': 'error', 'per_instrument': {}},
            'aligned_weights': [],
            'horizon_multipliers': {},
            'current_regime': self.current_regime,
            'current_session': self.current_session,
            'alignment_quality': dict(self.alignment_quality),
            'alignment_statistics': dict(self.alignment_stats),
            'decision_id': None,
            'horizon_decision_id': None,
            # Contract-expected keys
            'horizon_weights': [],
            'horizon_thesis': f'Horizon alignment error: {error}',
            'horizon_distances': {},
            'adaptation_status': {'status': 'error', 'error': error},
            '_thesis': f'Horizon alignment error: {error}',
        }
