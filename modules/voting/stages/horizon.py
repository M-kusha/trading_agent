"""
Horizon Aligner
===============
Time-based weight scaling for voting proposals.
Adjusts weights based on trading horizons, sessions, and market regimes.

Refactored from time_horizon_aligner.py (~1302 lines).
~350 lines focused on core alignment logic.
"""

from __future__ import annotations

import datetime
import time
from collections import deque
from typing import Any, Dict, List

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase


@module(**module_args("HorizonAligner"))
class HorizonAligner(VotingModuleBase):
    """
    Time horizon alignment for voting proposals.
    
    Aligns proposal weights based on:
    - Trading session (American, European, Asian)
    - Market regime (trending, volatile, ranging)
    - Time horizons (short, medium, long term)
    - Performance feedback
    
    Publishes:
    - horizon_alignment
    - aligned_weights
    - horizon_analysis
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
        
        # State
        self.current_regime = 'unknown'
        self.current_session = 'unknown'
        self.current_volatility = 0.02
        
        # Alignment multipliers
        self.adaptive_multipliers = np.ones_like(self.horizons)
        self.regime_multipliers = self._init_regime_multipliers()
        self.session_patterns = self._init_session_patterns()
        
        # Quality tracking
        self.alignment_quality: Dict[str, float] = {
            'effectiveness': 0.5,
            'consistency': 0.5,
            'adaptability': 0.5,
            'overall_quality': 0.5,
        }
        
        # History
        self.alignment_history: deque = deque(maxlen=200)
        self.volatility_history: deque = deque(maxlen=50)
        
        # Statistics
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
                {'status': 'initialized', 'aligned_weights': []},
                module='HorizonAligner',
                thesis='Baseline horizon alignment'
            )
        except Exception:
            pass
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """Align proposal weights by time horizon."""
        start = time.time()
        name = self.__class__.__name__
        
        try:
            # Get decision ID
            decision_id = self.smart_bus.get('kernel_decision_id', name)
            
            # Get voting data
            data = await self._get_voting_data()
            
            # Update market state
            await self._update_market_state(data)
            
            # Calculate aligned weights
            aligned = await self._calculate_aligned_weights(data)
            
            # Generate thesis
            thesis = self._generate_thesis(aligned)
            
            # Publish to bus
            await self._update_bus(aligned, thesis)
            
            # Update stats
            self.alignment_stats['total_alignments'] += 1
            
            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(name, 'process', elapsed_ms, True)
            
            return {
                'horizon_alignment': aligned,
                'aligned_weights': aligned.get('weights', []),
                'horizon_multipliers': aligned.get('multipliers', {}),
                'current_regime': self.current_regime,
                'current_session': self.current_session,
                'alignment_quality': dict(self.alignment_quality),
                'alignment_statistics': dict(self.alignment_stats),
                'decision_id': decision_id,
                'horizon_decision_id': decision_id,
                # Contract-expected keys
                'horizon_weights': aligned.get('weights', []),
                'horizon_thesis': thesis,
                'horizon_distances': aligned.get('distances', {}),
                'adaptation_status': aligned.get('adaptation', {'status': 'normal'}),
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
        """Get voting data from SmartInfoBus."""
        return {
            'member_confidences': self.smart_bus.get('committee_member_confidences', self.__class__.__name__) or [],
            'expert_weights': self.smart_bus.get('expert_weights', self.__class__.__name__) or {},
            'market_regime': self.smart_bus.get('market_regime', self.__class__.__name__) or 'unknown',
            'session': self.smart_bus.get('session_canonical', self.__class__.__name__) or 'unknown',
            'volatility': self.smart_bus.get('volatility_level', self.__class__.__name__) or 'medium',
        }
    
    async def _update_market_state(self, data: Dict[str, Any]) -> None:
        """Update current market state."""
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
        
        # Update volatility
        vol_str = str(data.get('volatility', 'medium')).lower()
        vol_map = {
            'very_low': 0.005, 'low': 0.01, 'medium': 0.02,
            'high': 0.04, 'extreme': 0.08
        }
        self.current_volatility = vol_map.get(vol_str, 0.02)
        self.volatility_history.append(self.current_volatility)
    
    async def _calculate_aligned_weights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate horizon-aligned weights."""
        # Get base weights
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
        
        # Get regime and session multipliers
        regime_mult = self.regime_multipliers.get(self.current_regime, np.ones_like(self.horizons))
        session_mult = self.session_patterns.get(self.current_session, np.ones_like(self.horizons))
        
        # Calculate combined multiplier (use first horizon for simplicity)
        combined_mult = float(regime_mult[0] * session_mult[0])
        
        # Apply adaptive scaling
        if self.adaptive_scaling:
            vol_mult = self._get_volatility_multiplier()
            combined_mult *= vol_mult
        
        # Apply to weights
        aligned_weights = [
            max(0.0, min(1.0, float(w) * combined_mult))
            for w in base_weights
        ]
        
        # Record history
        self.alignment_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'regime': self.current_regime,
            'session': self.current_session,
            'multiplier': combined_mult,
        })
        
        # Update quality metrics
        self._update_quality_metrics(combined_mult)
        
        return {
            'weights': aligned_weights,
            'multipliers': {
                'regime': float(regime_mult[0]),
                'session': float(session_mult[0]),
                'combined': combined_mult,
            },
            'regime': self.current_regime,
            'session': self.current_session,
            'status': 'aligned',
        }
    
    def _get_volatility_multiplier(self) -> float:
        """Get volatility-based multiplier."""
        vol = self.current_volatility
        if vol < 0.01:
            return 0.8  # Low vol: less adjustment
        elif vol > 0.05:
            return 1.3  # High vol: more adjustment
        return 1.0
    
    def _update_quality_metrics(self, multiplier: float) -> None:
        """Update alignment quality metrics."""
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
        """Generate alignment thesis."""
        mults = aligned.get('multipliers', {})
        combined = mults.get('combined', 1.0)
        n_weights = len(aligned.get('weights', []))
        
        return (
            f"HORIZON: regime={self.current_regime} | session={self.current_session} | "
            f"multiplier={combined:.2f} | weights={n_weights} | "
            f"quality={self.alignment_quality['overall_quality']:.2f}"
        )
    
    async def _update_bus(self, aligned: Dict[str, Any], thesis: str) -> None:
        """Update SmartInfoBus with results."""
        try:
            name = self.__class__.__name__
            
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
                thesis=f'Aligned {len(aligned.get("weights", []))} weights'
            )
        except Exception as e:
            self.logger.warning(f"Bus update failed: {e}")
    
    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        return {
            'horizon_alignment': {'error': error, 'status': 'error'},
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


