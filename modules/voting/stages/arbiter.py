"""
Final Arbiter
=============
Final gate decision for trading signals.
Combines all voting inputs (consensus, collusion, uncertainty, horizon)
to produce the final trading decision.

Refactored from strategy_arbiter.py (~2340 lines).
~500 lines focused on core arbitration logic.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase
from modules.voting.core.constants import (
    CONFIDENCE_THRESHOLD,
    CONSENSUS_THRESHOLD,
    VotingAction,
)


@module(**module_args("FinalArbiter"))
class FinalArbiter(VotingModuleBase):
    """
    Final arbitration for voting pipeline.
    
    Makes final gate decision based on:
    - Committee decision
    - Consensus score
    - Collusion score  
    - Fragility/uncertainty
    - Horizon alignment
    - Risk constraints
    
    Publishes:
    - final_decision
    - gate_passed
    - trading_signal
    - arbiter_analysis
    """
    
    def _module_specific_init(self) -> None:
        """Initialize arbiter state."""
        # Configuration
        self.min_confidence = float(self.config.get('min_confidence', CONFIDENCE_THRESHOLD))
        self.consensus_threshold = float(self.config.get('consensus_threshold', CONSENSUS_THRESHOLD))
        self.max_fragility = float(self.config.get('max_fragility', 0.7))
        self.max_collusion = float(self.config.get('max_collusion', 0.8))
        self.bootstrap_steps = int(self.config.get('bootstrap_steps', 50))
        
        # Gate criteria weights
        self.criteria_weights = self.config.get('criteria_weights', 
            [0.25, 0.20, 0.20, 0.20, 0.15]  # strength, consensus, reliability, risk, novelty
        )
        
        # State
        self._step_count = 0
        self._gate_passes = 0
        self._gate_attempts = 0
        
        # Member weights (for multi-expert blending)
        self.weights = np.ones(5, dtype=np.float32) / 5.0
        
        # History
        self.decision_history: deque = deque(maxlen=200)
        self.gate_decisions: deque = deque(maxlen=150)
        
        # Statistics
        self.arbiter_stats: Dict[str, Any] = {
            'total_decisions': 0,
            'gate_passes': 0,
            'gate_failures': 0,
            'gate_pass_rate': 0.0,
            'avg_confidence': 0.5,
        }
        
        # Quality metrics
        self.voting_quality: Dict[str, float] = {
            'avg_consensus': 0.5,
            'gate_effectiveness': 0.5,
            'decision_confidence': 0.5,
        }
        
        # Instruments (for per-instrument signals)
        self.instruments = self.config.get('instruments', ['EURUSD', 'XAUUSD'])
        
        self.logger.info(
            f"[ARBITER] FinalArbiter initialized | "
            f"min_conf={self.min_confidence:.2f} | "
            f"consensus_thresh={self.consensus_threshold:.2f}"
        )
        
        # Publish baseline
        self._publish_arbiter_baseline()
    
    def _publish_arbiter_baseline(self) -> None:
        """Publish baseline arbiter keys."""
        try:
            self.smart_bus.set(
                'final_decision',
                {'action': 'abstain', 'reason': 'initializing'},
                module='FinalArbiter',
                thesis='Baseline final decision'
            )
            self.smart_bus.set(
                'gate_passed',
                False,
                module='FinalArbiter',
                thesis='Baseline gate status'
            )
            self.smart_bus.set(
                'instruments',
                self.instruments,
                module='FinalArbiter',
                thesis='Instrument universe'
            )
        except Exception:
            pass

    def _extract_memory_gate(self, value: Any) -> float:
        """
        Normalize memory_gate into a numeric multiplier.
        UnifiedMemory publishes a dict; fall back gracefully to 1.0.
        """
        try:
            if isinstance(value, dict):
                for key in ('risk_multiplier', 'risk_score', 'score', 'value', 'gate'):
                    v = value.get(key)
                    if isinstance(v, (int, float)):
                        return float(v)
                # If veto flag is set, treat as closed gate.
                if value.get('veto') is True:
                    return 0.0
            if isinstance(value, (int, float)):
                return float(value)
        except Exception:
            pass
        return 1.0
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """Make final arbitration decision."""
        start = time.time()
        name = self.__class__.__name__
        
        try:
            # Get decision ID
            decision_id = self.smart_bus.get('kernel_decision_id', name)
            
            # Collect all voting inputs
            data = await self._collect_voting_data()
            
            # Evaluate gate criteria
            gate_result = await self._evaluate_gate(data)
            
            # Generate final decision
            final_decision = await self._generate_final_decision(data, gate_result)
            
            # Generate per-instrument signals
            instrument_signals = await self._generate_instrument_signals(final_decision, data)
            
            # Generate thesis
            thesis = self._generate_thesis(final_decision, gate_result)
            
            # Publish to bus
            await self._update_bus(final_decision, gate_result, instrument_signals, thesis)
            
            # Update statistics
            self._update_stats(gate_result)
            
            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(name, 'process', elapsed_ms, True)
            
            return {
                'final_decision': final_decision,
                'gate_passed': gate_result['passed'],
                'gate_result': gate_result,
                'trading_signal': final_decision,
                'instrument_signals': instrument_signals,
                'arbiter_analysis': {
                    'input_summary': self._summarize_inputs(data),
                    'gate_criteria': gate_result.get('criteria_scores', {}),
                    'final_confidence': final_decision.get('confidence', 0.0),
                },
                'arbiter_statistics': dict(self.arbiter_stats),
                'voting_quality': dict(self.voting_quality),
                'decision_id': decision_id,
                'arbiter_decision_id': decision_id,
                # Contract-expected keys
                'trade_vote': final_decision,
                'gate_decision': gate_result,
                'arbiter_thesis': thesis,
                'decision_confidence': final_decision.get('confidence', 0.0),
                'decision_rationale': final_decision.get('rationale', thesis),
                'arbiter_recommendations': {
                    'action': final_decision.get('action', 'abstain'),
                    'gate': gate_result['passed'],
                    'signals': instrument_signals
                },
                'member_weights': data.get('aligned_weights', {}),
                '_thesis': thesis,
            }
        
        except Exception as e:
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(e, 'arbiter_process')
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)
    
    async def _collect_voting_data(self) -> Dict[str, Any]:
        """Collect all voting inputs from SmartInfoBus."""
        name = self.__class__.__name__
        
        return {
            # Committee inputs
            'committee_decision': self.smart_bus.get('committee_decision', name) or {},
            'committee_confidence': float(self.smart_bus.get('committee_confidence', name) or 0.5),
            'trade_vote_v2': self.smart_bus.get('trade_vote_v2', name) or {},
            
            # Consensus inputs
            'consensus_score': float(self.smart_bus.get('consensus_score', name) or 0.5),
            'consensus_analysis': self.smart_bus.get('consensus_analysis', name) or {},
            
            # Collusion inputs
            'collusion_score': float(self.smart_bus.get('collusion_score', name) or 0.0),
            'suspicious_pairs': self.smart_bus.get('suspicious_pairs', name) or [],
            
            # Uncertainty inputs
            'fragility': float(self.smart_bus.get('fragility', name) or 0.5),
            'uncertainty_score': float(self.smart_bus.get('uncertainty_score', name) or 0.5),
            
            # Horizon inputs
            'horizon_alignment': self.smart_bus.get('horizon_alignment', name) or {},
            'aligned_weights': self.smart_bus.get('aligned_weights', name) or [],
            
            # Market context
            'market_regime': self.smart_bus.get('market_regime', name) or 'unknown',
            'session': self.smart_bus.get('session_canonical', name) or 'unknown',
            
            # Risk inputs
            'risk_data': self.smart_bus.get('risk_data', name) or {},
            # UnifiedMemory publishes a dict; prefer risk_multiplier/risk_score, else fallback to 1.0
            'memory_gate': self._extract_memory_gate(self.smart_bus.get('memory_gate', name)),
        }
    
    async def _evaluate_gate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate gate criteria."""
        self._step_count += 1
        self._gate_attempts += 1
        
        # During bootstrap, be lenient
        bootstrap_factor = 0.5 if self._step_count < self.bootstrap_steps else 1.0
        
        # Extract scores
        confidence = data.get('committee_confidence', 0.5)
        consensus = data.get('consensus_score', 0.5)
        collusion = data.get('collusion_score', 0.0)
        fragility = data.get('fragility', 0.5)
        memory_gate = data.get('memory_gate', 1.0)
        
        # Criteria evaluation
        criteria_scores = {
            'confidence': confidence,
            'consensus': consensus,
            'anti_collusion': 1.0 - collusion,
            'stability': 1.0 - fragility,
            'memory_clear': memory_gate,
        }
        
        # Weighted combination
        weights = self.criteria_weights
        if len(weights) < 5:
            weights = [0.2] * 5
        
        weighted_score = (
            weights[0] * criteria_scores['confidence'] +
            weights[1] * criteria_scores['consensus'] +
            weights[2] * criteria_scores['anti_collusion'] +
            weights[3] * criteria_scores['stability'] +
            weights[4] * criteria_scores['memory_clear']
        )
        
        # Individual thresholds
        confidence_ok = confidence >= (self.min_confidence * bootstrap_factor)
        consensus_ok = consensus >= (self.consensus_threshold * bootstrap_factor)
        collusion_ok = collusion < self.max_collusion
        fragility_ok = fragility < self.max_fragility
        memory_ok = memory_gate > 0.3
        
        # Final gate decision
        all_passed = confidence_ok and consensus_ok and collusion_ok and fragility_ok and memory_ok
        gate_passed = all_passed or weighted_score > 0.6
        
        if gate_passed:
            self._gate_passes += 1
        
        return {
            'passed': gate_passed,
            'weighted_score': weighted_score,
            'criteria_scores': criteria_scores,
            'criteria_passed': {
                'confidence': confidence_ok,
                'consensus': consensus_ok,
                'collusion': collusion_ok,
                'fragility': fragility_ok,
                'memory': memory_ok,
            },
            'bootstrap_active': self._step_count < self.bootstrap_steps,
        }
    
    async def _generate_final_decision(
        self, 
        data: Dict[str, Any], 
        gate_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final trading decision."""
        committee = data.get('committee_decision', {})
        trade_vote = data.get('trade_vote_v2', {})
        
        if not gate_result['passed']:
            fallback_action = committee.get('action', 'abstain')
            fallback_conf = float(data.get('committee_confidence', 0.1)) * 0.5
            return {
                'action': fallback_action,
                'confidence': max(0.05, fallback_conf),
                'reason': 'gate_failed',
                'gate_score': gate_result['weighted_score'],
                'original_action': committee.get('action', 'unknown'),
                'gate_passed': False,
            }
        
        # Use trade_vote_v2 if available, else committee
        action = trade_vote.get('action') or committee.get('action', 'abstain')
        confidence = float(trade_vote.get('confidence') or data.get('committee_confidence', 0.5))
        
        # Adjust confidence by gate score
        adjusted_confidence = confidence * gate_result['weighted_score']
        
        # Apply fragility penalty
        fragility = data.get('fragility', 0.5)
        if fragility > 0.5:
            adjusted_confidence *= (1.0 - (fragility - 0.5))
        
        # Cap confidence
        adjusted_confidence = max(0.1, min(1.0, adjusted_confidence))
        
        return {
            'action': action,
            'confidence': adjusted_confidence,
            'raw_confidence': confidence,
            'gate_score': gate_result['weighted_score'],
            'consensus_score': data.get('consensus_score', 0.5),
            'timestamp': datetime.datetime.now().isoformat(),
        }
    
    async def _generate_instrument_signals(
        self, 
        decision: Dict[str, Any], 
        data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate per-instrument trading signals."""
        signals = {}
        
        action = decision.get('action', 'abstain')
        confidence = decision.get('confidence', 0.0)
        
        for inst in self.instruments:
            # Basic signal (could be instrument-specific in future)
            signals[inst] = {
                'action': action,
                'confidence': confidence,
                'size_multiplier': confidence,  # Simple sizing by confidence
                'instrument': inst,
            }
        
        return signals
    
    def _summarize_inputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize voting inputs for analysis."""
        return {
            'committee_action': data.get('committee_decision', {}).get('action', 'unknown'),
            'committee_confidence': data.get('committee_confidence', 0.0),
            'consensus_score': data.get('consensus_score', 0.0),
            'collusion_score': data.get('collusion_score', 0.0),
            'fragility': data.get('fragility', 0.0),
            'regime': data.get('market_regime', 'unknown'),
        }
    
    def _update_stats(self, gate_result: Dict[str, Any]) -> None:
        """Update arbiter statistics."""
        self.arbiter_stats['total_decisions'] += 1
        
        if gate_result['passed']:
            self.arbiter_stats['gate_passes'] += 1
        else:
            self.arbiter_stats['gate_failures'] += 1
        
        self.arbiter_stats['gate_pass_rate'] = (
            self._gate_passes / max(1, self._gate_attempts)
        )
        
        # Track in history
        self.gate_decisions.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'passed': gate_result['passed'],
            'score': gate_result['weighted_score'],
        })
    
    def _generate_thesis(
        self, 
        decision: Dict[str, Any], 
        gate_result: Dict[str, Any]
    ) -> str:
        """Generate arbiter thesis."""
        action = decision.get('action', 'unknown')
        conf = decision.get('confidence', 0.0)
        passed = gate_result['passed']
        score = gate_result['weighted_score']
        
        status = 'PASSED' if passed else 'BLOCKED'
        
        return (
            f"ARBITER: {status} | action={action} | "
            f"confidence={conf:.1%} | gate_score={score:.2f} | "
            f"pass_rate={self.arbiter_stats['gate_pass_rate']:.1%}"
        )
    
    async def _update_bus(
        self, 
        decision: Dict[str, Any], 
        gate_result: Dict[str, Any],
        instrument_signals: Dict[str, Dict[str, Any]],
        thesis: str
    ) -> None:
        """Update SmartInfoBus with results."""
        try:
            name = self.__class__.__name__
            
            self.smart_bus.set(
                'final_decision',
                decision,
                module=name,
                thesis=thesis,
                confidence=decision.get('confidence', 0.0)
            )
            self.smart_bus.set(
                'gate_passed',
                gate_result['passed'],
                module=name,
                thesis=f'Gate {"passed" if gate_result["passed"] else "blocked"}'
            )
            self.smart_bus.set(
                'trading_signal',
                decision,
                module=name,
                thesis='Trading signal'
            )
            self.smart_bus.set(
                'instrument_signals',
                instrument_signals,
                module=name,
                thesis=f'Signals for {len(instrument_signals)} instruments'
            )
            self.smart_bus.set(
                'arbiter_analysis',
                {
                    'gate_result': gate_result,
                    'stats': dict(self.arbiter_stats),
                },
                module=name,
                thesis='Arbiter analysis'
            )
        except Exception as e:
            self.logger.warning(f"Bus update failed: {e}")
    
    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        return {
            'final_decision': {'action': 'abstain', 'reason': f'error: {error}', 'confidence': 0.0},
            'gate_passed': False,
            'gate_result': {'passed': False, 'error': error},
            'trading_signal': {'action': 'abstain'},
            'instrument_signals': {},
            'arbiter_analysis': {'error': error},
            'arbiter_statistics': dict(self.arbiter_stats),
            'voting_quality': dict(self.voting_quality),
            'decision_id': None,
            'arbiter_decision_id': None,
            # Contract-expected keys
            'trade_vote': {'action': 'abstain', 'confidence': 0.0},
            'gate_decision': {'passed': False, 'error': error},
            'arbiter_thesis': f'Arbiter error: {error}',
            'decision_confidence': 0.0,
            'decision_rationale': f'error: {error}',
            'arbiter_recommendations': {'action': 'abstain', 'gate': False, 'signals': {}},
            'member_weights': {},
            '_thesis': f'Arbiter error: {error}',
        }


