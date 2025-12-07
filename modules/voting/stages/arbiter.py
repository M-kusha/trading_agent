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
    CONFIDENCE_THRESHOLD_F,
    CONSENSUS_THRESHOLD_F,
    VotingAction,
    is_training_mode,
    is_live_mode,
    get_voting_mode,
)
# Per-instrument voting infrastructure
from modules.voting.core.per_instrument import (
    DEFAULT_INSTRUMENTS,
    normalize_instrument,
    extract_instrument_data,
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
        # Configuration - NOTE: Use dynamic threshold functions to support live/training mode switching
        # Store config overrides; actual thresholds fetched dynamically via _get_thresholds()
        self._config_min_confidence = self.config.get('min_confidence')
        self._config_consensus_threshold = self.config.get('consensus_threshold')
        self.max_fragility = float(self.config.get('max_fragility', 0.9))  # Relaxed from 0.7
        self.max_collusion = float(self.config.get('max_collusion', 0.995))  # Very high - only flag obvious data issues
        self.bootstrap_steps = int(self.config.get('bootstrap_steps', 50))

        # Technical override: default OFF so agent/committee has full power
        self.technical_override_enabled = bool(self.config.get('technical_override_enabled', False))
        
        # Gate criteria weights
        self.criteria_weights = self.config.get(
            'criteria_weights',
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
        
        # Log initialization with current mode-aware thresholds
        mode = get_voting_mode()
        self.logger.info(
            f"[ARBITER] FinalArbiter initialized | MODE={mode} | "
            f"min_conf={self.min_confidence:.2f} | "
            f"consensus_thresh={self.consensus_threshold:.2f} | "
            f"technical_override_enabled={self.technical_override_enabled}"
        )
        
        # Publish baseline
        self._publish_arbiter_baseline()
    
    @property
    def min_confidence(self) -> float:
        """Get minimum confidence threshold (mode-aware)."""
        if self._config_min_confidence is not None:
            return float(self._config_min_confidence)
        return CONFIDENCE_THRESHOLD_F()
    
    @property
    def consensus_threshold(self) -> float:
        """Get consensus threshold (mode-aware)."""
        if self._config_consensus_threshold is not None:
            return float(self._config_consensus_threshold)
        return CONSENSUS_THRESHOLD_F()
    
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
        
        # Log current mode and thresholds at start of each processing cycle
        # This helps diagnose mode propagation issues
        mode = get_voting_mode()
        if self._step_count == 0 or self._step_count % 100 == 0:
            self.logger.info(
                f"[ARBITER] Processing step {self._step_count} | MODE={mode} | "
                f"thresholds: min_conf={self.min_confidence:.2f}, consensus={self.consensus_threshold:.2f}"
            )
        
        try:
            # Get decision ID
            decision_id = self.smart_bus.get('kernel_decision_id', name)
            
            # Collect all voting inputs
            data = await self._collect_voting_data()
            
            # Evaluate gate criteria
            gate_result = await self._evaluate_gate(data)
            
            # Generate final decision (global)
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
            # Per-instrument fragility from UncertaintySampler
            'instrument_fragility': self.smart_bus.get('instrument_fragility', name) or {},
            
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
        """Evaluate gate criteria (global gate)."""
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
        # Fragility is WARN-ONLY - does not block trades (often high due to Monte Carlo noise)
        fragility_ok = True  # Always pass - we just warn on high fragility
        fragility_warning = fragility >= self.max_fragility  # Track for logging
        memory_ok = memory_gate > 0.3
        
        # Final gate decision (fragility excluded from blocking criteria)
        all_passed = confidence_ok and consensus_ok and collusion_ok and memory_ok
        gate_passed = all_passed or weighted_score > 0.55
        
        if gate_passed:
            self._gate_passes += 1
            # Add warning if fragility is high even though we passed
            fragility_msg = f" ⚠️ HIGH_FRAGILITY={fragility:.2f}" if fragility_warning else ""
            self.logger.debug(
                f"[ARBITER] Global gate PASSED [MODE={get_voting_mode()}]: "
                f"score={weighted_score:.2f}, conf={confidence:.2f}, consensus={consensus:.2f}{fragility_msg}"
            )
        else:
            # Log detailed failure reasons as WARNING so operators can see why trades are blocked
            failed_criteria = []
            if not confidence_ok:
                effective_thresh = self.min_confidence * bootstrap_factor
                failed_criteria.append(f"confidence({confidence:.2f}<{effective_thresh:.2f})")
            if not consensus_ok:
                effective_thresh = self.consensus_threshold * bootstrap_factor
                failed_criteria.append(f"consensus({consensus:.2f}<{effective_thresh:.2f})")
            if not collusion_ok:
                failed_criteria.append(f"collusion({collusion:.2f}>{self.max_collusion:.2f})")
            # Fragility is WARN-ONLY - still log it but note it's not blocking
            if fragility_warning:
                failed_criteria.append(f"⚠️fragility({fragility:.2f}>{self.max_fragility:.2f})[warn-only]")
            if not memory_ok:
                failed_criteria.append(f"memory_gate({memory_gate:.2f}<0.30)")
            
            self.logger.warning(
                f"[ARBITER] Global gate BLOCKED [MODE={get_voting_mode()}]: "
                    f"score={weighted_score:.2f}<0.55 | "
                f"failed=[{', '.join(failed_criteria)}] | "
                f"thresholds: min_conf={self.min_confidence:.2f}, consensus={self.consensus_threshold:.2f}"
            )
        
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
        """Generate final trading decision (global)."""
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
        
        # Apply global fragility penalty
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
        """
        Generate per-instrument trading signals.
        
        NEW ARCHITECTURE (v2.0):
        - Reads per-instrument committee decisions from 'committee_decisions_by_instrument'
        - Each instrument gets its OWN direction based on expert votes for that instrument
        - Falls back to global decision + alignment filtering if per-instrument not available
        
        This allows EURUSD to be LONG while XAUUSD is SHORT if the experts voted differently.
        """
        signals: Dict[str, Dict[str, Any]] = {}
        name = self.__class__.__name__
        
        # ========== Try per-instrument committee decisions first ========== #
        per_inst_decisions = self.smart_bus.get('committee_decisions_by_instrument', name) or {}
        
        if per_inst_decisions:
            # Use per-instrument decisions from CommitteeCoordinator
            self.logger.debug(f"[ARBITER] Using per-instrument decisions: {list(per_inst_decisions.keys())}")
            
            collusion_score = float(data.get('collusion_score', 0.0))
            global_fragility = float(data.get('fragility', 0.5))
            # Use per-instrument fragility if available from UncertaintySampler
            instrument_fragility_map = data.get('instrument_fragility', {}) or {}
            market_regime = str(data.get('market_regime', 'UNKNOWN'))
            
            for inst in self.instruments:
                inst_normalized = normalize_instrument(inst)
                inst_decision = per_inst_decisions.get(inst_normalized) or per_inst_decisions.get(inst, {})
                
                if inst_decision:
                    inst_action = str(inst_decision.get('action', 'flat')).upper()
                    inst_confidence = float(inst_decision.get('confidence', 0.0))
                    inst_consensus = float(inst_decision.get('consensus_score', 0.0))
                    
                    # Get per-instrument fragility or fall back to global
                    inst_fragility = float(instrument_fragility_map.get(inst, global_fragility))
                    
                    # Apply gate criteria per instrument (may override action)
                    gate_passed, final_action, final_confidence = self._check_instrument_gate(
                        instrument=inst,
                        action=inst_action,
                        confidence=inst_confidence,
                        consensus_score=inst_consensus,
                        collusion_score=collusion_score,
                        fragility=inst_fragility,
                        market_regime=market_regime,
                    )
                    
                    if gate_passed and final_action.upper() in ('LONG', 'SHORT', 'BUY', 'SELL'):
                        # Convert to BUY/SELL format
                        if final_action.upper() in ('LONG', 'BUY'):
                            output_action = 'BUY'
                            intensity = round(final_confidence, 4)  # Positive for BUY
                        else:
                            output_action = 'SELL'
                            intensity = round(-final_confidence, 4)  # Negative for SELL
                        signals[inst] = {
                            'action': output_action,
                            'confidence': round(final_confidence, 4),
                            'size_multiplier': round(final_confidence, 4),
                            'intensity': intensity,  # For PositionManager compatibility
                            'instrument': inst,
                            'consensus_score': inst_consensus,
                            'gate_passed': True,
                            'reason': f'Per-instrument decision: {output_action} (conf={final_confidence:.2f})',
                            'source': 'per_instrument_committee',
                            'original_action': inst_action,  # Track if overridden
                        }
                    else:
                        signals[inst] = {
                            'action': 'HOLD',
                            'confidence': 0.0,
                            'size_multiplier': 0.0,
                            'intensity': 0.0,  # Zero intensity for HOLD
                            'instrument': inst,
                            'gate_passed': False,
                            'reason': f'Gate blocked or flat signal (action={inst_action}, gate={gate_passed})',
                            'source': 'per_instrument_committee',
                        }
                else:
                    # No decision for this instrument
                    signals[inst] = {
                        'action': 'HOLD',
                        'confidence': 0.0,
                        'size_multiplier': 0.0,
                        'intensity': 0.0,  # Zero intensity for no decision
                        'instrument': inst,
                        'reason': 'No committee decision for instrument',
                        'source': 'fallback',
                    }
            
            self.logger.info(
                f"[ARBITER] Per-instrument signals: " +
                ", ".join(f"{k}={v['action']}" for k, v in signals.items())
            )
            return signals
        
        # ========== Fallback: Use global decision with alignment filtering ========== #
        self.logger.debug("[ARBITER] Falling back to global decision + alignment")
        
        action = decision.get('action', 'abstain')
        base_confidence = decision.get('confidence', 0.0)
        
        # For HOLD/ABSTAIN, no instrument should trade
        if action.lower() in ('hold', 'abstain', 'unknown', 'flat'):
            for inst in self.instruments:
                signals[inst] = {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'size_multiplier': 0.0,
                    'instrument': inst,
                    'reason': f'Global decision is {action}',
                    'source': 'global_fallback',
                }
            return signals
        
        # Get market data for instrument filtering
        market_data = self.smart_bus.get('market_data', name) or {}
        price_data = self.smart_bus.get('price_data', name) or {}
        indicators = self.smart_bus.get('technical_indicators', name) or {}
        
        # Determine which instruments align with the global action
        aligned_instruments: List[str] = []
        instrument_scores: Dict[str, float] = {}
        
        for inst in self.instruments:
            inst_market = extract_instrument_data(market_data, inst)
            inst_price = extract_instrument_data(price_data, inst)
            inst_indicators = extract_instrument_data(indicators, inst)
            
            # Calculate instrument-specific alignment score
            alignment_score = self._calculate_instrument_alignment(
                inst, action, inst_market, inst_price, inst_indicators
            )
            instrument_scores[inst] = alignment_score
            
            if alignment_score > 0.3:  # Threshold for including instrument
                aligned_instruments.append(inst)
        
        # If no instruments align, force the one with highest score (but mark low confidence)
        if not aligned_instruments and instrument_scores:
            best_inst = max(instrument_scores, key=lambda k: instrument_scores.get(k, 0.0))
            aligned_instruments = [best_inst]
            base_confidence *= 0.5  # Reduce confidence if forcing
        
        # Generate signals for each instrument
        for inst in self.instruments:
            if inst in aligned_instruments:
                inst_confidence = base_confidence * instrument_scores.get(inst, 0.5)
                # Intensity: positive for BUY, negative for SELL
                if action.upper() in ('BUY', 'LONG'):
                    intensity = round(inst_confidence, 4)
                elif action.upper() in ('SELL', 'SHORT'):
                    intensity = round(-inst_confidence, 4)
                else:
                    intensity = 0.0
                signals[inst] = {
                    'action': action,
                    'confidence': round(inst_confidence, 4),
                    'size_multiplier': round(inst_confidence, 4),
                    'intensity': intensity,  # For PositionManager compatibility
                    'instrument': inst,
                    'alignment_score': instrument_scores.get(inst, 0.5),
                    'reason': f'Aligned with global {action}',
                    'source': 'global_with_alignment',
                }
            else:
                signals[inst] = {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'size_multiplier': 0.0,
                    'intensity': 0.0,  # Zero intensity for HOLD
                    'instrument': inst,
                    'alignment_score': instrument_scores.get(inst, 0.0),
                    'reason': f'Not aligned with global {action} (score={instrument_scores.get(inst, 0):.2f})'
                }
        
        self.logger.debug(f"[ARBITER] Instrument signals: aligned={aligned_instruments}, scores={instrument_scores}")
        
        return signals
    
    def _calculate_instrument_alignment(
        self,
        instrument: str,
        action: str,
        market_data: Dict[str, Any],
        price_data: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> float:
        """
        Calculate how well an instrument aligns with the proposed action.
        
        Uses instrument-specific candle data and indicators to determine
        if this instrument should trade in the proposed direction.
        
        Returns:
            Alignment score 0.0-1.0 (higher = better alignment)
        """
        try:
            is_buy = action.upper() == 'BUY'
            scores: List[float] = []
            weights: List[float] = []
            
            # 1. Candle direction (most important - actual price movement)
            open_price = float(market_data.get('open', price_data.get('open', 0)) or 0)
            close_price = float(market_data.get('close', price_data.get('close', 0)) or 0)
            
            if open_price > 0 and close_price > 0:
                candle_direction = 1 if close_price > open_price else -1
                # For BUY: green candle is good; For SELL: red candle is good
                if is_buy:
                    candle_score = 0.8 if candle_direction > 0 else 0.3
                else:
                    candle_score = 0.8 if candle_direction < 0 else 0.3
                scores.append(candle_score)
                weights.append(2.0)  # High weight for candle direction
            
            # 2. Trend alignment (using momentum/RSI if available)
            momentum = float(indicators.get('momentum', indicators.get('roc', 0.0)) or 0.0)
            rsi = float(indicators.get('rsi', 50.0) or 50.0)
            
            # For BUY: positive momentum and RSI < 70 is good
            # For SELL: negative momentum and RSI > 30 is good
            if is_buy:
                momentum_score = 0.5 + min(0.5, max(-0.5, momentum / 10))  # Normalize momentum
                rsi_score = 1.0 if rsi < 70 else max(0.0, 1.0 - (rsi - 70) / 30)
            else:
                momentum_score = 0.5 - min(0.5, max(-0.5, momentum / 10))
                rsi_score = 1.0 if rsi > 30 else max(0.0, (rsi) / 30)
            
            scores.extend([momentum_score, rsi_score])
            weights.extend([1.0, 1.0])
            
            # 3. Price position (relative to recent range)
            high = float(price_data.get('high', 0) or 0)
            low = float(price_data.get('low', 0) or 0)
            close = float(price_data.get('close', price_data.get('last', 0)) or 0)
            
            if high > low and close > 0:
                price_position = (close - low) / (high - low)  # 0=at low, 1=at high
                # For BUY: prefer lower price position; For SELL: prefer higher
                if is_buy:
                    position_score = 1.0 - price_position * 0.5  # Max 1.0, min 0.5
                else:
                    position_score = 0.5 + price_position * 0.5  # Max 1.0, min 0.5
                scores.append(position_score)
                weights.append(0.8)  # Medium weight for price position
            
            # 4. Volatility check (very high volatility reduces score)
            atr = float(indicators.get('atr', indicators.get('volatility', 0)) or 0)
            if atr > 0 and close > 0:
                atr_pct = atr / close
                if atr_pct > 0.02:  # High volatility (>2% ATR)
                    vol_score = max(0.3, 1.0 - (atr_pct - 0.02) / 0.03)
                else:
                    vol_score = 1.0
                scores.append(vol_score)
                weights.append(0.5)  # Lower weight for volatility
            
            # Calculate weighted average
            if scores and weights:
                total_weight = sum(weights[:len(scores)])
                weighted_sum = sum(s * w for s, w in zip(scores, weights[:len(scores)]))
                return weighted_sum / total_weight if total_weight > 0 else 0.5
            elif scores:
                return sum(scores) / len(scores)
            else:
                return 0.5  # Default moderate alignment
                
        except Exception as e:
            self.logger.warning(f"[ARBITER] Error calculating alignment for {instrument}: {e}")
            return 0.5  # Default moderate alignment
    
    def _check_technical_override(self, instrument: str, action: str) -> tuple:
        """
        Check if technical experts (Momentum + Trend) should override this trade.
        
        If BOTH MomentumExpert AND TrendExpert vote OPPOSITE to the proposed action
        with HIGH confidence, flip the direction. This protects capital while the
        PPO agent is still learning, but doesn't completely choke the agent.
        
        Override only happens when:
        - Both technical experts agree with each other
        - Both disagree with the proposed direction  
        - Both have confidence > 50% (strong technical signal)
        
        Returns:
            tuple: (should_override: bool, new_action: str, avg_confidence: float)
        """
        try:
            name = self.__class__.__name__
            inst_normalized = normalize_instrument(instrument)
            
            # Get per-instrument votes from technical experts
            momentum_votes = self.smart_bus.get('MomentumExpert_per_instrument_votes', name) or {}
            trend_votes = self.smart_bus.get('TrendExpert_per_instrument_votes', name) or {}
            
            # Get vote for this specific instrument
            momentum_vote = momentum_votes.get(inst_normalized, {})
            trend_vote = trend_votes.get(inst_normalized, {})
            
            momentum_action = str(momentum_vote.get('action', 'flat')).upper()
            trend_action = str(trend_vote.get('action', 'flat')).upper()
            momentum_conf = float(momentum_vote.get('confidence', 0.0))
            trend_conf = float(trend_vote.get('confidence', 0.0))
            
            # Normalize actions
            proposed = action.upper()
            if proposed in ('BUY', 'LONG'):
                proposed_direction = 'LONG'
            elif proposed in ('SELL', 'SHORT'):
                proposed_direction = 'SHORT'
            else:
                return (False, action, 0.0)  # No override for HOLD/FLAT
            
            # Check if both technical experts agree WITH EACH OTHER and disagree with proposal
            momentum_dir = 'LONG' if momentum_action == 'LONG' else 'SHORT' if momentum_action == 'SHORT' else None
            trend_dir = 'LONG' if trend_action == 'LONG' else 'SHORT' if trend_action == 'SHORT' else None
            
            # Both must have a directional opinion and agree with each other
            if momentum_dir and trend_dir and momentum_dir == trend_dir:
                # They agree with each other - check if they disagree with the proposal
                if momentum_dir != proposed_direction:
                    # Technical experts say opposite! 
                    # Only override if BOTH have strong confidence (>50%)
                    min_override_conf = 0.50  # Higher threshold - only strong technical signals
                    
                    if momentum_conf >= min_override_conf and trend_conf >= min_override_conf:
                        avg_conf = (momentum_conf + trend_conf) / 2
                        new_action = 'BUY' if momentum_dir == 'LONG' else 'SELL'
                        
                        self.logger.warning(
                            f"[ARBITER] TECHNICAL OVERRIDE for {instrument}: "
                            f"Flipping {proposed_direction} → {momentum_dir} | "
                            f"Momentum={momentum_action}({momentum_conf:.2f}), "
                            f"Trend={trend_action}({trend_conf:.2f}) | "
                            f"Reason: Both technical experts have strong opposing signals"
                        )
                        return (True, new_action, avg_conf)
                    else:
                        # Technical experts disagree but not strongly enough
                        self.logger.info(
                            f"[ARBITER] Technical disagreement noted but NOT overriding {instrument} {proposed_direction}: "
                            f"Momentum={momentum_action}({momentum_conf:.2f}), "
                            f"Trend={trend_action}({trend_conf:.2f}) - confidence too low for override"
                        )
            
            return (False, action, 0.0)  # No override
            
        except Exception as e:
            self.logger.warning(f"[ARBITER] Error checking technical override: {e}")
            return (False, action, 0.0)  # Don't override on error
    
    def _check_instrument_gate(
        self,
        instrument: str,
        action: str,
        confidence: float,
        consensus_score: float,
        collusion_score: float = 0.0,
        fragility: float = 0.5,
        market_regime: str = "UNKNOWN"
    ) -> tuple:
        """
        Check if an instrument-specific trade should pass through the gate.
        
        This is a simplified gate check for per-instrument decisions since
        the committee has already done much of the aggregation work.
        
        LIVE MODE: Very strict - only pass high-quality signals (target 3-4 trades/day)
        TRAINING MODE: More permissive for learning
        
        Returns:
            tuple: (gate_passed: bool, final_action: str, final_confidence: float)
        """
        try:
            # HOLD always passes (it's a non-trade)
            if action.upper() in ('HOLD', 'FLAT', 'ABSTAIN'):
                return (True, 'HOLD', 0.0)
            
            # ========== Technical Override Check (optional) ==========
            if self.technical_override_enabled:
                should_override, override_action, override_conf = self._check_technical_override(
                    instrument,
                    action
                )
                if should_override:
                    # Use technical experts' direction instead
                    action = override_action
                    confidence = max(confidence * 0.8, override_conf)
                    self.logger.info(
                        f"[ARBITER] Using technical override: {instrument} → {action} "
                        f"(conf={confidence:.2f})"
                    )
            
            # Base confidence threshold (mode-aware from constants.py)
            min_confidence = self.min_confidence
            min_consensus = self.consensus_threshold
            
            # Adjust threshold based on regime
            if is_training_mode():
                # TRAINING: minimal regime adjustments to allow more trades for learning
                regime_conf_adj = {
                    'TRENDING': -0.05,
                    'MEAN_REVERTING': 0.0,
                    'VOLATILE': 0.0,
                    'UNKNOWN': 0.0,
                }
                regime_consensus_adj = {}
            else:
                # LIVE: moderate regime adjustments - balance quality with opportunity
                regime_conf_adj = {
                    'TRENDING': -0.08,       # Easier in clear trends (trend following)
                    'MEAN_REVERTING': 0.03,  # Slightly harder in ranging markets
                    'VOLATILE': 0.08,        # Harder in volatile (was 0.15 - too strict)
                    'UNKNOWN': 0.05,         # Slightly harder when regime unclear
                }
                regime_consensus_adj = {
                    'TRENDING': 0.0,
                    'MEAN_REVERTING': 0.03,
                    'VOLATILE': 0.05,        # Need more consensus in volatile markets
                    'UNKNOWN': 0.03,
                }
            
            min_confidence += regime_conf_adj.get(market_regime.upper(), 0.0)
            min_consensus += regime_consensus_adj.get(market_regime.upper(), 0.0)
            
            # ========== LIVE MODE: Additional strict checks ==========
            if is_live_mode():
                # In LIVE mode, require BOTH high confidence AND high consensus
                # This ensures we only take trades when experts strongly agree
                
                # Check 1: Confidence must meet threshold
                if confidence < min_confidence:
                    self.logger.warning(
                        f"[ARBITER] 🚫 Gate BLOCKED {instrument} {action} [LIVE]: "
                        f"confidence {confidence:.2f} < {min_confidence:.2f} "
                        f"(base={self.min_confidence:.2f}, regime={market_regime})"
                    )
                    return (False, action, confidence)
                
                # Check 2: Consensus must be strong (experts must agree)
                if consensus_score < min_consensus:
                    self.logger.warning(
                        f"[ARBITER] 🚫 Gate BLOCKED {instrument} {action} [LIVE]: "
                        f"consensus {consensus_score:.2f} < {min_consensus:.2f} "
                        f"(need strong expert agreement)"
                    )
                    return (False, action, confidence)
                
                # Check 3: In volatile regime, require slightly higher standards
                if market_regime.upper() == 'VOLATILE' and confidence < 0.65:
                    self.logger.warning(
                        f"[ARBITER] 🚫 Gate BLOCKED {instrument} {action} [LIVE/VOLATILE]: "
                        f"confidence {confidence:.2f} < 0.65 (volatile market requires higher confidence)"
                    )
                    return (False, action, confidence)
            
            else:
                # TRAINING MODE: More permissive checks
                if confidence < min_confidence:
                    self.logger.warning(
                        f"[ARBITER] Gate BLOCKED {instrument} {action} [TRAINING]: "
                        f"confidence {confidence:.2f} < {min_confidence:.2f}"
                    )
                    return (False, action, confidence)
                
                if consensus_score < 0.3:
                    self.logger.warning(
                        f"[ARBITER] Gate BLOCKED {instrument} {action} [TRAINING]: "
                        f"low consensus {consensus_score:.2f} < 0.30"
                    )
                    return (False, action, confidence)
            
            # Check collusion (too much agreement is suspicious - but 98%+ is likely data issue)
            if collusion_score > 0.98:
                self.logger.warning(
                    f"[ARBITER] Gate BLOCKED {instrument} {action} [MODE={get_voting_mode()}]: "
                    f"extreme collusion {collusion_score:.2f} > 0.98"
                )
                return (False, action, confidence)
            
            # Log at INFO level so operators can see successful gate passes per instrument
            self.logger.info(
                f"[ARBITER] ✅ Gate PASSED {instrument} {action} [MODE={get_voting_mode()}]: "
                f"conf={confidence:.2f}, consensus={consensus_score:.2f}, regime={market_regime}"
            )
            return (True, action, confidence)
            
        except Exception as e:
            self.logger.warning(f"[ARBITER] Error in instrument gate check for {instrument}: {e}")
            return (False, action, confidence)  # Block on error for safety
    
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
