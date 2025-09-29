#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plain ASCII header: voting_wrappers.py

Enhanced voting expert wrappers & committee coordinator integrated with SmartInfoBus.
All non-ASCII decorative characters and BOM removed.
"""

import asyncio
import time
from modules.contracts import module_args
import numpy as np
import datetime
import math
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union, Type, cast
from collections import deque, defaultdict
from abc import ABC, abstractmethod

# -----------------------------
# MODERN SMARTINFOBUS IMPORTS
# -----------------------------
from modules.core.module_base import BaseModule, module
from modules.core.mixins import (
    SmartInfoBusTradingMixin, SmartInfoBusVotingMixin, SmartInfoBusStateMixin,
)
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker


class EnhancedVotingExpertBase(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusVotingMixin, SmartInfoBusStateMixin):
    """
    Enhanced voting expert base class with modern SmartInfoBus integration
    """

    def _initialize(self):
        """Initialize enhanced voting expert systems"""
        # Initialize all base mixins
        self._initialize_trading_state()
        self._initialize_voting_state()
        self._initialize_state_management()
        self._initialize_modern_systems()

        # Core expert configuration
        self.max_signal_strength = float(self.config.get('max_signal_strength', 1.0))
        self.confidence_threshold = float(self.config.get('confidence_threshold', 0.3))
        self.adaptive_scaling = bool(self.config.get('adaptive_scaling', True))
        self.market_awareness = bool(self.config.get('market_awareness', True))
        self.emergency_mode_sensitivity = float(self.config.get('emergency_mode_sensitivity', 0.8))

        # Enhanced state tracking
        self.action_history = deque(maxlen=self.config.get('max_history', 100))
        self.confidence_history = deque(maxlen=100)
        self.performance_metrics = defaultdict(lambda: {'count': 0, 'success': 0, 'avg_confidence': 0.5})

        # Market context awareness
        self.market_context = {
            'regime': 'unknown',
            'session': 'unknown',
            'volatility_level': 'medium',
            'risk_score': 0.0,
            'emergency_mode': False,
            'market_open': True
        }

        # Expert-specific intelligence parameters
        self.intelligence_parameters = {
            'learning_rate': 0.1,
            'adaptation_threshold': 0.15,
            'market_sensitivity': 0.8,
            'performance_memory': 0.9,
            'confidence_momentum': 0.85,
            'emergency_response_factor': 0.3
        }

        # Performance and quality tracking
        self.expert_analytics = {
            'total_actions': 0,
            'successful_actions': 0,
            'avg_confidence': 0.5,
            'market_regime_performance': defaultdict(float),
            'session_performance': defaultdict(float),
            'emergency_activations': 0,
            'circuit_breaker_activations': 0
        }

        # Circuit breaker and error handling
        self.circuit_breaker = {
            'failure_count': 0,
            'threshold': 5,
            'reset_time': 300,  # 5 minutes
            'last_failure': 0,
            'state': 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        }

        # Generate initialization thesis
        self._generate_initialization_thesis()

    def _initialize_modern_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name=f"{self.__class__.__name__}",
            log_path=f"logs/voting/{self.__class__.__name__.lower()}.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler(self.__class__.__name__, self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        expert_type = self.__class__.__name__.replace('Expert', '').replace('Enhanced', '')

        thesis = (
            f"Enhanced {expert_type} Voting Expert v4.1 Initialization Complete:\n"
            f" - SmartInfoBus zero-wiring integration; async + hot-reload\n"
            f" - Max signal: {self.max_signal_strength}; confidence threshold: {self.confidence_threshold:.1%}\n"
            f" - Market-aware; emergency sensitivity: {self.emergency_mode_sensitivity:.1%}\n"
            f" - Circuit breaker threshold: {self.circuit_breaker['threshold']}"
        )

        self.smart_bus.set(f'{self.__class__.__name__}_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'max_signal_strength': self.max_signal_strength,
                'confidence_threshold': self.confidence_threshold,
                'adaptive_scaling': self.adaptive_scaling,
                'market_awareness': self.market_awareness,
                'intelligence_parameters': self.intelligence_parameters
            }
        }, module=self.__class__.__name__, thesis=thesis)

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive voting integration

        Returns:
            Dict containing voting proposal, confidence, and analytics
        """
        start_time = time.time()

        try:
            # Circuit breaker check
            if not self._check_circuit_breaker():
                return self._generate_circuit_breaker_response()

            # Get comprehensive market data
            market_data = await self._get_comprehensive_market_data()

            # Update market context
            await self._update_market_context_comprehensive(market_data)

            # Check emergency mode
            emergency_status = await self._check_emergency_mode(market_data)

            # Generate voting proposal
            voting_proposal = await self._generate_voting_proposal(market_data, emergency_status)

            # Calculate enhanced confidence
            confidence = await self._calculate_enhanced_confidence(voting_proposal, market_data)

            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_thesis(voting_proposal, confidence, market_data)

            # Create comprehensive results
            results = {
                'voting_proposal': voting_proposal,
                'confidence': confidence,
                'thesis': thesis,
                'market_context': self.market_context.copy(),
                'expert_analytics': self._get_analytics_summary(),
                'emergency_status': emergency_status,
                'health_metrics': self._get_health_metrics()
            }

            # Update SmartInfoBus
            await self._update_smartinfobus_comprehensive(results, thesis)

            # Record performance
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric(self.__class__.__name__, 'process_time', processing_time, True)

            # Reset circuit breaker on success (also move HALF_OPEN -> CLOSED)
            self.circuit_breaker['failure_count'] = 0
            if self.circuit_breaker['state'] == 'HALF_OPEN':
                self.circuit_breaker['state'] = 'CLOSED'

            return results

        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    @abstractmethod
    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate expert-specific voting proposal - to be implemented by subclasses"""
        pass

    @abstractmethod
    async def _calculate_expert_specific_confidence(self, proposal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate expert-specific confidence - to be implemented by subclasses"""
        pass

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """SmartInfoBusVotingMixin implementation - propose voting action"""
        try:
            market_data = await self._get_comprehensive_market_data()
            emergency_status = await self._check_emergency_mode(market_data)
            return await self._generate_voting_proposal(market_data, emergency_status)
        except Exception as e:
            self.logger.error(f"[FAIL] Error in propose_action: {e}")
            return {'action': 'abstain', 'reason': f'Error in proposal: {str(e)}'}

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """SmartInfoBusVotingMixin implementation - calculate action confidence"""
        try:
            market_data = await self._get_comprehensive_market_data()
            return await self._calculate_enhanced_confidence(action, market_data)
        except Exception as e:
            self.logger.error(f"[FAIL] Error in calculate_confidence: {e}")
            return 0.3  # Conservative default

    async def _get_comprehensive_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data from SmartInfoBus (post-migration: no 'voting_consensus')."""
        try:
            get = self.smart_bus.get
            return {
                'market_regime': get('market_regime', self.__class__.__name__) or 'unknown',
                'session_type': get('session_type', self.__class__.__name__) or 'unknown',
                'volatility_data': get('volatility_data', self.__class__.__name__) or {},
                'risk_score': get('risk_score', self.__class__.__name__) or 0.0,
                'market_open': get('market_open', self.__class__.__name__, default=True),
                'emergency_mode': get('emergency_mode', self.__class__.__name__, default=False),
                'portfolio_state': get('portfolio_state', self.__class__.__name__) or {},
                'recent_trades': get('recent_trades', self.__class__.__name__) or [],
                'expert_performance': get('expert_performance', self.__class__.__name__) or {},
                # Post-migration canonical surfaces:
                'consensus_score': get('consensus_score', self.__class__.__name__, default=None),
                'committee_consensus': get('committee_consensus', self.__class__.__name__) or {},
                'system_health': get('system_health', self.__class__.__name__) or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_data_retrieval")
            self.logger.warning(f"Market data retrieval incomplete: {error_context}")
            return self._get_safe_market_defaults()


    async def _update_market_context_comprehensive(self, market_data: Dict[str, Any]):
        """Update comprehensive market context tracking"""
        try:
            old_context = self.market_context.copy()

            self.market_context.update({
                'regime': market_data.get('market_regime', 'unknown'),
                'session': market_data.get('session_type', 'unknown'),
                'volatility_level': self._determine_volatility_level(market_data.get('volatility_data', {})),
                'risk_score': float(market_data.get('risk_score', 0.0) or 0.0),
                'emergency_mode': bool(market_data.get('emergency_mode', False)),
                'market_open': bool(market_data.get('market_open', True))
            })

            # Log significant context changes
            if old_context.get('regime') != self.market_context['regime']:
                self.logger.info(format_operator_message(
                    icon="[STATS]",
                    message="Market regime changed",
                    old_regime=old_context.get('regime'),
                    new_regime=self.market_context['regime'],
                    impact="Expert strategy will adapt"
                ))

            if old_context.get('emergency_mode') != self.market_context['emergency_mode']:
                self.logger.warning(format_operator_message(
                    icon="[ALERT]" if self.market_context['emergency_mode'] else "[INFO]",
                    message="Emergency mode status changed",
                    emergency_active=self.market_context['emergency_mode'],
                    impact="Risk parameters will adjust"
                ))

                if self.market_context['emergency_mode']:
                    self.expert_analytics['emergency_activations'] += 1

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_context_update")
            self.logger.warning(f"Market context update failed: {error_context}")

    async def _check_emergency_mode(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check and respond to emergency mode conditions"""
        try:
            emergency_active = bool(market_data.get('emergency_mode', False))
            risk_score = float(market_data.get('risk_score', 0.0) or 0.0)

            if emergency_active or risk_score > self.emergency_mode_sensitivity:
                response_level = 'HIGH' if risk_score > 0.9 else 'MEDIUM' if risk_score > 0.7 else 'LOW'
                return {
                    'emergency_active': True,
                    'response_level': response_level,
                    'risk_score': risk_score,
                    'recommended_action': self._determine_emergency_action(response_level),
                    'signal_adjustment': self._calculate_emergency_signal_adjustment(risk_score)
                }

            return {
                'emergency_active': False,
                'response_level': 'NORMAL',
                'risk_score': risk_score,
                'recommended_action': 'continue_normal',
                'signal_adjustment': 1.0
            }

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "emergency_mode_check")
            return {'emergency_active': False, 'response_level': 'UNKNOWN', 'signal_adjustment': 0.5}

    def _determine_emergency_action(self, response_level: str) -> str:
        """Determine appropriate emergency action"""
        emergency_actions = {'HIGH': 'reduce_positions', 'MEDIUM': 'conservative_sizing', 'LOW': 'cautious_monitoring'}
        return emergency_actions.get(response_level, 'monitor')

    def _calculate_emergency_signal_adjustment(self, risk_score: float) -> float:
        """Calculate signal strength adjustment for emergency conditions"""
        base_factor = float(self.intelligence_parameters['emergency_response_factor'])
        adjustment = 1.0 - (float(risk_score) * base_factor * 2.0)
        return max(0.1, min(1.0, adjustment))

    async def _generate_voting_proposal(self, market_data: Dict[str, Any], emergency_status: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive voting proposal"""
        try:
            base_proposal = await self._generate_expert_specific_proposal(market_data)

            # Apply emergency adjustments
            if emergency_status.get('emergency_active', False):
                base_proposal = self._apply_emergency_adjustments(base_proposal, emergency_status)

            # Apply market context adjustments
            adjusted_proposal = await self._apply_market_context_adjustments(base_proposal, market_data)

            # Record proposal
            self._record_action_proposal(adjusted_proposal)
            return adjusted_proposal

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "voting_proposal_generation")
            self.logger.error(f"Voting proposal generation failed: {error_context}")
            return {'action': 'abstain', 'reason': f'Proposal generation failed: {error_context}'}

    def _apply_emergency_adjustments(self, proposal: Dict[str, Any], emergency_status: Dict[str, Any]) -> Dict[str, Any]:
        """Apply emergency mode adjustments to proposal"""
        try:
            adjustment_factor = float(emergency_status.get('signal_adjustment', 0.5))
            response_level = emergency_status.get('response_level', 'LOW')

            if 'signal_strength' in proposal and isinstance(proposal['signal_strength'], (int, float)):
                proposal['signal_strength'] *= adjustment_factor

            if 'position_size' in proposal and isinstance(proposal['position_size'], (int, float)):
                proposal['position_size'] *= adjustment_factor

            proposal['emergency_adjustment'] = {
                'applied': True,
                'response_level': response_level,
                'adjustment_factor': adjustment_factor,
                'reason': f'Emergency mode active with {response_level} response level'
            }

            self.logger.warning(format_operator_message(
                icon="[ALERT]",
                message="Emergency adjustments applied to proposal",
                response_level=response_level,
                adjustment_factor=f"{adjustment_factor:.2f}"
            ))

            return proposal

        except Exception as e:
            self.logger.error(f"Emergency adjustment failed: {e}")
            return proposal

    async def _apply_market_context_adjustments(self, proposal: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply market context-specific adjustments"""
        try:
            regime = self.market_context.get('regime', 'unknown')
            session = self.market_context.get('session', 'unknown')
            volatility = self.market_context.get('volatility_level', 'medium')

            regime_multipliers = {
                'trending': 1.2, 'volatile': 0.8, 'ranging': 1.0,
                'breakout': 1.3, 'reversal': 0.9, 'noise': 0.6, 'unknown': 0.8
            }
            session_multipliers = {
                'american': 1.1, 'european': 1.0, 'asian': 0.9,
                'rollover': 0.4, 'weekend': 0.2, 'unknown': 0.8
            }
            volatility_multipliers = {
                'extreme': 0.5, 'high': 0.7, 'medium': 1.0, 'low': 1.2, 'very_low': 1.3
            }

            combined_multiplier = (
                regime_multipliers.get(regime, 0.8)
                * session_multipliers.get(session, 0.8)
                * volatility_multipliers.get(volatility, 1.0)
            )

            if 'signal_strength' in proposal and isinstance(proposal['signal_strength'], (int, float)):
                proposal['signal_strength'] *= combined_multiplier

            proposal['market_adjustments'] = {
                'regime_multiplier': regime_multipliers.get(regime, 0.8),
                'session_multiplier': session_multipliers.get(session, 0.8),
                'volatility_multiplier': volatility_multipliers.get(volatility, 1.0),
                'combined_multiplier': combined_multiplier,
                'regime': regime,
                'session': session,
                'volatility_level': volatility
            }

            return proposal

        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "market_context_adjustments")
            return proposal

    async def _calculate_enhanced_confidence(self, proposal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate enhanced confidence with market awareness"""
        try:
            base_confidence = await self._calculate_expert_specific_confidence(proposal, market_data)
            context_adjusted = self._apply_confidence_context_adjustments(base_confidence, market_data)
            performance_adjusted = self._apply_confidence_performance_adjustments(context_adjusted)
            emergency_adjusted = self._apply_confidence_emergency_adjustments(
                performance_adjusted, market_data.get('emergency_mode', False)
            )
            final_confidence = max(0.0, min(1.0, float(emergency_adjusted)))
            self.confidence_history.append(final_confidence)
            return final_confidence
        except Exception:
            return max(0.1, self.confidence_threshold)

    def _apply_confidence_context_adjustments(self, base_confidence: float, market_data: Dict[str, Any]) -> float:
        """Apply market context adjustments to confidence"""
        try:
            regime = self.market_context.get('regime', 'unknown')
            session = self.market_context.get('session', 'unknown')
            risk_score = float(market_data.get('risk_score', 0.0) or 0.0)

            regime_confidence_factors = {
                'trending': 1.1, 'volatile': 0.8, 'ranging': 0.9,
                'breakout': 1.2, 'reversal': 0.7, 'noise': 0.6, 'unknown': 0.7
            }
            session_confidence_factors = {
                'american': 1.0, 'european': 0.95, 'asian': 0.9,
                'rollover': 0.5, 'weekend': 0.3, 'unknown': 0.8
            }

            regime_factor = regime_confidence_factors.get(regime, 0.7)
            session_factor = session_confidence_factors.get(session, 0.8)
            risk_factor = 1.0 - (risk_score * 0.3)

            adjusted_confidence = float(base_confidence) * regime_factor * session_factor * risk_factor
            return max(0.1, min(1.0, adjusted_confidence))
        except Exception:
            return base_confidence

    def _apply_confidence_performance_adjustments(self, confidence: float) -> float:
        """Apply performance-based confidence adjustments"""
        try:
            if len(self.confidence_history) < 5:
                return confidence
            recent_confidences = list(self.confidence_history)[-10:]
            avg_recent = float(np.mean(recent_confidences))
            momentum = float(self.intelligence_parameters['confidence_momentum'])
            performance_factor = (avg_recent - 0.5) * momentum + 1.0
            adjusted = float(confidence) * performance_factor
            return max(0.1, min(1.0, adjusted))
        except Exception:
            return confidence

    def _apply_confidence_emergency_adjustments(self, confidence: float, emergency_mode: bool) -> float:
        """Apply emergency mode confidence adjustments"""
        if emergency_mode:
            emergency_factor = 1.0 - float(self.intelligence_parameters['emergency_response_factor'])
            return float(confidence) * emergency_factor
        return float(confidence)

    async def _generate_comprehensive_thesis(self, proposal: Dict[str, Any], confidence: float, market_data: Dict[str, Any]) -> str:
        """Generate comprehensive decision thesis"""
        try:
            expert_type = self.__class__.__name__.replace('Enhanced', '').replace('Expert', '')
            parts = []
            label = "HIGH" if confidence > 0.7 else "MODERATE" if confidence > 0.4 else "LOW"
            parts.append(f"{expert_type.upper()} EXPERT DECISION: {label} confidence ({confidence:.1%})")
            parts.append(f"PROPOSED ACTION: {proposal.get('action', 'unknown')}")
            parts.append(f"MARKET CONTEXT: {self.market_context.get('regime','unknown')} regime during {self.market_context.get('session','unknown')} session")
            if market_data.get('emergency_mode', False):
                parts.append("EMERGENCY MODE: Active risk protocols engaged")
            total_actions = self.expert_analytics.get('total_actions', 0)
            if total_actions > 0:
                sr = self.expert_analytics.get('successful_actions', 0) / total_actions
                parts.append(f"EXPERT PERFORMANCE: {sr:.1%} success over {total_actions} actions")
            return " | ".join(parts)
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Thesis generation failed: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive results (and optional normalized feed publish)."""
        try:
            name = self.__class__.__name__
            proposal = results['voting_proposal']
            confidence = float(results['confidence'])

            # Existing per-expert publications
            self.smart_bus.set(f'{name}_voting_proposal', proposal, module=name, thesis=thesis, confidence=confidence)
            self.smart_bus.set(f'{name}_confidence', confidence, module=name, thesis=f"{name} confidence: {confidence:.1%}")
            self.smart_bus.set(f'{name}_market_context', results['market_context'], module=name, thesis=f"Market context awareness for {name}")
            self.smart_bus.set(f'{name}_analytics', results['expert_analytics'], module=name, thesis=f"Performance analytics for {name}")

            # Optional: publish into normalized expert_votes feed (feed-first coordination)
            # Default enabled to reduce BUS MISS for 'expert_votes' in coordinators
            if bool(self.config.get('publish_to_expert_votes_feed', True)):
                try:
                    feed_key = self.config.get('expert_votes_bus_key', 'expert_votes')
                    # Guard: if a canonical provider exists (e.g., Coordinator), skip optional per-expert writes
                    try:
                        providers = self.smart_bus.get_providers(feed_key) or []
                    except Exception:
                        providers = []
                    if feed_key == 'expert_votes' and providers and (name not in providers):
                        # Avoid owner/conflict warnings; coordinator will publish the snapshot
                        pass
                    else:
                        entry = {
                            'expert': name,
                            'vote': dict(proposal),
                            'confidence': confidence,
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                        buf = self.smart_bus.get(feed_key, name) or []
                        if not isinstance(buf, list):
                            buf = []
                        # de-duplicate same expert (keep most recent)
                        buf = [e for e in buf if e.get('expert') != name]
                        buf.append(entry)

                        # ring buffer cap
                        cap = int(self.config.get('max_expert_votes_buffer', 200))
                        if len(buf) > cap:
                            buf = buf[-cap:]

                        self.smart_bus.set(feed_key, buf, module=name, thesis=f"{name} published normalized vote entry")
                except Exception as e:
                    self.logger.warning(f"Soft-fail publishing to expert_votes feed: {e}")

            # Always emit a lightweight stream event for diagnostics (no provider churn)
            try:
                stream_entry = {
                    'expert': name,
                    'vote': dict(proposal),
                    'confidence': float(confidence),
                    'timestamp': datetime.datetime.now().isoformat(),
                }
                # 'vote' is configured as a stream key on the bus via set_policy('vote', mode='stream')
                self.smart_bus.publish('vote', stream_entry, module=name, thesis=f"{name} vote stream")
            except Exception:
                pass
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    def _record_action_proposal(self, proposal: Dict[str, Any]):
        """Record action proposal for tracking"""
        try:
            record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'proposal': proposal,
                'market_context': self.market_context.copy(),
                'expert_analytics': self.expert_analytics.copy()
            }
            self.action_history.append(record)
            self.expert_analytics['total_actions'] += 1
        except Exception as e:
            self.logger.warning(f"Action recording failed: {e}")

    def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker status"""
        cb = self.circuit_breaker
        now = time.time()
        if cb['state'] == 'OPEN':
            if now - cb['last_failure'] > cb['reset_time']:
                cb['state'] = 'HALF_OPEN'
                cb['failure_count'] = 0
                self.logger.info("Circuit breaker moved to HALF_OPEN")
            else:
                return False
        return cb['state'] in ['CLOSED', 'HALF_OPEN']

    def _generate_circuit_breaker_response(self) -> Dict[str, Any]:
        """Generate response when circuit breaker is open"""
        return {
            'voting_proposal': {'action': 'abstain', 'reason': 'circuit_breaker_open'},
            'confidence': 0.0,
            'thesis': f"{self.__class__.__name__} circuit breaker is open due to repeated failures",
            'market_context': self.market_context.copy(),
            'expert_analytics': self._get_analytics_summary(),
            'emergency_status': {'circuit_breaker_open': True},
            'health_metrics': {'status': 'circuit_breaker_open'}
        }

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.circuit_breaker['failure_count'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        if self.circuit_breaker['failure_count'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            self.expert_analytics['circuit_breaker_activations'] += 1

        error_context = self.error_pinpointer.analyze_error(error, self.__class__.__name__)
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric(self.__class__.__name__, 'process_time', processing_time, False, error=str(error_context))

        return {
            'voting_proposal': {'action': 'abstain', 'reason': f'processing_error: {str(error_context)}'},
            'confidence': 0.1,
            'thesis': f"Processing error in {self.__class__.__name__}: {error_context}",
            'market_context': self.market_context.copy(),
            'expert_analytics': {'error': str(error_context)},
            'emergency_status': {'processing_error': True},
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def _get_safe_market_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when market data retrieval fails (no legacy 'voting_consensus')."""
        return {
            'market_regime': 'unknown',
            'session_type': 'unknown',
            'volatility_data': {},
            'risk_score': 0.5,
            'market_open': True,
            'emergency_mode': False,
            'portfolio_state': {},
            'recent_trades': [],
            'expert_performance': {},
            'consensus_score': None,
            'committee_consensus': {},
            'system_health': {}
        }

    def _determine_volatility_level(self, volatility_data: Dict[str, Any]) -> str:
        """Determine current volatility level"""
        try:
            if not volatility_data:
                return 'medium'
            vals = [float(v) for v in volatility_data.values() if isinstance(v, (int, float))]
            avg = float(np.mean(vals)) if vals else 0.02
            if avg > 0.05: return 'extreme'
            if avg > 0.03: return 'high'
            if avg > 0.015: return 'medium'
            if avg > 0.008: return 'low'
            return 'very_low'
        except Exception:
            return 'medium'

    def _get_analytics_summary(self) -> Dict[str, Any]:
        """Get expert analytics summary"""
        try:
            total = int(self.expert_analytics.get('total_actions', 0))
            success = int(self.expert_analytics.get('successful_actions', 0))
            return {
                'total_actions': total,
                'successful_actions': success,
                'success_rate': (success / max(1, total)),
                'avg_confidence': float(self.expert_analytics.get('avg_confidence', 0.5)),
                'emergency_activations': int(self.expert_analytics.get('emergency_activations', 0)),
                'circuit_breaker_activations': int(self.expert_analytics.get('circuit_breaker_activations', 0))
            }
        except Exception:
            return {'status': 'error'}

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring"""
        total = int(self.expert_analytics.get('total_actions', 0))
        success = int(self.expert_analytics.get('successful_actions', 0))
        return {
            'module_name': self.__class__.__name__,
            'status': 'circuit_breaker_open' if self.circuit_breaker['state'] == 'OPEN' else 'healthy',
            'circuit_breaker_state': self.circuit_breaker['state'],
            'failure_count': self.circuit_breaker['failure_count'],
            'total_actions': total,
            'success_rate': (success / max(1, total)),
            'avg_confidence': float(self.expert_analytics.get('avg_confidence', 0.5)),
            'market_regime': self.market_context.get('regime', 'unknown'),
            'emergency_mode': self.market_context.get('emergency_mode', False)
        }

    def get_health_status(self) -> Dict[str, Any]:
        """
        Custom health status for voting experts.
        Returns detailed health information about expert performance and state.
        """
        try:
            # Get base health from BaseModule
            base_health = super().get_health_status()

            # Calculate expert-specific metrics
            total_actions = self.expert_analytics.get('total_actions', 0)
            successful_actions = self.expert_analytics.get('successful_actions', 0)
            success_rate = successful_actions / max(total_actions, 1)

            # Determine health status
            status = base_health.get('status', 'OK')
            is_healthy = base_health.get('is_healthy', True)
            issues = []

            # Check circuit breaker state
            cb_state = self.circuit_breaker.get('state', 'CLOSED')
            if cb_state == 'OPEN':
                status = 'DEGRADED'
                is_healthy = False
                issues.append(f"Circuit breaker OPEN ({self.circuit_breaker.get('failure_count', 0)} failures)")
            elif cb_state == 'HALF_OPEN':
                status = 'WARNING'
                issues.append("Circuit breaker in HALF_OPEN state (recovery testing)")

            # Check success rate
            if total_actions > 10 and success_rate < 0.5:
                status = 'DEGRADED'
                is_healthy = False
                issues.append(f"Low success rate: {success_rate:.1%}")
            elif total_actions > 10 and success_rate < 0.7:
                if status == 'OK':
                    status = 'WARNING'
                issues.append(f"Suboptimal success rate: {success_rate:.1%}")

            # Check emergency mode
            if self.market_context.get('emergency_mode', False):
                if status == 'OK':
                    status = 'WARNING'
                issues.append("Operating in emergency mode")

            # Enhanced health info
            return {
                **base_health,
                'status': status,
                'is_healthy': is_healthy,
                'total_actions': total_actions,
                'successful_actions': successful_actions,
                'success_rate': success_rate,
                'avg_confidence': self.expert_analytics.get('avg_confidence', 0.5),
                'circuit_breaker_state': cb_state,
                'circuit_breaker_failures': self.circuit_breaker.get('failure_count', 0),
                'emergency_activations': self.expert_analytics.get('emergency_activations', 0),
                'market_regime': self.market_context.get('regime', 'unknown'),
                'market_open': self.market_context.get('market_open', True),
                'issues': issues if issues else None,
                'performance': {
                    **base_health.get('performance', {}),
                    'action_history_size': len(self.action_history),
                    'confidence_history_size': len(self.confidence_history),
                    'market_regimes_tracked': len(self.expert_analytics.get('market_regime_performance', {}))
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'module': self.__class__.__name__,
                'version': self.metadata.version,
                'is_healthy': False,
                'error': str(e),
                'last_error': str(e)
            }

    def reset(self):
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self.action_history.clear()
        self.confidence_history.clear()
        self.performance_metrics.clear()
        self.market_context = {
            'regime': 'unknown', 'session': 'unknown', 'volatility_level': 'medium',
            'risk_score': 0.0, 'emergency_mode': False, 'market_open': True
        }
        self.expert_analytics = {
            'total_actions': 0, 'successful_actions': 0, 'avg_confidence': 0.5,
            'market_regime_performance': defaultdict(float), 'session_performance': defaultdict(float),
            'emergency_activations': 0, 'circuit_breaker_activations': 0
        }
        self.circuit_breaker = {
            'failure_count': 0, 'threshold': 5, 'reset_time': 300,
            'last_failure': 0, 'state': 'CLOSED'
        }
        self.logger.info(format_operator_message(
            icon="[RELOAD]",
            message=f"{self.__class__.__name__} reset completed",
            status="All expert state cleared and systems reinitialized"
        ))


# =============================
# ENHANCED THEME EXPERT
# =============================

@module(**module_args(
    "EnhancedThemeExpert",
    description="Enhanced theme-based trading expert with modern InfoBus integration",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class EnhancedThemeExpert(EnhancedVotingExpertBase):
    """
    PRODUCTION-GRADE Enhanced Theme Expert v4.1
    """

    def _initialize(self):
        """Initialize enhanced theme expert"""
        super()._initialize()

        if not hasattr(self, 'performance_tracker') or self.performance_tracker is None:
            from modules.monitoring.performance_tracker import PerformanceTracker
            self.performance_tracker = PerformanceTracker()

        # Theme-specific configuration
        self.theme_sensitivity = float(self.config.get('theme_sensitivity', 0.8))
        self.theme_momentum = float(self.config.get('theme_momentum', 0.9))
        self.theme_decay_factor = float(self.config.get('theme_decay_factor', 0.95))

        # Theme state tracking
        self.current_theme = 0
        self.theme_strength = 0.0
        self.theme_history = deque(maxlen=50)
        self.theme_performance = {
            0: {'signals': 0, 'success': 0, 'avg_strength': 0.0},  # Risk-on
            1: {'signals': 0, 'success': 0, 'avg_strength': 0.0},  # Risk-off
            2: {'signals': 0, 'success': 0, 'avg_strength': 0.0},  # High volatility
            3: {'signals': 0, 'success': 0, 'avg_strength': 0.0},  # Trending
        }

        self.logger.info(format_operator_message(
            icon="[THEME]",
            message="Enhanced Theme Expert v4.1 initialized",
            theme_sensitivity=self.theme_sensitivity,
            theme_momentum=self.theme_momentum
        ))

        # Publish baseline provides to SmartInfoBus to prevent early BUS MISS
        try:
            name = self.__class__.__name__
            baseline_proposal = {
                'action': 'neutral',
                'signal_strength': 0.0,
                'position_size': 0.0,
                'duration': 'short',
                'theme_type': 'unknown'
            }
            self.smart_bus.set(f'{name}_voting_proposal', baseline_proposal, module=name, thesis='Baseline theme voting proposal')
            self.smart_bus.set(f'{name}_confidence', 0.1, module=name, thesis=f"{name} baseline confidence: 10%")
            self.smart_bus.set(f'{name}_market_context', self.market_context, module=name, thesis=f"Baseline market context for {name}")
            self.smart_bus.set(f'{name}_analytics', self.expert_analytics, module=name, thesis=f"Baseline performance analytics for {name}")
        except Exception:
            pass



    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate theme-based voting proposal"""
        try:
            theme_data = self.smart_bus.get('theme_detection', self.__class__.__name__) or {}
            self.current_theme = int(theme_data.get('current_theme', 0))
            self.theme_strength = float(theme_data.get('theme_strength', 0.0) or 0.0)

            if self.current_theme == 0:  # Risk-on
                proposal = {
                    'action': 'long_risk_assets',
                    'signal_strength': self.theme_strength * 0.8,
                    'position_size': self.theme_strength * self.max_signal_strength,
                    'duration': 'medium',
                    'theme_type': 'risk_on'
                }
            elif self.current_theme == 1:  # Risk-off
                proposal = {
                    'action': 'safe_haven_rotation',
                    'signal_strength': self.theme_strength * 0.9,
                    'position_size': self.theme_strength * self.max_signal_strength,
                    'duration': 'long',
                    'theme_type': 'risk_off'
                }
            elif self.current_theme == 2:  # High volatility
                proposal = {
                    'action': 'volatility_hedging',
                    'signal_strength': self.theme_strength * 0.6,
                    'position_size': self.theme_strength * self.max_signal_strength * 0.5,
                    'duration': 'short',
                    'theme_type': 'high_volatility'
                }
            elif self.current_theme == 3:  # Trending
                proposal = {
                    'action': 'trend_following',
                    'signal_strength': self.theme_strength,
                    'position_size': self.theme_strength * self.max_signal_strength,
                    'duration': 'long',
                    'theme_type': 'trending'
                }
            else:
                proposal = {
                    'action': 'neutral',
                    'signal_strength': 0.3,
                    'position_size': 0.1,
                    'duration': 'short',
                    'theme_type': 'unknown'
                }

            proposal['theme_metadata'] = {
                'current_theme': self.current_theme,
                'theme_strength': self.theme_strength,
                'theme_performance': self.theme_performance.get(self.current_theme, {}),
                'theme_momentum': self._calculate_theme_momentum()
            }

            self._record_theme_signal()
            return proposal

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "theme_proposal_generation")
            return {'action': 'abstain', 'reason': f'Theme analysis failed: {error_context}', 'signal_strength': 0.0}

    async def _calculate_expert_specific_confidence(self, proposal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate theme-specific confidence"""
        try:
            base = float(self.theme_strength) * float(self.theme_sensitivity)
            perf = self.theme_performance.get(self.current_theme, {})
            if perf.get('signals', 0) > 5:
                sr = perf.get('success', 0) / perf['signals']
                base *= (0.5 + sr * 0.5)
            momentum = self._calculate_theme_momentum()
            base *= (0.8 + momentum * 0.4)
            regime = market_data.get('market_regime', 'unknown')
            if self._is_theme_regime_aligned(self.current_theme, regime):
                base *= 1.2
            return max(0.1, min(1.0, base))
        except Exception:
            return 0.5

    def _calculate_theme_momentum(self) -> float:
        """Calculate theme momentum from recent history"""
        try:
            if len(self.theme_history) < 3:
                return 0.5
            recent_themes = [e['theme'] for e in list(self.theme_history)[-5:]]
            consistency = recent_themes.count(self.current_theme) / len(recent_themes)
            recent_strengths = [e['strength'] for e in list(self.theme_history)[-3:]]
            trend = np.polyfit(range(len(recent_strengths)), recent_strengths, 1)[0]
            momentum = (consistency + max(0.0, float(trend))) / 2.0
            return max(0.0, min(1.0, momentum))
        except Exception:
            return 0.5

    def _is_theme_regime_aligned(self, theme: int, regime: str) -> bool:
        alignments = {
            0: ['trending', 'breakout'],
            1: ['volatile', 'reversal'],
            2: ['volatile', 'noise'],
            3: ['trending', 'breakout']
        }
        return regime in alignments.get(int(theme), [])

    def _record_theme_signal(self):
        """Record theme signal for performance tracking"""
        try:
            signal_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'theme': self.current_theme,
                'strength': self.theme_strength,
                'market_context': self.market_context.copy()
            }
            self.theme_history.append(signal_record)
            if self.current_theme in self.theme_performance:
                d = self.theme_performance[self.current_theme]
                d['signals'] += 1
                count = d['signals']
                d['avg_strength'] = (d['avg_strength'] * (count - 1) + self.theme_strength) / count
        except Exception as e:
            self.logger.warning(f"Theme signal recording failed: {e}")

    async def process(self, **inputs) -> Dict[str, Any]:
        """Contract-compliant process for EnhancedThemeExpert.
        Produces: theme_voting_proposal, theme_confidence, theme_analysis,
                  agreement_score, consensus_direction, member_confidences,
                  raw_proposals, strategy_arbiter_weights, _thesis, and
                  EnhancedThemeExpert_voting_proposal/EnhancedThemeExpert_confidence mirrors.
        """
        start = time.time()
        name = self.__class__.__name__
        try:
            base = await super().process(**inputs)

            proposal = dict(base.get('voting_proposal') or {})
            confidence = float(base.get('confidence', 0.0))
            thesis_base = base.get('thesis', f"{name} thesis unavailable")

            # Compute momentum and alignment for agreement scoring
            momentum = self._calculate_theme_momentum()
            regime = self.market_context.get('regime', 'unknown')
            aligned = 1.0 if self._is_theme_regime_aligned(self.current_theme, regime) else 0.5
            strength = float(self.theme_strength or 0.0)
            # Agreement blends strength, momentum, and alignment
            agreement_score = max(0.0, min(1.0, 0.4 * strength + 0.4 * float(momentum) + 0.2 * aligned))

            # Theme analysis bundle
            theme_analysis = {
                'current_theme': int(self.current_theme),
                'theme_strength': strength,
                'theme_momentum': float(momentum),
                'regime': regime,
                'aligned_with_regime': bool(aligned > 0.9),
                'avg_strength': float(self.theme_performance.get(self.current_theme, {}).get('avg_strength', 0.0)),
            }

            # Normalized keys expected by downstream
            theme_voting_proposal = dict(proposal)
            theme_confidence = confidence
            consensus_direction = str(proposal.get('action', 'neutral'))

            # Single-expert publications in committee-like shape for compatibility
            member_confidences = {name: theme_confidence}
            raw_proposals = {name: dict(theme_voting_proposal)}
            strategy_arbiter_weights = {name: 1.0}

            # Build thesis
            thesis = (
                f"{name}: theme {theme_analysis['current_theme']} | "
                f"strength {theme_analysis['theme_strength']:.2f} | "
                f"momentum {theme_analysis['theme_momentum']:.2f} | "
                f"regime {regime} | aligned {theme_analysis['aligned_with_regime']} | "
                f"agreement {agreement_score:.2f}. {thesis_base}"
            )

            # Publish to SmartInfoBus explicit normalized keys for robustness
            try:
                self.smart_bus.set(f'{name}_voting_proposal', theme_voting_proposal, module=name, thesis=thesis, confidence=theme_confidence)
                self.smart_bus.set(f'{name}_confidence', theme_confidence, module=name, thesis=f"{name} confidence: {theme_confidence:.1%}")
                # Normalized mirrors for downstream consumers
                self.smart_bus.set('theme_voting_proposal', theme_voting_proposal, module=name, thesis=thesis, confidence=theme_confidence)
                self.smart_bus.set('theme_confidence', theme_confidence, module=name, thesis=f"Theme confidence {theme_confidence:.1%}")
                self.smart_bus.set('agreement_score', agreement_score, module=name, thesis=f"Agreement score {agreement_score:.2f}")
            except Exception:
                pass

            return {
                'theme_voting_proposal': theme_voting_proposal,
                'theme_confidence': theme_confidence,
                'theme_analysis': theme_analysis,
                'agreement_score': agreement_score,
                'consensus_direction': consensus_direction,
                'member_confidences': member_confidences,
                'raw_proposals': raw_proposals,
                'strategy_arbiter_weights': strategy_arbiter_weights,
                'EnhancedThemeExpert_voting_proposal': dict(theme_voting_proposal),
                'EnhancedThemeExpert_confidence': theme_confidence,
                # Provide local expert vote if enabled (for early consumers / contract uniformity)
                'expert_votes': ([{
                    'expert': name,
                    'vote': dict(theme_voting_proposal),
                    'confidence': float(theme_confidence),
                    'timestamp': datetime.datetime.now().isoformat()
                }] if bool(self.config.get('include_local_expert_vote', True)) else []),
                '_thesis': thesis,
            }

        except Exception as error:
            # Degraded contract-compliant payload on error
            error_msg = f"{type(error).__name__}: {str(error)[:160]}"
            self.logger.error(f"EnhancedThemeExpert error: {error_msg}")
            return {
                'theme_voting_proposal': {'action': 'abstain', 'reason': 'theme-expert-error'},
                'theme_confidence': 0.1,
                'theme_analysis': {
                    'current_theme': int(getattr(self, 'current_theme', 0) or 0),
                    'theme_strength': float(getattr(self, 'theme_strength', 0.0) or 0.0),
                    'theme_momentum': float(self._calculate_theme_momentum() if hasattr(self, '_calculate_theme_momentum') else 0.5),
                    'regime': self.market_context.get('regime', 'unknown'),
                    'aligned_with_regime': False,
                    'avg_strength': float(self.theme_performance.get(self.current_theme, {}).get('avg_strength', 0.0)) if hasattr(self, 'theme_performance') else 0.0,
                },
                'agreement_score': 0.0,
                'consensus_direction': 'neutral',
                'member_confidences': {self.__class__.__name__: 0.1},
                'raw_proposals': {self.__class__.__name__: {'action': 'abstain'}},
                'strategy_arbiter_weights': {self.__class__.__name__: 1.0},
                'EnhancedThemeExpert_voting_proposal': {'action': 'abstain'},
                'EnhancedThemeExpert_confidence': 0.1,
                'expert_votes': ([{
                    'expert': self.__class__.__name__,
                    'vote': {'action': 'abstain', 'reason': 'theme-expert-error'},
                    'confidence': 0.1,
                    'timestamp': datetime.datetime.now().isoformat()
                }] if bool(self.config.get('include_local_expert_vote', True)) else []),
                '_thesis': f"Operating in degraded mode due to {error_msg}",
            }


# =============================
# ENHANCED SEASONALITY RISK EXPERT
# =============================

@module(**module_args(
    "EnhancedSeasonalityRiskExpert",
    description="Enhanced seasonality-based risk expert with modern InfoBus integration",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class EnhancedSeasonalityRiskExpert(EnhancedVotingExpertBase):
    """
    PRODUCTION-GRADE Enhanced Seasonality Risk Expert v4.1
    """

    def _initialize(self):
        """Initialize enhanced seasonality expert"""
        super()._initialize()

        self.base_signal_strength = float(self.config.get('base_signal_strength', 0.3))
        self.seasonality_sensitivity = float(self.config.get('seasonality_sensitivity', 0.7))
        self.session_bias_strength = float(self.config.get('session_bias_strength', 0.8))

        self.current_seasonality_factor = 1.0
        self.seasonality_history = deque(maxlen=100)
        self.session_performance = {
            'american': {'signals': 0, 'success': 0, 'avg_factor': 1.0},
            'european': {'signals': 0, 'success': 0, 'avg_factor': 1.0},
            'asian': {'signals': 0, 'success': 0, 'avg_factor': 1.0},
            'rollover': {'signals': 0, 'success': 0, 'avg_factor': 1.0}
        }

        self.logger.info(format_operator_message(
            icon="[SEASON]",
            message="Enhanced Seasonality Risk Expert v4.1 initialized",
            base_signal_strength=self.base_signal_strength,
            seasonality_sensitivity=self.seasonality_sensitivity
        ))

        # Publish baseline provides to SmartInfoBus to prevent early BUS MISS
        try:
            name = self.__class__.__name__
            baseline_proposal = {
                'action': 'abstain',
                'signal_strength': 0.0,
                'position_size': 0.0,
                'duration': 'short',
                'seasonality_type': 'unknown'
            }
            self.smart_bus.set(f'{name}_voting_proposal', baseline_proposal, module=name, thesis='Baseline seasonality voting proposal')
            self.smart_bus.set(f'{name}_confidence', 0.1, module=name, thesis=f"{name} baseline confidence: 10%")
            self.smart_bus.set(f'{name}_market_context', self.market_context, module=name, thesis=f"Baseline market context for {name}")
            self.smart_bus.set(f'{name}_analytics', self.expert_analytics, module=name, thesis=f"Baseline performance analytics for {name}")
        except Exception:
            pass

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Contract-compliant process for EnhancedSeasonalityRiskExpert.
        Produces: seasonality_voting_proposal, seasonality_confidence, seasonality_analysis,
                  seasonality_risk_analysis, expert_performance, _thesis.
        Note: committee_decision/committee_confidence are owned by
              EnhancedVotingCommitteeCoordinator and are not produced here.
        """
        start = time.time()
        name = self.__class__.__name__
        try:
            base = await super().process(**inputs)

            proposal = dict(base.get('voting_proposal') or {})
            confidence = float(base.get('confidence', 0.0))
            thesis = base.get('thesis', 'Seasonality expert thesis unavailable')
            mc = dict(base.get('market_context') or {})
            analytics = dict(base.get('expert_analytics') or {})
            emergency = dict(base.get('emergency_status') or {})
            health = dict(base.get('health_metrics') or {})

            seasonality_analysis = {
                'market_context': mc,
                'expert_analytics': analytics,
                'emergency_status': emergency,
                'health_metrics': health,
                'seasonality_metadata': proposal.get('seasonality_metadata', {}),
                'session_adjustment': proposal.get('session_adjustment', {}),
            }

            # Seasonality risk analysis built from real factors (no placeholders)
            try:
                session = str(self.market_context.get('session', mc.get('session_type', 'unknown')))
                vol = str(self.market_context.get('volatility_level', 'medium')).lower()
                vol_score = 0.5 if vol not in ('low', 'quiet', 'high', 'elevated', 'very_high') else (0.2 if vol in ('low', 'quiet') else 0.8)
                deviation = abs(float(getattr(self, 'current_seasonality_factor', 1.0)) - 1.0)
                trend = float(self._calculate_seasonality_trend())
                emergency_active = bool(emergency.get('emergency_active', False))

                risk_score = 0.4 * min(1.0, deviation * 2.0) \
                           + 0.25 * min(1.0, abs(trend) * 2.0) \
                           + 0.2 * vol_score \
                           + 0.15 * (1.0 if emergency_active else 0.0)
                risk_score = max(0.0, min(1.0, float(risk_score)))
                risk_bias = 'high' if risk_score >= 0.7 else ('elevated' if risk_score >= 0.4 else 'normal')

                seasonality_risk_analysis = {
                    'session': session,
                    'volatility_level': vol,
                    'current_factor': float(getattr(self, 'current_seasonality_factor', 1.0)),
                    'trend': trend,
                    'risk_score': risk_score,
                    'risk_bias': risk_bias,
                    'emergency': emergency_active,
                    'seasonality_metadata': seasonality_analysis['seasonality_metadata'],
                    'session_adjustment': seasonality_analysis['session_adjustment'],
                }
            except Exception:
                seasonality_risk_analysis = {
                    'session': self.market_context.get('session', 'unknown'),
                    'volatility_level': self.market_context.get('volatility_level', 'medium'),
                    'current_factor': float(getattr(self, 'current_seasonality_factor', 1.0)),
                    'trend': 0.0,
                    'risk_score': 0.5,
                    'risk_bias': 'elevated',
                    'emergency': bool(emergency.get('emergency_active', False)),
                    'seasonality_metadata': seasonality_analysis['seasonality_metadata'],
                    'session_adjustment': seasonality_analysis['session_adjustment'],
                }

            # Local expert performance index (kept internal; committee owns expert_performance on bus)
            expert_performance = {self.__class__.__name__: self._get_local_expert_performance_index()}

            name = self.__class__.__name__
            include_local_vote = bool(self.config.get('include_local_expert_vote', True))
            expert_votes_list = []
            if include_local_vote:
                try:
                    expert_votes_list.append({
                        'expert': name,
                        'vote': proposal,
                        'confidence': confidence,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                except Exception:
                    pass
            out = {
                'seasonality_voting_proposal': proposal,
                'seasonality_confidence': confidence,
                'seasonality_risk_analysis': seasonality_risk_analysis,
                # Aliases for older readers
                'seasonal_voting_proposal': proposal,
                'seasonal_confidence': confidence,
                f'{name}_voting_proposal': proposal,
                f'{name}_confidence': confidence,
                'seasonality_analysis': seasonality_analysis,

                # Provide local expert vote (gated) to satisfy contract and enable early consumers
                'expert_votes': expert_votes_list,
                '_thesis': thesis,
            }
            self.performance_tracker.record_metric(self.__class__.__name__, 'process', (time.time() - start) * 1000, True)
            return out
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "seasonality_process")
            # Record failure metric
            try:
                self.performance_tracker.record_metric(self.__class__.__name__, 'process', (time.time() - start) * 1000, False, error=str(error_context))
            except Exception:
                pass

            safe_proposal = {
                'action': 'abstain',
                'signal_strength': 0.0,
                'position_size': 0.0,
                'duration': 'short',
                'seasonality_type': 'unknown',
                'reason': f'error: {error_context}'
            }
            return {
                'seasonality_voting_proposal': safe_proposal,
                'seasonality_confidence': 0.2,
                'seasonality_risk_analysis': {
                    'session': self.market_context.get('session', 'unknown'),
                    'volatility_level': self.market_context.get('volatility_level', 'medium'),
                    'current_factor': float(getattr(self, 'current_seasonality_factor', 1.0)),
                    'trend': 0.0,
                    'risk_score': 0.6,
                    'risk_bias': 'elevated',
                    'emergency': False,
                    'seasonality_metadata': {},
                    'session_adjustment': {},
                },
                'seasonal_voting_proposal': safe_proposal,
                'seasonal_confidence': 0.2,
                f'{name}_voting_proposal': safe_proposal,
                f'{name}_confidence': 0.2,
                'seasonality_analysis': {'error': str(error_context)},
                'expert_performance': {name: self._get_local_expert_performance_index()},
                'expert_votes': [{
                    'expert': name,
                    'vote': safe_proposal,
                    'confidence': 0.2,
                    'timestamp': datetime.datetime.now().isoformat()
                }] if bool(self.config.get('include_local_expert_vote', True)) else [],
                '_thesis': f"Processing error in {name}: {error_context}",
            }

    def _get_local_expert_performance_index(self) -> float:
        """Compute a bounded performance index from expert analytics (0..1)."""
        try:
            total = int(self.expert_analytics.get('total_actions', 0))
            success = int(self.expert_analytics.get('successful_actions', 0))
            success_rate = success / max(1, total)
            avg_conf = float(self.expert_analytics.get('avg_confidence', 0.5))
            # Weighted blend; decay slight to avoid overconfidence
            idx = 0.55 * success_rate + 0.45 * avg_conf
            return max(0.0, min(1.0, float(idx)))
        except Exception:
            return 0.5

    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate seasonality-based voting proposal"""
        try:
            risk_data = self.smart_bus.get('risk_data', self.__class__.__name__) or {}
            self.current_seasonality_factor = float(risk_data.get('seasonality_factor', 1.0) or 1.0)
            if not math.isfinite(self.current_seasonality_factor):
                self.current_seasonality_factor = 1.0

            current_session = str(market_data.get('session_type', 'unknown'))

            if self.current_seasonality_factor > 1.2:
                proposal = {
                    'action': 'seasonal_long_bias',
                    'signal_strength': self.base_signal_strength * self.current_seasonality_factor,
                    'position_size': min(self.max_signal_strength, self.base_signal_strength * self.current_seasonality_factor),
                    'duration': 'medium',
                    'seasonality_type': 'strong_positive'
                }
            elif self.current_seasonality_factor < 0.8:
                proposal = {
                    'action': 'seasonal_short_bias',
                    'signal_strength': self.base_signal_strength * (2.0 - self.current_seasonality_factor),
                    'position_size': min(self.max_signal_strength, self.base_signal_strength * abs(1.0 - self.current_seasonality_factor)),
                    'duration': 'medium',
                    'seasonality_type': 'strong_negative'
                }
            else:
                proposal = {
                    'action': 'seasonal_neutral',
                    'signal_strength': self.base_signal_strength * 0.5,
                    'position_size': self.base_signal_strength * 0.3,
                    'duration': 'short',
                    'seasonality_type': 'neutral'
                }

            proposal = self._apply_session_seasonality_adjustments(proposal, current_session)

            proposal['seasonality_metadata'] = {
                'seasonality_factor': self.current_seasonality_factor,
                'session': current_session,
                'session_performance': self.session_performance.get(current_session, {}),
                'seasonality_trend': self._calculate_seasonality_trend()
            }

            self._record_seasonality_signal(current_session)
            return proposal

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "seasonality_proposal_generation")
            return {'action': 'abstain', 'reason': f'Seasonality analysis failed: {error_context}', 'signal_strength': 0.0}

    async def _calculate_expert_specific_confidence(self, proposal: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate seasonality-specific confidence"""
        try:
            deviation = abs(self.current_seasonality_factor - 1.0)
            base = 0.5 + deviation * self.seasonality_sensitivity

            session = str(market_data.get('session_type', 'unknown'))
            perf = self.session_performance.get(session, {})
            if perf.get('signals', 0) > 5:
                sr = perf.get('success', 0) / perf['signals']
                base *= (0.7 + sr * 0.6)

            trend = self._calculate_seasonality_trend()
            if trend > 0.1: base *= 1.1
            elif trend < -0.1: base *= 0.9

            return max(0.2, min(0.9, base))
        except Exception:
            return 0.5

    def _apply_session_seasonality_adjustments(self, proposal: Dict[str, Any], session: str) -> Dict[str, Any]:
        """Apply session-specific seasonality adjustments"""
        try:
            session_multipliers = {
                'american': 1.0, 'european': 0.9, 'asian': 0.8, 'rollover': 0.3, 'unknown': 0.7
            }
            mult = session_multipliers.get(session, 0.7)
            perf = self.session_performance.get(session, {})
            if perf.get('signals', 0) > 10:
                avg = perf.get('avg_factor', 1.0)
                if avg > 1.1: mult *= 1.1
                elif avg < 0.9: mult *= 0.9

            if isinstance(proposal.get('signal_strength', None), (int, float)):
                proposal['signal_strength'] *= mult
            if isinstance(proposal.get('position_size', None), (int, float)):
                proposal['position_size'] *= mult

            proposal['session_adjustment'] = {'session': session, 'multiplier': mult, 'session_performance': perf}
            return proposal
        except Exception:
            return proposal

    def _calculate_seasonality_trend(self) -> float:
        """Calculate seasonality trend from recent history"""
        try:
            if len(self.seasonality_history) < 5:
                return 0.0
            recent = [e['factor'] for e in list(self.seasonality_history)[-10:]]
            trend = np.polyfit(range(len(recent)), recent, 1)[0]
            return max(-0.5, min(0.5, float(trend)))
        except Exception:
            return 0.0

    def _record_seasonality_signal(self, session: str):
        """Record seasonality signal for performance tracking"""
        try:
            rec = {
                'timestamp': datetime.datetime.now().isoformat(),
                'factor': self.current_seasonality_factor,
                'session': session,
                'market_context': self.market_context.copy()
            }
            self.seasonality_history.append(rec)
            if session in self.session_performance:
                d = self.session_performance[session]
                d['signals'] += 1
                count = d['signals']
                d['avg_factor'] = (d['avg_factor'] * (count - 1) + self.current_seasonality_factor) / count
        except Exception as e:
            self.logger.warning(f"Seasonality signal recording failed: {e}")


# =============================
# FACTORY FUNCTION FOR CREATING ALL ENHANCED EXPERTS
# =============================

def create_enhanced_voting_experts(config: Dict[str, Any]) -> List[EnhancedVotingExpertBase]:
    """
    Create all enhanced voting experts with modern InfoBus integration.
    """
    experts: List[EnhancedVotingExpertBase] = []
    try:
        expert_classes: List[Type[BaseModule]] = [
            EnhancedThemeExpert,
            EnhancedSeasonalityRiskExpert,
            # Add more expert classes as they are implemented
        ]
        for cls in expert_classes:
            try:
                expert_config = config.get(cls.__name__, {})
                instance = cast(EnhancedVotingExpertBase, cls(config=expert_config))
                experts.append(instance)
                print(f"[OK] Created {cls.__name__}")
            except Exception as e:
                print(f"[FAIL] Failed to create {cls.__name__}: {e}")
        print(f"[OK] Successfully created {len(experts)} enhanced voting experts")
        return experts
    except Exception as e:
        print(f"[FAIL] Enhanced voting expert creation failed: {e}")
        return []


# =============================
# ENHANCED VOTING COMMITTEE COORDINATOR (HARDENED)
# =============================

_EVCC_INSTANCE = None  # process-lifetime singleton for coordinator reuse

@module(**module_args(
    "EnhancedVotingCommitteeCoordinator",
    description="Enhanced voting committee coordinator with modern InfoBus integration",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
    priority=-100,  # Negative priority = runs after voters (priority=0)
))
class EnhancedVotingCommitteeCoordinator(BaseModule, SmartInfoBusVotingMixin, SmartInfoBusStateMixin):
    """PRODUCTION-GRADE Enhanced Voting Committee Coordinator v4.1 (hardened)"""

    def __new__(cls, *args, **kwargs):
        global _EVCC_INSTANCE
        if _EVCC_INSTANCE is None:
            # Avoid referencing class name to prevent NameError during early registration
            _EVCC_INSTANCE = super().__new__(cls)
        return _EVCC_INSTANCE

    def _initialize(self):
        """Initialize enhanced voting committee coordinator"""
        # Prevent repeated heavy initialization if re-instantiated
        if getattr(self, "_singleton_init_done", False):
            return
        self._initialize_voting_state()
        self._initialize_state_management()

        # Committee configuration
        self.consensus_threshold = float(self.config.get('consensus_threshold', 0.6))
        self.minimum_voters = int(self.config.get('minimum_voters', 2))
        self.performance_weighting = bool(self.config.get('performance_weighting', True))
        self.emergency_override = bool(self.config.get('emergency_override', True))

        # Ingestion & discovery knobs (hardened)
        self.expert_votes_bus_key = str(self.config.get('expert_votes_bus_key', 'expert_votes'))
        # Prefer registry-only by default to avoid dependency on expert_votes feed
        self.discovery_mode = str(self.config.get('discovery_mode', 'registry_only'))  # feed_only | registry_only | feed_then_registry
        self.voter_flag_name = str(self.config.get('voter_flag_name', 'is_voting_member'))
        self.voters_from_config = list(self.config.get('voters', []))  # optional static list of module names
        self.ingest_minimum = int(self.config.get('ingest_minimum', self.minimum_voters))
        self.ignore_actions = set(self.config.get('ignore_actions', ['abstain', None]))
        self.enable_fallback_discovery = bool(self.config.get('enable_fallback_discovery', True))
        self.max_votes_per_tick = int(self.config.get('max_votes_per_tick', 128))

        # Committee state
        self.active_experts: List[str] = []
        self.expert_weights: Dict[str, float] = {}
        self.voting_history = deque(maxlen=100)
        self.consensus_history = deque(maxlen=50)

        # Committee analytics
        self.committee_analytics = {
            'total_decisions': 0,
            'consensus_decisions': 0,
            'emergency_overrides': 0,
            'average_confidence': 0.5,
            'expert_performance': defaultdict(float)
        }

        # Initialize modern systems
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="EnhancedVotingCommitteeCoordinator",
            log_path="logs/voting/committee_coordinator.log",
            max_lines=5000,
            operator_mode=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.performance_tracker = PerformanceTracker()
        self._decision_counter = 0  # KERNEL v1: track per-tick monotonic id

        self.logger.info(format_operator_message(
            icon="[COMMITTEE]",
            message="Enhanced Voting Committee Coordinator v4.1 initialized",
            consensus_threshold=f"{self.consensus_threshold:.1%}",
            minimum_voters=self.minimum_voters
        ))

        # Publish initial empty snapshots to prevent early BUS MISS and declare ownership
        try:
            try:
                # Ensure canonical ownership; blocks other modules from writing this key
                self.smart_bus.declare_owner('expert_votes', self.__class__.__name__)
            except Exception:
                pass

            # Initialize expert_votes snapshot for consumers
            try:
                self.smart_bus.set('expert_votes', [], module=self.__class__.__name__, thesis='Baseline expert votes initialized')
            except Exception:
                pass
            # Baseline committee votes collections
            self.smart_bus.set(
                'committee_votes',
                [],
                module=self.__class__.__name__,
                thesis='Baseline committee votes initialized'
            )
            self.smart_bus.set(
                'votes',
                [],
                module=self.__class__.__name__,
                thesis='Baseline raw votes initialized'
            )
        except Exception:
            pass
        # Mark initialization complete to avoid re-inits
        self._singleton_init_done = True

    # ---------------- HARDENED HELPERS ----------------

    def _normalize_vote_entry(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Coerce a raw vote into a canonical structure or return None if invalid."""
        try:
            if not isinstance(raw, dict):
                return None
            expert = str(raw.get('expert') or raw.get('name') or 'unknown').strip()
            if not expert:
                return None
            vote = raw.get('vote') or raw.get('proposal') or {}
            if not isinstance(vote, dict):
                vote = {}
            action = vote.get('action', 'abstain')
            confidence = float(raw.get('confidence', 0.0) or 0.0)
            ts = raw.get('timestamp') or datetime.datetime.now().isoformat()
            norm = {
                'expert': expert,
                'vote': dict(vote),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': ts
            }
            return norm
        except Exception:
            return None

    # Helper to map action strings to a sign (+1 long, -1 short, 0 neutral)
    def _action_sign(self, action: str) -> float:
        if not action:
            return 0.0
        a = action.strip().lower()
        # broad mapping: treat names with "long" as +, "short" as -
        if a.startswith('long') or 'long_' in a:
            return 1.0
        if a.startswith('short') or 'short_' in a:
            return -1.0
        # some domain actions:
        if a in ('trend_following', 'seasonal_long_bias', 'long_risk_assets', 'safe_haven_rotation'):
            return 1.0
        if a in ('seasonal_short_bias',):
            return -1.0
        return 0.0

    # ---------------- MAIN PROCESSING ----------------

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process committee voting with enhanced coordination (contract-compliant)."""
        start_time = time.time()
        try:
            # 1) Collect inputs
            expert_votes = await self._collect_expert_votes()
            expert_weights = await self._calculate_expert_weights(expert_votes)

            # 2) Committee decisions & metrics
            decision = await self._determine_committee_decision(expert_votes, expert_weights)
            consensus = await self._analyze_voting_consensus(expert_votes, expert_weights)
            committee_confidence = await self._calculate_committee_confidence(expert_votes, expert_weights, consensus)
            thesis = await self._generate_committee_thesis(decision, committee_confidence, consensus, expert_votes)

            # ---- KERNEL v1: canonical analytics surfaces --------------------------------
            committee_members = [v.get('expert', 'unknown') for v in expert_votes]
            n_members = len(committee_members)

            member_confidences_map = {v['expert']: float(v.get('confidence', 0.0) or 0.0) for v in expert_votes}
            member_confidences_ordered = [float(member_confidences_map.get(name, 0.0)) for name in committee_members]

            def _vectorize(ev: Dict[str, Any]) -> List[float]:
                vote = dict(ev.get('vote') or {})
                action = str(vote.get('action', 'abstain')).lower()
                ss = float(vote.get('signal_strength', ev.get('confidence', 0.0)) or 0.0)
                conf = float(ev.get('confidence', 0.0) or 0.0)
                return [self._action_sign(action) * ss, conf]

            proposal_vectors: List[List[float]] = [_vectorize(v) for v in expert_votes]

            # decision id (monotonic per instance)
            self._decision_counter += 1
            decision_id = f"{datetime.datetime.now().isoformat()}#{self._decision_counter}"

            # 3) Build declared outputs
            voting_summary = {
                'action': decision.get('action', 'abstain'),
                'decision_type': decision.get('decision_type', 'unknown'),
                'consensus_strength': float(consensus.get('consensus_strength', 0.0)),
                'consensus_exists': bool(consensus.get('consensus_exists', False)),
                'vote_count': int(consensus.get('vote_count', len(expert_votes))),
            }
            strategy_arbiter_weights = dict(expert_weights)
            consensus_direction = decision.get('action', 'neutral')
            agreement_score = float(consensus.get('consensus_strength', 0.0))

            raw_proposals = {v['expert']: dict(v['vote'] or {}) for v in expert_votes}
            member_confidences = {v['expert']: float(v.get('confidence', 0.0)) for v in expert_votes}

            votes = list(expert_votes)
            member_proposals = dict(raw_proposals)
            voting_weights = dict(expert_weights)

            # minutes since midnight (numeric; better for downstream consumers like the Aligner)
            now = datetime.datetime.now()
            time_of_day = now.hour * 60 + now.minute + now.second / 60.0

            performance_feedback = {
                'average_confidence': float(self.committee_analytics.get('average_confidence', 0.5)),
                'per_expert_confidence': member_confidences
            }
            horizon_alignment = await self._compute_horizon_alignment_hint(expert_votes, expert_weights)

            # lightweight signals snapshot
            signals = {
                'committee_action': decision.get('action', 'abstain'),
                'committee_confidence': committee_confidence
            }

            # 4) Update SmartInfoBus (side-effect) -- include downstream-friendly keys
            await self._update_smartinfobus_committee({
                'committee_decision': decision,
                # IMPORTANT: publish structured 'committee_consensus' (not 'voting_consensus' on bus)
                'committee_consensus': consensus,
                # keep legacy object in return payload only (not bus) as 'voting_consensus'
                'committee_confidence': committee_confidence,
                'expert_votes': expert_votes,
                'expert_weights': expert_weights,
                'committee_analytics': self.committee_analytics.copy(),
                # downstream:
                'voting_weights': voting_weights,
                'time_of_day': time_of_day,
                'performance_feedback': performance_feedback,
                # canonical surfaces
                'committee_members': committee_members,
                'n_members': n_members,
                'member_confidences_ordered': member_confidences_ordered,
                'member_confidences_map': member_confidences_map,
                'proposal_vectors': proposal_vectors,
                'decision_id': decision_id,
                'horizon_alignment': horizon_alignment,
                'signals': signals,
            }, thesis)

            # 5) Record and return
            self._record_committee_decision({
                'committee_decision': decision,
                'committee_confidence': committee_confidence,
                'voting_consensus': consensus,  # compat for internal analytics
                'expert_votes': expert_votes,
            })

            out = {
                'committee_decision': decision,
                'committee_consensus': consensus,
                'voting_consensus': consensus,  # TEMPORARY COMPAT for callers (not bus)
                'committee_confidence': committee_confidence,
                'expert_votes': expert_votes,
                'expert_weights': expert_weights,
                'committee_analytics': self.committee_analytics.copy(),
                'voting_summary': voting_summary,
                'strategy_arbiter_weights': strategy_arbiter_weights,
                'consensus_direction': consensus_direction,
                'agreement_score': agreement_score,
                'raw_proposals': raw_proposals,
                'member_confidences': member_confidences,
                'member_confidences_ordered': member_confidences_ordered,
                'member_confidences_map': member_confidences_map,
                'committee_members': committee_members,
                'n_members': n_members,
                'proposal_vectors': proposal_vectors,
                'votes': votes,
                'committee_votes': [
                    {
                        'action': v.get('vote', {}).get('action', 'abstain'),
                        'confidence': float(v.get('confidence', 0.0)),
                        'expert': v.get('expert', 'unknown'),
                        'timestamp': v.get('timestamp')
                    } for v in expert_votes
                ],
                'member_proposals': member_proposals,
                'voting_weights': voting_weights,
                'time_of_day': time_of_day,
                'performance_feedback': performance_feedback,
                'horizon_alignment': horizon_alignment,
                'decision_id': decision_id,
                'signals': signals,
                # Canonical trade_vote publication for downstream consumers
                'trade_vote': {
                    'action': decision.get('action', 'abstain'),
                    'confidence': float(committee_confidence),
                    'reason': decision.get('reason', ''),
                    'timestamp': datetime.datetime.now().isoformat()
                },
                # KERNEL v1: expanded vote bundle
                'trade_vote_v2': {
                    'action': decision.get('action', 'abstain'),
                    'size': float(committee_confidence),  # will be throttled by fragility/collusion later
                    'horizon_minutes': int(horizon_alignment.get('intended_horizon_minutes', 0))
                        if isinstance(horizon_alignment, dict) else 0,
                    'confidence': float(committee_confidence),
                    'reason': decision.get('reason', ''),
                    'decision_id': decision_id,
                    'timestamp': datetime.datetime.now().isoformat(),
                },
                '_thesis': thesis,
            }
            self.performance_tracker.record_metric(self.__class__.__name__, 'process', (time.time() - start_time) * 1000, True)
            return out

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "committee_coordination")
            fail = {
                'committee_decision': {'action': 'abstain', 'reason': f'Committee error: {error_context}'},
                'committee_consensus': {'consensus_exists': False, 'error': str(error_context)},
                'voting_consensus': {'consensus_exists': False, 'error': str(error_context)},  # compat
                'committee_confidence': 0.1,
                'expert_votes': [],
                'expert_weights': {},
                'committee_analytics': {'error': str(error_context)},
                'voting_summary': {'action': 'abstain', 'decision_type': 'error', 'consensus_strength': 0.0, 'consensus_exists': False, 'vote_count': 0},
                'strategy_arbiter_weights': {},
                'consensus_direction': 'neutral',
                'agreement_score': 0.0,
                'raw_proposals': {},
                'member_confidences_ordered': [],
                'member_confidences_map': {},
                'committee_members': [],
                'n_members': 0,
                'proposal_vectors': [],
                'votes': [],
                'member_proposals': {},
                'voting_weights': {},
                'time_of_day': datetime.datetime.now().hour * 60,
                'performance_feedback': {'error': str(error_context)},
                'horizon_alignment': {'status': 'unknown'},
                'decision_id': None,
                'signals': {},
                'trade_vote': {
                    'action': 'abstain', 'confidence': 0.1, 'reason': 'committee-error',
                    'timestamp': datetime.datetime.now().isoformat()
                },
                'trade_vote_v2': {
                    'action': 'abstain', 'size': 0.0, 'horizon_minutes': 0, 'confidence': 0.1,
                    'reason': 'committee-error', 'decision_id': None, 'timestamp': datetime.datetime.now().isoformat()
                },
                '_thesis': f"Committee coordination failed: {error_context}",
            }
            return fail

    # ---------- helpers (add inside EnhancedVotingCommitteeCoordinator) ----------
    def _to_bool(self, v: Any, default: bool = False) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        s = str(v).strip().lower()
        return s in ("true", "1", "yes", "y", "on")

    def _voter_key_pairs(self, name: str) -> List[Tuple[str, str]]:
        """
        Return (proposal_key, confidence_key) pairs to try for a given voter.
        First pair is canonical <Name>_voting_proposal / <Name>_confidence.
        Some voters also publish alt keys (Theme/Seasonality wrappers).
        """
        pairs: List[Tuple[str, str]] = [(f"{name}_voting_proposal", f"{name}_confidence")]
        if name == "EnhancedThemeExpert":
            pairs += [
                ("EnhancedThemeExpert_voting_proposal", "EnhancedThemeExpert_confidence"),
                ("theme_voting_proposal", "theme_confidence"),
            ]
        if name == "EnhancedSeasonalityRiskExpert":
            pairs += [
                ("EnhancedSeasonalityRiskExpert_voting_proposal", "EnhancedSeasonalityRiskExpert_confidence"),
                ("seasonality_voting_proposal", "seasonality_confidence"),
            ]
        return pairs

    # ---------- patched discovery ----------
    def _discover_voters(self) -> List[str]:
        """
        Discover voter module names in prioritized order:
        1) config list (self.voters_from_config)
        2) CONTRACTS where meta[is_voting_member] == True (robust string/bool)
        3) (optional) minimal fallback if still empty
        """
        discovered: List[str] = []

        # 1) include any explicit config voters first (preserve order)
        for n in self.voters_from_config:
            if isinstance(n, str) and n and n not in discovered:
                discovered.append(n)

        # 2) registry voters with is_voting_member == True
        try:
            from modules.contracts import CONTRACTS  # type: ignore
            for name, mc in CONTRACTS.items():
                try:
                    meta = getattr(mc, "meta", {}) or {}
                    if self._to_bool(meta.get(self.voter_flag_name, False), False):
                        if name not in discovered:
                            discovered.append(name)
                except Exception:
                    continue
        except Exception:
            pass

        # 3) last-resort safety (avoid empty set)
        if not discovered:
            for fallback in ["EnhancedThemeExpert", "EnhancedSeasonalityRiskExpert"]:
                if fallback not in discovered:
                    discovered.append(fallback)

        # de-dup while preserving order
        seen = set()
        out: List[str] = []
        for n in discovered:
            if n not in seen:
                out.append(n)
                seen.add(n)

        self.logger.info(format_operator_message(
            icon="[DISCOVERY]",
            message="Voter discovery",
            voters=",".join(out),
            count=len(out),
            mode=self.discovery_mode
        ))
        return out

    # ---------- patched vote collection ----------
    async def _collect_expert_votes(self) -> List[Dict[str, Any]]:
        """Collect votes from active voters only, with feed-first ingestion and clean fallback."""
        try:
            voters = self._discover_voters()
            voters_set = set(voters)

            expert_votes: List[Dict[str, Any]] = []

            # --- 1) FEED-FIRST: normalized 'expert_votes' list, filtered to voters only
            if self.discovery_mode in ("feed_only", "feed_then_registry"):
                try:
                    feed = self.smart_bus.get(self.expert_votes_bus_key, self.__class__.__name__) or []
                    if isinstance(feed, list):
                        for raw in feed[-self.max_votes_per_tick:]:
                            norm = self._normalize_vote_entry(raw)
                            if norm and norm.get("expert") in voters_set:
                                expert_votes.append(norm)
                except Exception as e:
                    self.logger.warning(f"Failed to read {self.expert_votes_bus_key}: {e}")

            # de-dup by expert (keep latest)
            by_expert: Dict[str, Dict[str, Any]] = {}
            for v in expert_votes:
                by_expert[v["expert"]] = v
            expert_votes = list(by_expert.values())

            # --- 2) FALLBACK: pull per-voter bus keys ONLY for discovered voters
            need_more = (
                len(expert_votes) < max(self.ingest_minimum, 1)
                and self.enable_fallback_discovery
                and self.discovery_mode in ("registry_only", "feed_then_registry")
            )
            if need_more:
                for name in voters:
                    if name in by_expert:
                        continue
                    try:
                        # try canonical + alt key pairs; only query confidence if proposal exists
                        for prop_key, conf_key in self._voter_key_pairs(name):
                            proposal = self.smart_bus.get(prop_key, self.__class__.__name__, default=None)
                            if proposal is None:
                                continue  # avoid an extra MISS on confidence
                            confidence = self.smart_bus.get(conf_key, self.__class__.__name__, default=None)
                            if confidence is None:
                                continue
                            raw = {
                                "expert": name,
                                "vote": dict(proposal) if isinstance(proposal, dict) else {},
                                "confidence": float(confidence),
                                "timestamp": datetime.datetime.now().isoformat(),
                            }
                            norm = self._normalize_vote_entry(raw)
                            if norm:
                                by_expert[name] = norm
                            break  # stop after first successful pair
                    except Exception as e:
                        self.logger.warning(f"Failed to collect vote from {name}: {e}")

                expert_votes = list(by_expert.values())

            # --- 3) If any non-abstain present, drop abstains (keeps committee decisive)
            if any((v.get("vote", {}).get("action") not in self.ignore_actions) for v in expert_votes):
                expert_votes = [v for v in expert_votes if v.get("vote", {}).get("action") not in self.ignore_actions]

            # --- 4) cap
            if len(expert_votes) > self.max_votes_per_tick:
                expert_votes = expert_votes[-self.max_votes_per_tick:]

            self.logger.info(format_operator_message(
                icon="[VOTES]",
                message="Expert votes collected",
                vote_count=len(expert_votes),
                voters=len(voters),
                mode=self.discovery_mode,
                feed_key=self.expert_votes_bus_key
            ))
            return expert_votes

        except Exception as e:
            self.logger.error(f"Vote collection failed: {e}")
            return []

    async def _calculate_expert_weights(self, expert_votes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate dynamic expert weights; uses bus-level expert_performance where available."""
        try:
            weights: Dict[str, float] = {}
            if not expert_votes:
                return weights

            # try bus-level performance (authoritative), else fall back to local analytics map
            bus_perf = {}
            try:
                bus_perf = self.smart_bus.get('expert_performance', self.__class__.__name__) or {}
                if not isinstance(bus_perf, dict):
                    bus_perf = {}
            except Exception:
                pass

            # If we have BOTH ignored actions and non-ignored ones, drop the ignored ones.
            actions = [v.get('vote', {}).get('action') for v in expert_votes]
            if any(a not in self.ignore_actions for a in actions):
                expert_votes = [v for v in expert_votes if v.get('vote', {}).get('action') not in self.ignore_actions]

            # build weights
            total = 0.0
            for v in expert_votes:
                name = v['expert']
                conf = float(v.get('confidence', 0.0))
                base = max(0.0, conf)

                if self.performance_weighting:
                    # prefer bus-tracked performance; fallback to committee-local analytics
                    perf = bus_perf.get(name, None)
                    if perf is None:
                        perf = float(self.committee_analytics['expert_performance'].get(name, 0.5))
                    else:
                        try:
                            perf = float(perf)
                        except Exception:
                            perf = 0.5
                    base *= (0.5 + max(0.0, min(1.0, perf)))  # 0.5..1.5x

                regime = self.smart_bus.get('market_regime', self.__class__.__name__) or 'unknown'
                base *= self._get_expert_regime_adjustment(name, regime)

                # floor to avoid zeroing an active voter; cap to 2.0 pre-normalization
                w = max(1e-6, min(2.0, base))
                weights[name] = w
                total += w

            if total <= 0.0:
                # equal weights
                n = len(expert_votes)
                return {v['expert']: 1.0 / n for v in expert_votes}

            # normalize
            for k in list(weights.keys()):
                weights[k] = weights[k] / total
            return weights

        except Exception as e:
            self.logger.error(f"Expert weight calculation failed: {e}")
            return {}

    def _get_expert_regime_adjustment(self, expert_name: str, regime: str) -> float:
        """Get expert performance adjustment based on market regime"""
        regime_adjustments = {
            'EnhancedThemeExpert': {'trending': 1.2, 'volatile': 0.9, 'ranging': 1.0, 'unknown': 0.8},
            'EnhancedSeasonalityRiskExpert': {'trending': 1.0, 'volatile': 1.1, 'ranging': 1.2, 'unknown': 0.9}
        }
        return regime_adjustments.get(expert_name, {}).get(regime, 1.0)

    async def _determine_committee_decision(self, expert_votes: List[Dict[str, Any]],
                                            expert_weights: Dict[str, float]) -> Dict[str, Any]:
        """Determine final committee decision using weighted voting with abstain-aware logic."""
        try:
            if not expert_votes:
                return {'action': 'abstain', 'reason': 'no_expert_votes'}

            weighted_actions = defaultdict(float)
            total_weight = 0.0
            for v in expert_votes:
                name = v['expert']
                w = float(expert_weights.get(name, 1.0))
                action = v.get('vote', {}).get('action', 'abstain')
                weighted_actions[action] += w
                total_weight += w

            if not weighted_actions or total_weight <= 0.0:
                return {'action': 'abstain', 'reason': 'no_valid_actions'}

            # if there are non-abstain votes, ignore abstain in the max
            has_non_abstain = any(a not in self.ignore_actions for a in weighted_actions.keys())
            if has_non_abstain:
                filtered = {a: w for a, w in weighted_actions.items() if a not in self.ignore_actions}
                best_action, best_weight = max(filtered.items(), key=lambda x: x[1])
                denom = sum(filtered.values()) or total_weight
                consensus_strength = best_weight / denom if denom > 0 else 0.0
            else:
                best_action, best_weight = max(weighted_actions.items(), key=lambda x: x[1])
                consensus_strength = best_weight / total_weight if total_weight > 0 else 0.0

            return {
                'action': best_action,
                'consensus_strength': consensus_strength,
                'total_weight': total_weight,
                'action_weights': dict(weighted_actions),
                'decision_type': 'consensus' if consensus_strength >= self.consensus_threshold else 'plurality'
            }

        except Exception as e:
            self.logger.error(f"Committee decision determination failed: {e}")
            return {'action': 'abstain', 'reason': f'decision_error: {str(e)}'}

    async def _calculate_committee_confidence(self, expert_votes: List[Dict[str, Any]],
                                              expert_weights: Dict[str, float],
                                              consensus: Optional[Dict[str, Any]] = None) -> float:
        """Calculate overall committee confidence (weight-adjusted x consensus strength)"""
        try:
            if not expert_votes or not expert_weights:
                return 0.1

            weighted_conf = 0.0
            total_w = 0.0
            for v in expert_votes:
                name = v['expert']
                conf = float(v.get('confidence', 0.0))
                w = float(expert_weights.get(name, 1.0))
                weighted_conf += conf * w
                total_w += w

            avg_conf = (weighted_conf / total_w) if total_w > 0 else 0.0

            if consensus is None:
                # compute a quick consensus proxy
                action_w = defaultdict(float)
                for v in expert_votes:
                    action_w[v.get('vote', {}).get('action', 'abstain')] += expert_weights.get(v['expert'], 1.0)
                total = sum(action_w.values())
                dom = max(action_w.values()) if action_w else 0.0
                consensus_strength = (dom / total) if total > 0 else 0.0
            else:
                consensus_strength = float(consensus.get('consensus_strength', 0.0))

            final = avg_conf * (0.7 + 0.6 * consensus_strength)  # scale 0.7..1.3x
            return max(0.1, min(1.0, final))
        except Exception:
            return 0.3

    async def _analyze_voting_consensus(self, expert_votes: List[Dict[str, Any]],
                                        expert_weights: Dict[str, float]) -> Dict[str, Any]:
        """Analyze voting consensus and identify conflicts"""
        try:
            if not expert_votes:
                return {'consensus_exists': False, 'reason': 'no_votes'}

            action_weights = defaultdict(float)
            for v in expert_votes:
                action = v.get('vote', {}).get('action', 'abstain')
                w = float(expert_weights.get(v['expert'], 1.0))
                action_weights[action] += w

            total_weight = sum(action_weights.values())
            if total_weight <= 0:
                return {'consensus_exists': False, 'reason': 'zero_weight'}

            dominant_action, dom_w = max(action_weights.items(), key=lambda x: x[1])
            consensus_strength = dom_w / total_weight

            conflict_level = self._assess_voting_conflict_level(action_weights, total_weight)

            return {
                'consensus_exists': consensus_strength >= self.consensus_threshold,
                'consensus_strength': consensus_strength,
                'dominant_action': dominant_action,
                'action_distribution': dict(action_weights),
                'conflict_level': conflict_level,
                'vote_count': len(expert_votes),
                'total_weight': total_weight
            }
        except Exception as e:
            return {'consensus_exists': False, 'error': str(e)}

    def _assess_voting_conflict_level(self, action_weights: Dict[str, float], total_weight: float) -> str:
        """Assess level of voting conflict via entropy of the action distribution"""
        try:
            if len(action_weights) <= 1:
                return 'NONE'
            probs = [w / total_weight for w in action_weights.values() if total_weight > 0]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(len(action_weights))
            ne = (entropy / max_entropy) if max_entropy > 0 else 0.0
            if ne < 0.3: return 'LOW'
            if ne < 0.6: return 'MEDIUM'
            if ne < 0.8: return 'HIGH'
            return 'SEVERE'
        except Exception:
            return 'UNKNOWN'

    async def _generate_committee_thesis(self, decision: Dict[str, Any], confidence: float,
                                         consensus: Dict[str, Any], expert_votes: List[Dict[str, Any]]) -> str:
        """Generate comprehensive committee decision thesis"""
        try:
            parts = []
            action = decision.get('action', 'unknown')
            cs = float(consensus.get('consensus_strength', 0.0))
            label = "HIGH" if confidence > 0.7 else "MODERATE" if confidence > 0.4 else "LOW"
            parts.append(f"COMMITTEE DECISION: {action.upper()} with {label} confidence ({confidence:.1%})")

            if consensus.get('consensus_exists', False):
                parts.append(f"STRONG CONSENSUS: {cs:.1%} agreement among {len(expert_votes)} experts")
            else:
                parts.append(f"DIVIDED OPINION: {consensus.get('conflict_level', 'UNKNOWN')} conflict, plurality decision")

            parts.append(f"EXPERT PARTICIPATION: {len(expert_votes)} voting experts active")
            parts.append(f"DECISION TYPE: {decision.get('decision_type', 'unknown').upper()} via weighted voting")
            return " | ".join(parts)
        except Exception as e:
            return f"Committee thesis generation failed: {e}"

    async def _compute_horizon_alignment_hint(
        self,
        expert_votes: List[Dict[str, Any]],
        expert_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Lightweight, local hint (the full THA module owns alignment in depth).
        """
        try:
            intended = 30 if any((float(v.get("confidence", 0.0)) >= 0.6) for v in expert_votes) else 10
            return {
                "status": "hint",
                "intended_horizon_minutes": intended,
                "reason": "local-heuristic; THA provides final alignment"
            }
        except Exception:
            return {"status": "hint", "intended_horizon_minutes": 0, "reason": "fallback"}

    async def _update_smartinfobus_committee(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with committee results (no bus write to 'voting_consensus')."""
        try:
            # Structured decision
            self.smart_bus.set(
                'committee_decision',
                results['committee_decision'],
                module=self.__class__.__name__,
                thesis=thesis,
                confidence=results['committee_confidence']
            )

            # Structured, committee-owned consensus (NOT 'voting_consensus')
            try:
                self.smart_bus.set(
                    'committee_consensus',
                    results.get('committee_consensus', {}),
                    module=self.__class__.__name__,
                    thesis=(
                        "Committee consensus (structured) | strength: "
                        f"{float(results.get('committee_consensus', {}).get('consensus_strength', 0.0)):.1%}"
                    ),
                )
            except Exception:
                pass

            # Committee confidence
            self.smart_bus.set(
                'committee_confidence',
                results['committee_confidence'],
                module=self.__class__.__name__,
                thesis=f"Committee confidence: {results['committee_confidence']:.1%}",
            )

            # Canonical expert_votes snapshot (state key; coordinator-owned)
            try:
                self.smart_bus.set(
                    'expert_votes',
                    list(results.get('expert_votes', [])),
                    module=self.__class__.__name__,
                    thesis='Normalized expert votes snapshot'
                )
            except Exception:
                pass

            # Publish committee_votes (simplified list)
            try:
                simplified_votes = [
                    {
                        'action': v.get('vote', {}).get('action', 'abstain'),
                        'confidence': float(v.get('confidence', 0.0)),
                        'expert': v.get('expert', 'unknown'),
                        'timestamp': v.get('timestamp')
                    }
                    for v in results.get('expert_votes', [])
                ]
                self.smart_bus.set(
                    'committee_votes',
                    simplified_votes,
                    module=self.__class__.__name__,
                    thesis='Per-expert committee votes snapshot'
                )
                # Optional mirror to flat 'votes' for legacy consumers
                self.smart_bus.set('votes', simplified_votes, module=self.__class__.__name__, thesis='Legacy flat votes mirror')
            except Exception as e:
                self.logger.warning(f"Committee votes publish soft-fail: {e}")

            # Canonical analytics surfaces & hints
            for k in [
                "committee_members",
                "n_members",
                "member_confidences_map",
                "member_confidences_ordered",
                "proposal_vectors",
                "signals",
                "horizon_alignment",
                "voting_weights",
                "time_of_day",
                "performance_feedback",
                "member_confidences",
                # decision_id/tick_ts are owned by VotingKernel; avoid overriding
            ]:
                if k in results:
                    try:
                        self.smart_bus.set(k, results[k], module=self.__class__.__name__, thesis=f"committee.{k}")
                    except Exception:
                        pass

            # Intentionally avoid publishing trade_vote/trade_vote_v2 here to prevent provider churn

            # Avoid publishing decision bundle; VotingKernel is the canonical bundle publisher

        except Exception as e:
            self.logger.error(f"SmartInfoBus committee update failed: {e}")

    def _record_committee_decision(self, results: Dict[str, Any]):
        """Record committee decision for analytics"""
        try:
            rec = {
                'timestamp': datetime.datetime.now().isoformat(),
                'decision': results['committee_decision'],
                'confidence': results['committee_confidence'],
                'consensus': results.get('voting_consensus') or results.get('committee_consensus', {}),
                'expert_count': len(results.get('expert_votes', []))
            }
            self.voting_history.append(rec)
            self.committee_analytics['total_decisions'] += 1

            consensus_obj = results.get('voting_consensus') or results.get('committee_consensus', {})
            if consensus_obj.get('consensus_exists', False):
                self.committee_analytics['consensus_decisions'] += 1

            n = self.committee_analytics['total_decisions']
            old = float(self.committee_analytics.get('average_confidence', 0.5))
            newc = float(results['committee_confidence'])
            self.committee_analytics['average_confidence'] = (old * (n - 1) + newc) / n

        except Exception as e:
            self.logger.warning(f"Committee decision recording failed: {e}")

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """SmartInfoBusVotingMixin implementation"""
        try:
            results = await self.process()
            return results.get('committee_decision', {'action': 'abstain', 'reason': 'processing_failed'})
        except Exception as e:
            return {'action': 'abstain', 'reason': f'committee_error: {str(e)}'}

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """SmartInfoBusVotingMixin implementation"""
        try:
            results = await self.process()
            return float(results.get('committee_confidence', 0.3))
        except Exception:
            return 0.2

    def get_health_status(self) -> Dict[str, Any]:
        """
        Custom health status for voting committee coordinator.
        Returns detailed health information about committee operations.
        """
        try:
            # Get base health from BaseModule
            base_health = super().get_health_status()

            # Calculate committee-specific metrics
            total_decisions = self.committee_analytics.get('total_decisions', 0)
            consensus_decisions = self.committee_analytics.get('consensus_decisions', 0)
            avg_confidence = self.committee_analytics.get('average_confidence', 0.5)
            consensus_rate = consensus_decisions / max(total_decisions, 1)

            # Determine health status
            status = base_health.get('status', 'OK')
            is_healthy = base_health.get('is_healthy', True)
            issues = []

            # Check if any experts are registered
            expert_count = len(self.expert_registry)
            if expert_count == 0:
                status = 'DEGRADED'
                is_healthy = False
                issues.append("No experts registered in committee")
            elif expert_count < 3:
                if status == 'OK':
                    status = 'WARNING'
                issues.append(f"Low expert count: {expert_count} experts")

            # Check consensus rate
            if total_decisions > 10 and consensus_rate < 0.3:
                if status == 'OK':
                    status = 'WARNING'
                issues.append(f"Low consensus rate: {consensus_rate:.1%}")

            # Check average confidence
            if total_decisions > 5 and avg_confidence < 0.3:
                status = 'DEGRADED'
                is_healthy = False
                issues.append(f"Low average confidence: {avg_confidence:.1%}")
            elif total_decisions > 5 and avg_confidence < 0.5:
                if status == 'OK':
                    status = 'WARNING'
                issues.append(f"Suboptimal average confidence: {avg_confidence:.1%}")

            return {
                **base_health,
                'status': status,
                'is_healthy': is_healthy,
                'expert_count': expert_count,
                'total_decisions': total_decisions,
                'consensus_decisions': consensus_decisions,
                'consensus_rate': consensus_rate,
                'average_confidence': avg_confidence,
                'voting_history_size': len(self.voting_history),
                'last_member_check': getattr(self, '_last_member_check', 0),
                'issues': issues if issues else None,
                'performance': {
                    **base_health.get('performance', {}),
                    'experts_registered': expert_count,
                    'decisions_made': total_decisions,
                    'consensus_achieved': consensus_decisions
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'module': self.__class__.__name__,
                'version': self.metadata.version,
                'is_healthy': False,
                'error': str(e),
                'last_error': str(e)
            }
