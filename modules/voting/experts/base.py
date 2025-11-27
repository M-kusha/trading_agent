"""
Voting Expert Base Class
========================
Specialized base class for all voting experts.
Extends VotingModuleBase with expert-specific functionality.

This eliminates ~200 lines of duplicate code from each expert.
"""

from __future__ import annotations

import datetime
import time
from abc import abstractmethod
from collections import deque
from typing import Any, Dict, Optional

from modules.voting.core.base import VotingModuleBase
from modules.voting.core.types import VotingProposal


class VotingExpertBase(VotingModuleBase):
    """
    Base class for all voting experts.
    
    Provides:
    - Standard proposal/confidence publication to SmartInfoBus
    - Action history tracking
    - Market context management
    - Intelligence parameters (adjustable signals)
    - Expert analytics (success tracking)
    
    Subclasses must implement:
    - _module_specific_init(): Initialize expert-specific state
    - _generate_expert_specific_proposal(): Generate the actual proposal
    - _calculate_expert_specific_confidence(): Calculate expert confidence
    """
    
    # Template method pattern: child classes override these
    @abstractmethod
    async def _generate_expert_specific_proposal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate expert-specific voting proposal. Must be implemented by subclass."""
        raise NotImplementedError
    
    @abstractmethod
    async def _calculate_expert_specific_confidence(
        self, 
        proposal: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate expert-specific confidence. Must be implemented by subclass."""
        raise NotImplementedError
    
    def _module_specific_init(self) -> None:
        """
        Initialize expert-specific state.
        Called by VotingModuleBase._initialize() after common setup.
        """
        # Action history for tracking recent decisions
        self.action_history: deque = deque(maxlen=100)
        
        # Market context (updated each tick)
        self.market_context: Dict[str, Any] = {
            'regime': 'unknown',
            'volatility_level': 'medium',
            'trend_strength': 0.0,
            'session': 'unknown'
        }
        
        # Intelligence parameters (adjustable signal generation)
        self.intelligence_parameters: Dict[str, float] = {
            'signal_threshold': float(self.config.get('signal_threshold', 0.3)),
            'confidence_decay': float(self.config.get('confidence_decay', 0.95)),
            'history_weight': float(self.config.get('history_weight', 0.3)),
            'regime_sensitivity': float(self.config.get('regime_sensitivity', 0.8))
        }
        
        # Expert analytics (success tracking)
        self.expert_analytics: Dict[str, Any] = {
            'total_actions': 0,
            'successful_actions': 0,
            'avg_confidence': 0.5,
            'last_action': None,
            'last_action_time': None
        }
        
        # Signal strength limits
        self.max_signal_strength = float(self.config.get('max_signal_strength', 1.0))
        self.min_signal_strength = float(self.config.get('min_signal_strength', 0.1))
        
        # Circuit breaker state
        self._consecutive_errors = 0
        self._circuit_open = False
        self._circuit_open_until: Optional[datetime.datetime] = None
        self._max_consecutive_errors = int(self.config.get('max_consecutive_errors', 5))
        self._circuit_reset_seconds = float(self.config.get('circuit_reset_seconds', 60.0))
        
        # Call subclass-specific initialization
        self._expert_specific_init()
    
    def _expert_specific_init(self) -> None:
        """
        Override in subclasses for expert-specific initialization.
        Called after common expert setup.
        """
        pass
    
    def _check_circuit_breaker(self) -> bool:
        """
        Check if circuit breaker is tripped.
        Returns True if we should skip processing (circuit open).
        """
        if not self._circuit_open:
            return False
        
        now = datetime.datetime.now()
        if self._circuit_open_until and now >= self._circuit_open_until:
            # Reset circuit breaker
            self._circuit_open = False
            self._consecutive_errors = 0
            self._circuit_open_until = None
            self.logger.info(f"[{self.__class__.__name__}] Circuit breaker reset")
            return False
        
        return True
    
    def _record_error(self, error: Exception) -> None:
        """Record an error and potentially trip circuit breaker."""
        self._consecutive_errors += 1
        if self._consecutive_errors >= self._max_consecutive_errors:
            self._circuit_open = True
            self._circuit_open_until = datetime.datetime.now() + datetime.timedelta(
                seconds=self._circuit_reset_seconds
            )
            self.logger.warning(
                f"[{self.__class__.__name__}] Circuit breaker OPEN until "
                f"{self._circuit_open_until.isoformat()}"
            )
    
    def _record_success(self) -> None:
        """Record a successful operation, reset error count."""
        self._consecutive_errors = 0
    
    def _update_market_context(self, market_data: Dict[str, Any]) -> None:
        """Update market context from incoming market data."""
        if not isinstance(market_data, dict):
            return
        
        # Read from SmartInfoBus for authoritative values
        self.market_context['regime'] = (
            self.smart_bus.get('market_regime', self.__class__.__name__, default='unknown') or
            market_data.get('market_regime', 'unknown')
        )
        self.market_context['volatility_level'] = (
            market_data.get('volatility_level') or
            market_data.get('volatility', 'medium')
        )
        self.market_context['trend_strength'] = float(
            market_data.get('trend_strength', 0.0) or 0.0
        )
        
        # Session from bus or data
        session = (
            self.smart_bus.get('session_canonical', self.__class__.__name__) or
            market_data.get('current_session') or
            market_data.get('session_type', 'unknown')
        )
        self.market_context['session'] = str(session).lower()
    
    def _record_action(self, action: str, confidence: float, proposal: Dict[str, Any]) -> None:
        """Record an action for history tracking."""
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'action': action,
            'confidence': confidence,
            'regime': self.market_context.get('regime', 'unknown'),
            'session': self.market_context.get('session', 'unknown'),
            'proposal_summary': {
                'action': proposal.get('action'),
                'signal_strength': proposal.get('signal_strength', 0.0)
            }
        }
        self.action_history.append(record)
        
        # Update analytics
        self.expert_analytics['total_actions'] += 1
        self.expert_analytics['last_action'] = action
        self.expert_analytics['last_action_time'] = record['timestamp']
        
        # Running average of confidence
        n = self.expert_analytics['total_actions']
        old_avg = self.expert_analytics['avg_confidence']
        self.expert_analytics['avg_confidence'] = (old_avg * (n - 1) + confidence) / n
    
    def _publish_baseline_keys(self) -> None:
        """
        Publish baseline values to SmartInfoBus to prevent early BUS MISS.
        Called during initialization.
        """
        name = self.__class__.__name__
        try:
            baseline_proposal = {
                'action': 'abstain',
                'signal_strength': 0.0,
                'position_size': 0.0,
                'duration': 'short',
                'reason': 'baseline'
            }
            self.smart_bus.set(
                f'{name}_voting_proposal', 
                baseline_proposal, 
                module=name, 
                thesis=f'Baseline proposal for {name}'
            )
            self.smart_bus.set(
                f'{name}_confidence', 
                0.1, 
                module=name, 
                thesis=f'{name} baseline confidence: 10%'
            )
            self.smart_bus.set(
                f'{name}_market_context', 
                self.market_context, 
                module=name, 
                thesis=f'Baseline market context for {name}'
            )
            self.smart_bus.set(
                f'{name}_analytics', 
                self.expert_analytics, 
                module=name, 
                thesis=f'Baseline analytics for {name}'
            )
        except Exception as e:
            self.logger.debug(f"Baseline key publication skipped: {e}")
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Main processing method for voting experts.
        
        1. Check circuit breaker
        2. Get market data
        3. Update market context
        4. Generate proposal (subclass method)
        5. Calculate confidence (subclass method)
        6. Publish to SmartInfoBus
        7. Record action
        8. Return contract-compliant output
        """
        start = time.time()
        name = self.__class__.__name__
        
        try:
            # 1. Circuit breaker check
            if self._check_circuit_breaker():
                return self._degraded_output("circuit_breaker_open")
            
            # 2. Get market data
            market_data = (
                inputs.get('market_data') or
                self.smart_bus.get('market_data', name) or
                {}
            )
            
            # 3. Update context
            self._update_market_context(market_data)
            
            # 4. Generate proposal (subclass)
            proposal = await self._generate_expert_specific_proposal(market_data)
            if not isinstance(proposal, dict):
                proposal = {'action': 'abstain', 'reason': 'invalid_proposal_type'}
            
            # 5. Calculate confidence (subclass)
            confidence = await self._calculate_expert_specific_confidence(proposal, market_data)
            confidence = max(0.0, min(1.0, float(confidence)))
            
            # 6. Publish to SmartInfoBus
            thesis = self._generate_thesis(proposal, confidence)
            self.smart_bus.set(
                f'{name}_voting_proposal', 
                proposal, 
                module=name, 
                thesis=thesis,
                confidence=confidence
            )
            self.smart_bus.set(
                f'{name}_confidence', 
                confidence, 
                module=name, 
                thesis=f'{name} confidence: {confidence:.1%}'
            )
            
            # 7. Record action
            self._record_action(proposal.get('action', 'unknown'), confidence, proposal)
            self._record_success()
            
            # 8. Build output
            elapsed_ms = (time.time() - start) * 1000
            self.performance_tracker.record_metric(name, 'process', elapsed_ms, True)
            
            return {
                'voting_proposal': proposal,
                'confidence': confidence,
                'thesis': thesis,
                'market_context': self.market_context.copy(),
                'expert_analytics': self.expert_analytics.copy(),
                'emergency_status': {'emergency_active': False},
                'health_metrics': {'processing_time_ms': elapsed_ms},
                f'{name}_voting_proposal': proposal,
                f'{name}_confidence': confidence,
                '_thesis': thesis
            }
        
        except Exception as e:
            self._record_error(e)
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(e, f"{name}_process")
                msg = str(error_context)
            else:
                msg = str(e)
            self.logger.error(f"[{name}] Process error: {msg}")
            return self._degraded_output(msg)
    
    def _generate_thesis(self, proposal: Dict[str, Any], confidence: float) -> str:
        """Generate a thesis string explaining the proposal."""
        name = self.__class__.__name__
        action = proposal.get('action', 'unknown')
        signal = proposal.get('signal_strength', 0.0)
        regime = self.market_context.get('regime', 'unknown')
        
        return (
            f"{name}: action={action} | signal={signal:.2f} | "
            f"confidence={confidence:.1%} | regime={regime}"
        )
    
    def _degraded_output(self, reason: str) -> Dict[str, Any]:
        """Return a contract-compliant degraded output."""
        name = self.__class__.__name__
        proposal = {'action': 'abstain', 'reason': reason, 'signal_strength': 0.0}
        
        return {
            'voting_proposal': proposal,
            'confidence': 0.1,
            'thesis': f'{name} operating in degraded mode: {reason}',
            'market_context': self.market_context.copy(),
            'expert_analytics': self.expert_analytics.copy(),
            'emergency_status': {'emergency_active': True, 'reason': reason},
            'health_metrics': {'degraded': True},
            f'{name}_voting_proposal': proposal,
            f'{name}_confidence': 0.1,
            '_thesis': f'{name} degraded: {reason}'
        }
    
    def create_proposal(
        self,
        action: str,
        signal_strength: float,
        confidence: float,
        reason: str = "",
        **extra_fields
    ) -> VotingProposal:
        """
        Create a standardized VotingProposal.
        Helper method for subclasses.
        """
        return VotingProposal(
            action=action,
            confidence=confidence,
            signal_strength=signal_strength,
            reason=reason,
            expert=self.__class__.__name__,
            timestamp=datetime.datetime.now().isoformat(),
            metadata=extra_fields,
        )
