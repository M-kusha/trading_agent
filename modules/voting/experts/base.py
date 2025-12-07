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
from modules.voting.core.constants import (
    VotingBusKeys,
    CONFIDENCE_THRESHOLD_F,
    MIN_SIGNAL_STRENGTH_F,
    HIGH_CONFIDENCE_THRESHOLD_F,
    PRIMARY_TIMEFRAME,
)


class VotingExpertBase(VotingModuleBase):
    """
    Base class for all voting experts.
    
    Provides:
    - Standard proposal/confidence publication to SmartInfoBus
    - Mode-aware gating of weak/noisy signals
    - Action history tracking
    - Market context management
    - Intelligence parameters (adjustable signals)
    - Expert analytics (success tracking)
    
    Subclasses must implement:
    - _expert_specific_init(): Initialize expert-specific state
    - _generate_expert_specific_proposal(): Generate the actual proposal
    - _calculate_expert_specific_confidence(): Calculate expert confidence
    """
    
    # ────────────────────────────────────────────────────────────────
    # Template methods for subclasses
    # ────────────────────────────────────────────────────────────────
    
    @abstractmethod
    async def _generate_expert_specific_proposal(
        self, 
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
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
    
    # ────────────────────────────────────────────────────────────────
    # Initialization
    # ────────────────────────────────────────────────────────────────
    
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
            'session': 'unknown',
        }
        
        # Intelligence parameters (adjustable signal generation)
        self.intelligence_parameters: Dict[str, float] = {
            'signal_threshold': float(self.config.get('signal_threshold', 0.3)),
            'confidence_decay': float(self.config.get('confidence_decay', 0.95)),
            'history_weight': float(self.config.get('history_weight', 0.3)),
            'regime_sensitivity': float(self.config.get('regime_sensitivity', 0.8)),
        }
        
        # Expert analytics (success tracking)
        self.expert_analytics: Dict[str, Any] = {
            'total_actions': 0,
            'successful_actions': 0,
            'avg_confidence': 0.5,
            'last_action': None,
            'last_action_time': None,
        }
        
        # Signal strength limits (expert-local)
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
    
    # ────────────────────────────────────────────────────────────────
    # Circuit breaker
    # ────────────────────────────────────────────────────────────────
    
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
    
    # ────────────────────────────────────────────────────────────────
    # Market context
    # ────────────────────────────────────────────────────────────────
    
    def _update_market_context(self, market_data: Dict[str, Any]) -> None:
        """Update market context from incoming market data."""
        if not isinstance(market_data, dict):
            return
        
        # Read from SmartInfoBus for authoritative values
        self.market_context['regime'] = (
            self.smart_bus.get('market_regime', self.__class__.__name__, default='unknown')
            or market_data.get('market_regime', 'unknown')
        )
        self.market_context['volatility_level'] = (
            market_data.get('volatility_level')
            or market_data.get('volatility', 'medium')
        )
        self.market_context['trend_strength'] = float(
            market_data.get('trend_strength', 0.0) or 0.0
        )
        
        # Session from bus or data
        session = (
            self.smart_bus.get('session_canonical', self.__class__.__name__)
            or market_data.get('current_session')
            or market_data.get('session_type', 'unknown')
        )
        self.market_context['session'] = str(session).lower()
    
    # ────────────────────────────────────────────────────────────────
    # Analytics / history
    # ────────────────────────────────────────────────────────────────
    
    def _record_action(
        self, 
        action: str, 
        confidence: float, 
        proposal: Dict[str, Any]
    ) -> None:
        """Record an action for history tracking."""
        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'action': action,
            'confidence': confidence,
            'regime': self.market_context.get('regime', 'unknown'),
            'session': self.market_context.get('session', 'unknown'),
            'proposal_summary': {
                'action': proposal.get('action'),
                'signal_strength': proposal.get('signal_strength', 0.0),
            },
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
        proposal_key = VotingBusKeys.expert_proposal(name)
        confidence_key = VotingBusKeys.expert_confidence(name)
        
        try:
            baseline_proposal = {
                'action': 'abstain',
                'signal_strength': 0.0,
                'position_size': 0.0,
                'duration': 'short',
                'reason': 'baseline',
            }
            self.smart_bus.set(
                proposal_key,
                baseline_proposal,
                module=name,
                thesis=f'Baseline proposal for {name}',
            )
            self.smart_bus.set(
                confidence_key,
                0.1,
                module=name,
                thesis=f'{name} baseline confidence: 10%',
            )
            self.smart_bus.set(
                f'{name}_market_context',
                self.market_context,
                module=name,
                thesis=f'Baseline market context for {name}',
            )
            self.smart_bus.set(
                f'{name}_analytics',
                self.expert_analytics,
                module=name,
                thesis=f'Baseline analytics for {name}',
            )
        except Exception as e:
            self.logger.debug(f"Baseline key publication skipped: {e}")
    
    # ────────────────────────────────────────────────────────────────
    # Market data canonicalization
    # ────────────────────────────────────────────────────────────────
    
    def _build_market_data(self, raw_market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a canonical market_data view from SmartInfoBus so all experts
        see the same OHLCV snapshot per step.
        """
        name = self.__class__.__name__
        market_data: Dict[str, Any] = {}
        
        if isinstance(raw_market_data, dict):
            market_data.update(raw_market_data)
        
        try:
            historical = self.smart_bus.get('historical_prices', name, default=None)
        except Exception:
            historical = None
        
        try:
            price_data = self.smart_bus.get('price_data', name, default=None)
        except Exception:
            price_data = None
        
        try:
            prices = self.smart_bus.get('prices', name, default=None)
        except Exception:
            prices = None
        
        primary_symbol = self.config.get('primary_symbol')
        if not isinstance(primary_symbol, str) or not primary_symbol:
            if isinstance(historical, dict):
                for candidate in ('XAU_USD', 'EUR_USD'):
                    if candidate in historical:
                        primary_symbol = candidate
                        break
                if not primary_symbol and historical:
                    primary_symbol = next(iter(historical.keys()))
            elif isinstance(price_data, dict) and price_data:
                for candidate in ('XAU_USD', 'EUR_USD'):
                    if candidate in price_data:
                        primary_symbol = candidate
                        break
                if not primary_symbol:
                    primary_symbol = next(iter(price_data.keys()))
        
        # M15 is the primary trading timeframe; H1/H4/D1 are context only.
        primary_tf = str(self.config.get('primary_timeframe', PRIMARY_TIMEFRAME) or PRIMARY_TIMEFRAME)
        
        # OHLCV snapshot
        if 'ohlcv' not in market_data:
            ohlcv: Dict[str, Any] = {}
            if isinstance(historical, dict) and isinstance(primary_symbol, str) and primary_symbol in historical:
                sym_block = historical.get(primary_symbol)
                if isinstance(sym_block, dict):
                    rec = sym_block.get(primary_tf)
                    if not isinstance(rec, dict):
                        # Fallback order: M15 (primary), then context TFs
                        for tf in ('M15', 'H1', 'H4', 'D1'):
                            candidate = sym_block.get(tf)
                            if isinstance(candidate, dict):
                                rec = candidate
                                break
                    if not isinstance(rec, dict) and sym_block:
                        first_key = next(iter(sym_block.keys()))
                        rec = sym_block.get(first_key)
                    if isinstance(rec, dict):
                        for key in ('open', 'high', 'low', 'close', 'volume'):
                            seq = rec.get(key)
                            if isinstance(seq, (list, tuple)):
                                ohlcv[key] = list(seq)
            if ohlcv:
                market_data['ohlcv'] = ohlcv
                if 'close' in ohlcv and 'close_prices' not in market_data:
                    market_data['close_prices'] = ohlcv['close']
        
        # Prices array
        if 'prices' not in market_data:
            if isinstance(prices, dict) and prices:
                value = None
                if isinstance(primary_symbol, str) and primary_symbol in prices:
                    value = prices.get(primary_symbol)
                else:
                    value = next(iter(prices.values()))
                try:
                    market_data['prices'] = [float(value)] if value is not None else []
                except Exception:
                    pass
            elif isinstance(market_data.get('ohlcv'), dict):
                close_seq = market_data['ohlcv'].get('close')
                if isinstance(close_seq, list):
                    market_data['prices'] = close_seq
        
        # Current price
        if 'current_price' not in market_data:
            value = None
            if isinstance(price_data, dict) and price_data:
                sym_block = None
                if isinstance(primary_symbol, str) and primary_symbol in price_data:
                    sym_block = price_data.get(primary_symbol)
                else:
                    sym_block = next(iter(price_data.values()))
                if isinstance(sym_block, dict):
                    value = sym_block.get('last') or sym_block.get('close')
            if value is None:
                prices_list = market_data.get('prices')
                if isinstance(prices_list, list) and prices_list:
                    value = prices_list[-1]
            if value is not None:
                try:
                    market_data['current_price'] = float(value)
                except Exception:
                    pass
        
        # Volume / high / low last values
        if 'volume' not in market_data and isinstance(market_data.get('ohlcv'), dict):
            vol_seq = market_data['ohlcv'].get('volume')
            if isinstance(vol_seq, list) and vol_seq:
                try:
                    market_data['volume'] = float(vol_seq[-1])
                except Exception:
                    pass
        
        if 'high' not in market_data and isinstance(market_data.get('ohlcv'), dict):
            high_seq = market_data['ohlcv'].get('high')
            if isinstance(high_seq, list) and high_seq:
                try:
                    market_data['high'] = float(high_seq[-1])
                except Exception:
                    pass
        
        if 'low' not in market_data and isinstance(market_data.get('ohlcv'), dict):
            low_seq = market_data['ohlcv'].get('low')
            if isinstance(low_seq, list) and low_seq:
                try:
                    market_data['low'] = float(low_seq[-1])
                except Exception:
                    pass
        
        return market_data
    
    # ────────────────────────────────────────────────────────────────
    # Main process
    # ────────────────────────────────────────────────────────────────
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Main processing method for voting experts.
        
        1. Check circuit breaker
        2. Get market data
        3. Update market context
        4. Generate proposal (subclass method)
        5. Calculate confidence (subclass method)
        6. Apply mode-aware gating / normalization
        7. Publish to SmartInfoBus
        8. Record action
        9. Return contract-compliant output
        """
        start = time.time()
        name = self.__class__.__name__
        proposal_key = VotingBusKeys.expert_proposal(name)
        confidence_key = VotingBusKeys.expert_confidence(name)
        
        try:
            # 1. Circuit breaker
            if self._check_circuit_breaker():
                return self._degraded_output("circuit_breaker_open")
            
            # 2. Market data
            raw_market_data = (
                inputs.get('market_data')
                or self.smart_bus.get('market_data', name)
                or {}
            )
            market_data = self._build_market_data(raw_market_data)
            
            # 3. Context
            self._update_market_context(market_data)
            
            # 4. Expert-specific proposal
            proposal = await self._generate_expert_specific_proposal(market_data)
            if not isinstance(proposal, dict):
                proposal = {'action': 'abstain', 'reason': 'invalid_proposal_type'}
            
            # 5. Expert-specific confidence
            confidence = await self._calculate_expert_specific_confidence(
                proposal, market_data
            )
            confidence = max(0.0, min(1.0, float(confidence)))
            
            # 6. Mode-aware gating / normalization
            proposal, confidence = self._postprocess_proposal_for_voting(
                proposal, confidence
            )
            
            # 7. Publish to SmartInfoBus
            thesis = self._generate_thesis(proposal, confidence)
            self.smart_bus.set(
                proposal_key,
                proposal,
                module=name,
                thesis=thesis,
                confidence=confidence,
            )
            self.smart_bus.set(
                confidence_key,
                confidence,
                module=name,
                thesis=f'{name} confidence: {confidence:.1%}',
            )
            
            # 8. Record action
            self._record_action(proposal.get('action', 'unknown'), confidence, proposal)
            self._record_success()
            
            # 9. Performance metrics
            elapsed_ms = (time.time() - start) * 1000
            try:
                self.performance_tracker.record_metric(
                    name, 'process', elapsed_ms, True
                )
            except Exception:
                pass
            
            # 10. Build output payload with per-instrument votes for contract compliance
            per_instrument_key = f"{name}_per_instrument_votes"
            per_instrument_votes = proposal.get('proposals', proposal.get('per_instrument', {}))
            
            return {
                'voting_proposal': proposal,
                'confidence': confidence,
                'thesis': thesis,
                'market_context': self.market_context.copy(),
                'expert_analytics': self.expert_analytics.copy(),
                'emergency_status': {'emergency_active': False},
                'health_metrics': {'processing_time_ms': elapsed_ms},
                proposal_key: proposal,
                confidence_key: confidence,
                per_instrument_key: per_instrument_votes,  # Per-instrument votes for contract
                '_thesis': thesis,
            }
        
        except Exception as e:
            self._record_error(e)
            name = self.__class__.__name__
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(e, f"{name}_process")
                msg = str(error_context)
            else:
                msg = str(e)
            self.logger.error(f"[{name}] Process error: {msg}")
            return self._degraded_output(msg)
    
    # ────────────────────────────────────────────────────────────────
    # Proposal post-processing / gating
    # ────────────────────────────────────────────────────────────────
    
    def _postprocess_proposal_for_voting(
        self, 
        proposal: Dict[str, Any], 
        confidence: float,
    ) -> tuple[Dict[str, Any], float]:
        """
        Normalize and gate the proposal using mode-aware thresholds.
        
        - Ensures signal_strength is in [0, 1]
        - Ensures position_size respects max_signal_strength
        - Soft-kills weak directional signals (turns into 'flat')
        - Keeps per-instrument payloads intact (we only touch top-level fields)
        """
        # Extract and normalize action
        action_raw = str(proposal.get('action', 'abstain')).lower().strip()
        if not action_raw:
            action_raw = 'abstain'
        
        # Extract signal strength (experts may use 'magnitude')
        sig = proposal.get('signal_strength', proposal.get('magnitude', 0.0))
        try:
            sig_f = float(sig or 0.0)
        except Exception:
            sig_f = 0.0
        sig_f = max(0.0, min(1.0, sig_f))
        
        proposal['signal_strength'] = sig_f
        
        # Ensure position_size exists and is bounded
        if 'position_size' in proposal:
            try:
                ps = float(proposal['position_size'])
            except Exception:
                ps = 0.0
        else:
            ps = sig_f
        ps = max(0.0, min(self.max_signal_strength, ps))
        proposal['position_size'] = ps
        
        # Mode-aware thresholds
        min_strength = MIN_SIGNAL_STRENGTH_F()
        conf_floor = CONFIDENCE_THRESHOLD_F()
        high_conf = HIGH_CONFIDENCE_THRESHOLD_F()
        
        confidence = max(0.0, min(1.0, confidence))
        
        # Classify action
        is_long = (action_raw == 'long')
        is_short = (action_raw == 'short')
        is_directional = is_long or is_short
        is_flat_like = action_raw in ('flat', 'hold')
        is_abstain_like = action_raw in ('abstain', 'none', 'skip')
        
        # Directional signals: apply hard gating
        if is_directional:
            if sig_f < min_strength or confidence < conf_floor:
                # Demote to flat; keep raw in metadata for debugging
                proposal.setdefault('raw_action', action_raw)
                proposal.setdefault('raw_signal_strength', sig_f)
                proposal.setdefault('raw_confidence', confidence)
                
                proposal['action'] = 'flat'
                proposal['signal_strength'] = min(sig_f, min_strength * 0.5)
                proposal['position_size'] = min(
                    proposal['signal_strength'], self.max_signal_strength
                )
                
                # Confidence becomes "we are fairly sure this is neutral"
                confidence = max(0.15, min(confidence, conf_floor * 0.9))
            else:
                # Good directional signal – keep but clip to sane range
                proposal['action'] = action_raw
                proposal['signal_strength'] = max(sig_f, min_strength)
                proposal['position_size'] = min(
                    proposal['signal_strength'], self.max_signal_strength
                )
                confidence = max(conf_floor, min(confidence, high_conf))
        
        # Flat-like or abstain: keep weak, low-strength
        elif is_flat_like:
            proposal['action'] = 'flat'
            proposal['signal_strength'] = min(sig_f, min_strength * 0.5)
            proposal['position_size'] = min(
                proposal['signal_strength'], self.max_signal_strength
            )
            confidence = min(confidence, conf_floor * 0.9)
        
        elif is_abstain_like:
            proposal['action'] = 'abstain'
            proposal['signal_strength'] = 0.0
            proposal['position_size'] = 0.0
            confidence = min(confidence, conf_floor * 0.8)
        
        else:
            # Unknown action string – treat as abstain but keep raw for debugging
            proposal.setdefault('raw_action', action_raw)
            proposal['action'] = 'abstain'
            proposal['signal_strength'] = 0.0
            proposal['position_size'] = 0.0
            confidence = max(0.05, min(confidence, conf_floor * 0.7))
        
        return proposal, confidence
    
    # ────────────────────────────────────────────────────────────────
    # Misc helpers
    # ────────────────────────────────────────────────────────────────
    
    def _generate_thesis(self, proposal: Dict[str, Any], confidence: float) -> str:
        """Generate a thesis string explaining the proposal."""
        name = self.__class__.__name__
        action = proposal.get('action', 'unknown')
        signal = float(proposal.get('signal_strength', 0.0) or 0.0)
        regime = self.market_context.get('regime', 'unknown')
        
        return (
            f"{name}: action={action} | signal={signal:.2f} | "
            f"confidence={confidence:.1%} | regime={regime}"
        )
    
    def _degraded_output(self, reason: str) -> Dict[str, Any]:
        """Return a contract-compliant degraded output."""
        name = self.__class__.__name__
        proposal_key = VotingBusKeys.expert_proposal(name)
        confidence_key = VotingBusKeys.expert_confidence(name)
        per_instrument_key = f"{name}_per_instrument_votes"
        
        proposal = {
            'action': 'abstain',
            'reason': reason,
            'signal_strength': 0.0,
            'position_size': 0.0,
        }
        
        return {
            'voting_proposal': proposal,
            'confidence': 0.1,
            'thesis': f'{name} operating in degraded mode: {reason}',
            'market_context': self.market_context.copy(),
            'expert_analytics': self.expert_analytics.copy(),
            'emergency_status': {'emergency_active': True, 'reason': reason},
            'health_metrics': {'degraded': True},
            proposal_key: proposal,
            confidence_key: 0.1,
            per_instrument_key: {},  # Empty per-instrument votes for contract compliance
            '_thesis': f'{name} degraded: {reason}',
        }
    
    def create_proposal(
        self,
        action: str,
        signal_strength: float,
        confidence: float,
        reason: str = "",
        **extra_fields,
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
