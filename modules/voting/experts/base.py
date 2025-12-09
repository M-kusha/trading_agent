"""
Voting Expert Base Class
========================
Specialized base class for all voting experts.
Extends VotingModuleBase with expert-specific functionality.

This eliminates ~200 lines of duplicate code from each expert.

Position Focus Mode:
- When positions are open, experts switch to "position management" mode
- Instead of looking for new trades, they evaluate: should we hold, scale, or exit?
- Signals are reframed as "supports position" or "threatens position"
"""

from __future__ import annotations

import datetime
import time
from abc import abstractmethod
from collections import deque
from typing import Any, Dict, Optional, Tuple, cast

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
        
        # ════════════════════════════════════════════════════════════════
        # INDICATOR CACHE: Skip expensive recalculations when data unchanged
        # ════════════════════════════════════════════════════════════════
        self._indicator_cache: Dict[str, Dict[str, Any]] = {}  # {instrument: {hash, results, timestamp}}
        self._cache_ttl_seconds = float(self.config.get('indicator_cache_ttl', 5.0))  # Cache valid for 5s
        
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
    # Indicator Cache (Performance Optimization)
    # ────────────────────────────────────────────────────────────────
    
    def _compute_price_hash(self, prices: list) -> str:
        """Compute a fast hash of price data for cache invalidation."""
        if not prices:
            return ""
        # Use last price + length + sum of last 5 prices for fast, unique-enough hash
        last_5 = prices[-5:] if len(prices) >= 5 else prices
        return f"{len(prices)}:{prices[-1]:.5f}:{sum(last_5):.5f}"
    
    def _get_cached_indicators(self, instrument: str, prices: list) -> Optional[Dict[str, Any]]:
        """
        Get cached indicator results if data hasn't changed.
        
        Returns None if cache miss or expired, otherwise the cached results dict.
        """
        cache_entry = self._indicator_cache.get(instrument)
        if not cache_entry:
            return None
        
        # Check TTL
        cached_time = cache_entry.get('timestamp', 0)
        if time.time() - cached_time > self._cache_ttl_seconds:
            return None
        
        # Check price hash
        price_hash = self._compute_price_hash(prices)
        if cache_entry.get('hash') != price_hash:
            return None
        
        return cache_entry.get('results')
    
    def _set_cached_indicators(self, instrument: str, prices: list, results: Dict[str, Any]) -> None:
        """Cache indicator calculation results for an instrument."""
        self._indicator_cache[instrument] = {
            'hash': self._compute_price_hash(prices),
            'results': results,
            'timestamp': time.time(),
        }
    
    # ────────────────────────────────────────────────────────────────
    # Position Focus Mode (Per-Instrument)
    # ────────────────────────────────────────────────────────────────
    
    def _get_position_focus_context(self) -> Optional[Dict[str, Any]]:
        """
        Get the position focus context from the bus.
        Returns None if no position focus is active.
        """
        ctx = self.smart_bus.get('position_focus_context', self.__class__.__name__, default=None)
        if not ctx or not isinstance(ctx, dict):
            return None
        if not ctx.get('focus_mode_active', False):
            return None
        return ctx
    
    def _is_position_focus_mode(self) -> bool:
        """Check if we're in position focus mode (have active positions to manage)."""
        ctx = self._get_position_focus_context()
        return ctx is not None and ctx.get('focus_mode_active', False)
    
    def _get_position_for_instrument(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Get position details for a specific instrument.
        Returns None if no position exists for this instrument.
        
        This supports MULTIPLE instruments having positions simultaneously
        (e.g., XAUUSD and EURUSD can both have active positions).
        """
        ctx = self._get_position_focus_context()
        if not ctx:
            return None
        
        positions = ctx.get('positions', {})
        
        # Normalize instrument name for lookup
        inst_norm = instrument.upper().replace('/', '').replace('_', '').replace('-', '')
        
        # Direct lookup
        if inst_norm in positions:
            return positions[inst_norm]
        
        # Try variations
        for key in positions.keys():
            key_norm = key.upper().replace('/', '').replace('_', '').replace('-', '')
            if key_norm == inst_norm:
                return positions[key]
        
        return None
    
    def _has_position_for_instrument(self, instrument: str) -> bool:
        """Check if we have a position for a specific instrument."""
        return self._get_position_for_instrument(instrument) is not None
    
    def _get_position_side_for_instrument(self, instrument: str) -> int:
        """
        Get position side for instrument: 1=LONG, -1=SHORT, 0=FLAT.
        """
        pos = self._get_position_for_instrument(instrument)
        if not pos:
            return 0
        return int(pos.get('side', 0))
    
    def _evaluate_signal_for_position(
        self,
        proposal: Dict[str, Any],
        instrument: str,
    ) -> Tuple[Dict[str, Any], float, bool]:
        """
        Evaluate an expert's signal in the context of an existing position FOR A SPECIFIC INSTRUMENT.
        
        This method is per-instrument aware - it checks if THIS instrument has a position
        and evaluates the signal accordingly. Other instruments without positions
        can still generate normal signals.
        
        Instead of generating a new trade signal, we evaluate:
        - Does this signal SUPPORT holding the current position?
        - Does this signal THREATEN the current position (suggest exit)?
        - How confident are we in this assessment?
        
        Returns:
            (modified_proposal, confidence, supports_position)
        """
        # Get position for THIS specific instrument
        inst_position = self._get_position_for_instrument(instrument)
        
        if not inst_position:
            # No position for this instrument - return original proposal unchanged
            original_conf = float(proposal.get('confidence', proposal.get('signal_strength', 0.5)))
            return proposal, original_conf, True
        
        position_side = int(inst_position.get('side', 0))
        position_pnl = float(inst_position.get('unrealized_pnl', inst_position.get('pnl', 0.0)))
        position_entry = float(inst_position.get('entry_price', 0.0))
        
        # Get the expert's original action for this instrument
        action = str(proposal.get('action', 'hold')).lower()
        original_confidence = float(proposal.get('confidence', proposal.get('signal_strength', 0.5)))
        
        # Check if this expert's signal is for our instrument (per-instrument proposals)
        per_instrument = proposal.get('proposals', proposal.get('per_instrument', {}))
        inst_norm = instrument.upper().replace('/', '').replace('_', '').replace('-', '')
        
        if isinstance(per_instrument, dict):
            for key, inst_proposal in per_instrument.items():
                key_norm = str(key).upper().replace('/', '').replace('_', '').replace('-', '')
                if key_norm == inst_norm and isinstance(inst_proposal, dict):
                    action = str(inst_proposal.get('action', action)).lower()
                    original_confidence = float(
                        inst_proposal.get('confidence', inst_proposal.get('signal_strength', original_confidence))
                    )
                    break
        
        # Determine if signal supports or threatens the position
        supports_position = True
        position_evaluation = "neutral"
        
        if position_side > 0:  # LONG position
            if action in ('buy', 'long', 'scale_up'):
                supports_position = True
                position_evaluation = "supports_long"
            elif action in ('sell', 'short', 'close', 'exit'):
                supports_position = False
                position_evaluation = "threatens_long"
            else:  # hold, flat, abstain, tighten
                supports_position = True
                position_evaluation = "neutral_for_long"
        elif position_side < 0:  # SHORT position
            if action in ('sell', 'short', 'scale_up'):
                supports_position = True
                position_evaluation = "supports_short"
            elif action in ('buy', 'long', 'close', 'exit'):
                supports_position = False
                position_evaluation = "threatens_short"
            else:
                supports_position = True
                position_evaluation = "neutral_for_short"
        
        # Modify proposal to reflect position management context
        modified_proposal: Dict[str, Any] = dict(proposal)
        modified_proposal['position_management_mode'] = True
        modified_proposal['position_instrument'] = instrument
        modified_proposal['supports_position'] = supports_position
        modified_proposal['position_evaluation'] = position_evaluation
        modified_proposal['position_side'] = position_side
        modified_proposal['original_action'] = action
        modified_proposal['position_entry_price'] = position_entry
        modified_proposal['position_unrealized_pnl'] = position_pnl
        
        # Remap action to position management actions
        if supports_position:
            # When signal supports position, default to HOLD; scaling decisions are handled upstream/downstream.
            if action in ('buy', 'sell', 'long', 'short', 'scale_up'):
                modified_proposal['action'] = 'hold'
            else:
                modified_proposal['action'] = 'hold'
        else:
            # Signal threatens position
            if original_confidence > 0.7:
                modified_proposal['action'] = 'exit'      # Strong opposing signal
            elif original_confidence > 0.5:
                modified_proposal['action'] = 'tighten'   # Moderate opposing signal
            else:
                modified_proposal['action'] = 'hold'      # Weak opposing signal, just watch
        
        # Adjust confidence based on position context
        confidence = original_confidence
        if position_pnl > 0:
            # Position is profitable - be more conservative about exits
            if not supports_position:
                confidence *= 0.8  # Reduce exit confidence when in profit
        elif position_pnl < 0:
            # Position is losing - be more responsive to exit signals
            if not supports_position:
                confidence *= 1.2  # Increase exit confidence when losing
        
        confidence = max(0.0, min(1.0, confidence))
        
        return modified_proposal, confidence, supports_position
    
    def _evaluate_per_instrument_proposals_for_positions(
        self,
        proposals_dict: Dict[str, Any],
        position_focus: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, bool]]:
        """
        Evaluate all per-instrument proposals against their respective positions.
        
        For each instrument:
        - If it has a position: reframe signal as supports/threatens
        - If no position: keep original signal for potential new entry (but may be blocked downstream)
        
        Returns:
            (modified_proposals_dict, supports_dict)
            - modified_proposals_dict: Same structure but with position_management fields
            - supports_dict: {instrument: bool} indicating if signal supports position
        """
        if not position_focus or not position_focus.get('focus_mode_active', False):
            # No position focus mode - return originals
            # Normalize proposals_dict into the expected shape
            normalized: Dict[str, Dict[str, Any]] = {}
            for k, v in proposals_dict.items():
                if isinstance(v, dict):
                    normalized[str(k)] = v
            return normalized, {}
        
        modified_proposals: Dict[str, Dict[str, Any]] = {}
        supports_dict: Dict[str, bool] = {}
        
        positions = position_focus.get('positions', {})
        
        for instrument, prop_any in proposals_dict.items():
            # Ensure we are working with a dict
            if not isinstance(prop_any, dict):
                continue
            prop: Dict[str, Any] = dict(prop_any)
            inst_norm = str(instrument).upper().replace('/', '').replace('_', '').replace('-', '')
            
            # Check if this instrument has a position
            inst_position = None
            for pos_key, pos_data in positions.items():
                pos_key_norm = str(pos_key).upper().replace('/', '').replace('_', '').replace('-', '')
                if pos_key_norm == inst_norm:
                    inst_position = pos_data
                    break
            
            if inst_position:
                # This instrument has a position - evaluate signal against it
                position_side = int(inst_position.get('side', 0))
                position_pnl = float(inst_position.get('unrealized_pnl', inst_position.get('pnl', 0.0)))
                
                action = str(prop.get('action', 'hold')).lower()
                original_confidence = float(prop.get('confidence', prop.get('signal_strength', 0.5)))
                
                # Determine support
                supports = True
                evaluation = "neutral"
                
                if position_side > 0:  # LONG
                    if action in ('sell', 'short', 'close', 'exit'):
                        supports = False
                        evaluation = "threatens_long"
                    else:
                        supports = True
                        evaluation = "supports_long" if action in ('buy', 'long') else "neutral_for_long"
                elif position_side < 0:  # SHORT
                    if action in ('buy', 'long', 'close', 'exit'):
                        supports = False
                        evaluation = "threatens_short"
                    else:
                        supports = True
                        evaluation = "supports_short" if action in ('sell', 'short') else "neutral_for_short"
                
                # Modify proposal
                modified_prop: Dict[str, Any] = dict(prop)
                modified_prop['position_management_mode'] = True
                modified_prop['position_instrument'] = instrument
                modified_prop['supports_position'] = supports
                modified_prop['position_evaluation'] = evaluation
                modified_prop['position_side'] = position_side
                modified_prop['original_action'] = action
                modified_prop['position_unrealized_pnl'] = position_pnl
                
                # Remap action for position management
                if supports:
                    modified_prop['action'] = 'hold'
                else:
                    if original_confidence > 0.7:
                        modified_prop['action'] = 'exit'
                    elif original_confidence > 0.5:
                        modified_prop['action'] = 'tighten'
                    else:
                        modified_prop['action'] = 'hold'
                
                # Adjust confidence
                conf = original_confidence
                if position_pnl > 0 and not supports:
                    conf *= 0.8
                elif position_pnl < 0 and not supports:
                    conf *= 1.2
                conf = max(0.0, min(1.0, conf))
                modified_prop['confidence'] = conf
                
                modified_proposals[str(instrument)] = modified_prop
                supports_dict[str(instrument)] = supports
            else:
                # No position for this instrument - keep original but mark for potential blocking
                modified_prop = dict(prop)
                modified_prop['position_management_mode'] = False
                modified_prop['no_position_for_instrument'] = True
                modified_proposals[str(instrument)] = modified_prop
                supports_dict[str(instrument)] = True  # N/A really
        
        return modified_proposals, supports_dict

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
            
            # 3.5 CHECK POSITION FOCUS MODE
            # If we have active positions, switch to position management mode
            position_focus = self._get_position_focus_context()
            in_position_focus_mode = position_focus is not None and position_focus.get('focus_mode_active', False)
            
            # 4. Expert-specific proposal
            raw_proposal = await self._generate_expert_specific_proposal(market_data)
            if not isinstance(raw_proposal, dict):
                proposal: Dict[str, Any] = {'action': 'abstain', 'reason': 'invalid_proposal_type'}
            else:
                # Copy to avoid weird aliasing if subclass keeps references
                proposal = dict(raw_proposal)
            
            # 5. Expert-specific confidence
            confidence = await self._calculate_expert_specific_confidence(
                proposal, market_data
            )
            confidence = max(0.0, min(1.0, float(confidence)))
            
            # 5.5 POSITION FOCUS MODE: Reframe signals for position management (PER-INSTRUMENT)
            supports_position = True
            if in_position_focus_mode and position_focus is not None:
                # Evaluate per-instrument proposals against their positions
                per_inst_raw: Any = proposal.get('proposals')
                if not isinstance(per_inst_raw, dict):
                    per_inst_raw = proposal.get('per_instrument')
                
                per_inst_proposals: Dict[str, Any] = {}
                if isinstance(per_inst_raw, dict):
                    # Normalize keys to str and ensure dict values
                    for k, v in per_inst_raw.items():
                        if isinstance(v, dict):
                            per_inst_proposals[str(k)] = v
                
                if per_inst_proposals:
                    modified_proposals, supports_dict = self._evaluate_per_instrument_proposals_for_positions(
                        per_inst_proposals, position_focus
                    )
                    proposal_mut = cast(Dict[str, Any], proposal)
                    proposal_mut['proposals'] = modified_proposals
                    proposal_mut['per_instrument'] = modified_proposals
                    proposal_mut['position_supports'] = supports_dict
                    proposal = proposal_mut
                    
                    # Log per-instrument position evaluations
                    for inst, supports in supports_dict.items():
                        inst_eval = modified_proposals.get(inst, {}).get('position_evaluation', 'N/A')
                        self.logger.debug(
                            f"[{name}] {inst}: supports_position={supports}, eval={inst_eval}"
                        )
                else:
                    # No per-instrument proposals - evaluate global against primary instrument
                    primary_inst = position_focus.get('primary_instrument', 'EURUSD')
                    proposal, confidence, supports_position = self._evaluate_signal_for_position(
                        proposal, primary_inst
                    )
                    self.logger.debug(
                        f"[{name}] Position focus mode: supports={supports_position}, "
                        f"action={proposal.get('action')}, conf={confidence:.2%}"
                    )
            
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
            per_instrument_votes_raw: Any = (
                proposal.get('proposals') or proposal.get('per_instrument') or {}
            )
            if isinstance(per_instrument_votes_raw, dict):
                per_instrument_votes: Dict[str, Any] = dict(per_instrument_votes_raw)
            else:
                per_instrument_votes = {}
            
            # Add position management info to output
            output: Dict[str, Any] = {
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
            
            # Include position management context in output
            if in_position_focus_mode:
                output['position_focus_mode'] = True
                output['supports_position'] = supports_position
                output['position_evaluation'] = proposal.get('position_evaluation', 'neutral')
            
            return output
        
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
    ) -> Tuple[Dict[str, Any], float]:
        """
        Normalize and gate the proposal using mode-aware thresholds.
        
        - Ensures signal_strength is in [0, 1]
        - Ensures position_size respects max_signal_strength
        - Soft-kills weak directional signals (turns into 'flat')
        - Keeps per-instrument payloads intact (we only touch top-level fields)
        - PRESERVES 'exit'/'tighten' actions for position management mode
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
        is_exit = (action_raw == 'exit')
        is_tighten = (action_raw == 'tighten')
        
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
        
        # Exit / tighten: keep semantics; they are critical for position lock-in
        elif is_exit or is_tighten:
            proposal['action'] = action_raw
            # For exits/tighten, treat weak signals as soft, but don't kill them.
            effective_min = min_strength * 0.75
            proposal['signal_strength'] = max(sig_f, effective_min)
            proposal['position_size'] = min(
                proposal['signal_strength'], self.max_signal_strength
            )
            # Allow lower confidence floor than directional entries, but don't clamp to 0
            confidence = max(conf_floor * 0.5, min(confidence, 1.0))
        
        # Flat-like: explicit "no trade" stance
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
        
        proposal: Dict[str, Any] = {
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
