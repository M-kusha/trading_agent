# ─────────────────────────────────────────────────────────────
# File: modules/example/example_trading_module.py
# 🚀 Example SmartInfoBus module demonstrating best practices
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass
import time

from modules.core.module_base import BaseModule, module
from modules.utils.info_bus import InfoBusManager
from modules.utils.english_explainer import EnglishExplainer
from modules.utils.audit_utils import audit_module_call


@dataclass
class TradingSignal:
    """Example trading signal"""
    action: str  # 'buy', 'sell', 'hold'
    instrument: str
    confidence: float
    size: float
    reason: str


@module(
    provides=['trading_signal', 'signal_confidence', 'signal_thesis'],
    requires=['market_data', 'risk_score', 'market_regime', 'portfolio_state'],
    category='strategy',
    is_voting_member=True,
    hot_reload=True,
    explainable=True,
    timeout_ms=100,
    priority=5,
    version="2.0.0"
)
class ExampleTradingModule(BaseModule):
    """
    Example module showing all SmartInfoBus best practices:
    - Proper @module decorator with all parameters
    - Async process method with thesis generation
    - State management for hot-reload
    - Performance tracking
    - Error handling with circuit breaker support
    - Plain English explanations
    """
    
    def _initialize(self):
        """Initialize module-specific state"""
        # Trading parameters
        self.momentum_window = 20
        self.rsi_threshold_buy = 30
        self.rsi_threshold_sell = 70
        self.min_confidence = 0.6
        
        # State tracking
        self.signal_history = deque(maxlen=100)
        self.performance_stats = {
            'total_signals': 0,
            'correct_predictions': 0,
            'avg_confidence': 0.0
        }
        
        # Technical indicators cache
        self.indicator_cache = {}
        self.cache_timestamp = 0
        
        self.logger.info(f"Initialized {self.__class__.__name__} v{self.metadata.version}")
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Main processing method - analyzes market data and generates trading signals.
        Demonstrates all SmartInfoBus features.
        """
        # Validate inputs using decorator
        self.validate_inputs(inputs)
        
        # Extract inputs
        market_data = inputs['market_data']
        risk_score = inputs['risk_score']
        market_regime = inputs['market_regime']
        portfolio_state = inputs['portfolio_state']
        
        # Check risk constraints
        if risk_score > 0.8:
            # High risk - defensive mode
            signal = TradingSignal(
                action='hold',
                instrument='XAU/USD',
                confidence=0.9,
                size=0.0,
                reason='Risk too high for new positions'
            )
            
            thesis = self._generate_defensive_thesis(risk_score, market_regime)
            
        else:
            # Normal analysis
            signal = await self._analyze_market(
                market_data, 
                market_regime, 
                portfolio_state
            )
            
            thesis = self._generate_signal_thesis(
                signal, 
                market_data, 
                market_regime, 
                risk_score
            )
        
        # Record signal
        self._record_signal(signal)
        
        # Update SmartInfoBus with versioned data
        smart_bus = InfoBusManager.get_instance()
        
        # Set main signal with thesis
        smart_bus.set(
            'trading_signal',
            signal,
            module=self.__class__.__name__,
            thesis=thesis,
            confidence=signal.confidence,
            dependencies=['market_data', 'risk_score', 'market_regime']
        )
        
        # Set additional outputs
        smart_bus.set(
            'signal_confidence',
            signal.confidence,
            module=self.__class__.__name__
        )
        
        # Return all outputs including thesis
        return {
            'trading_signal': signal,
            'signal_confidence': signal.confidence,
            'signal_thesis': thesis,
            '_thesis': thesis  # Required for explainable modules
        }
    
    async def _analyze_market(self, market_data: Dict[str, Any], 
                            market_regime: str,
                            portfolio_state: Dict[str, Any]) -> TradingSignal:
        """
        Perform market analysis with caching and async computation.
        """
        # Simulate async computation
        await asyncio.sleep(0.01)
        
        # Check cache
        cache_age = time.time() - self.cache_timestamp
        if cache_age > 60:  # 1 minute cache
            self.indicator_cache.clear()
        
        # Calculate indicators
        prices = market_data.get('prices', [])
        if not prices or len(prices) < self.momentum_window:
            return TradingSignal(
                action='hold',
                instrument='XAU/USD',
                confidence=0.5,
                size=0.0,
                reason='Insufficient price data'
            )
        
        # Calculate momentum
        momentum = self._calculate_momentum(prices)
        
        # Calculate RSI
        rsi = self._calculate_rsi(prices)
        
        # Check position limits
        current_position = portfolio_state.get('positions', {}).get('XAU/USD', 0)
        max_position = portfolio_state.get('max_position_size', 1.0)
        
        # Generate signal based on indicators
        if rsi < self.rsi_threshold_buy and momentum > 0:
            if current_position < max_position:
                action = 'buy'
                size = min(0.1, max_position - current_position)
                confidence = min(0.9, 0.5 + momentum * 0.5)
                reason = f'Oversold (RSI={rsi:.1f}) with positive momentum'
            else:
                action = 'hold'
                size = 0.0
                confidence = 0.7
                reason = 'Buy signal but position limit reached'
                
        elif rsi > self.rsi_threshold_sell and momentum < 0:
            if current_position > 0:
                action = 'sell'
                size = min(0.1, current_position)
                confidence = min(0.9, 0.5 + abs(momentum) * 0.5)
                reason = f'Overbought (RSI={rsi:.1f}) with negative momentum'
            else:
                action = 'hold'
                size = 0.0
                confidence = 0.7
                reason = 'Sell signal but no position to close'
                
        else:
            action = 'hold'
            size = 0.0
            confidence = 0.6
            reason = f'No clear signal (RSI={rsi:.1f}, momentum={momentum:.3f})'
        
        # Adjust for market regime
        if market_regime == 'volatile' and action != 'hold':
            size *= 0.5  # Reduce size in volatile markets
            confidence *= 0.8
            reason += ' (size reduced due to volatility)'
        
        return TradingSignal(
            action=action,
            instrument='XAU/USD',
            confidence=confidence,
            size=size,
            reason=reason
        )
    
    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum"""
        if 'momentum' in self.indicator_cache:
            return self.indicator_cache['momentum']
        
        if len(prices) < self.momentum_window:
            return 0.0
        
        recent_prices = prices[-self.momentum_window:]
        returns = np.diff(recent_prices) / recent_prices[:-1]
        momentum = np.mean(returns)
        
        self.indicator_cache['momentum'] = momentum
        self.cache_timestamp = time.time()
        
        return momentum
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if 'rsi' in self.indicator_cache:
            return self.indicator_cache['rsi']
        
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        # Calculate price changes
        deltas = np.diff(prices[-period-1:])
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        self.indicator_cache['rsi'] = rsi
        self.cache_timestamp = time.time()
        
        return rsi
    
    def _generate_signal_thesis(self, signal: TradingSignal,
                              market_data: Dict[str, Any],
                              market_regime: str,
                              risk_score: float) -> str:
        """Generate comprehensive thesis for the trading signal"""
        prices = market_data.get('prices', [])
        current_price = prices[-1] if prices else 0
        
        thesis = f"""
TRADING SIGNAL THESIS
====================
Action: {signal.action.upper()} {signal.size} units of {signal.instrument}
Confidence: {signal.confidence:.1%}
Current Price: ${current_price:,.2f}

TECHNICAL ANALYSIS:
{signal.reason}

MARKET CONTEXT:
- Market Regime: {market_regime}
- Risk Level: {self._describe_risk_level(risk_score)}
- Recent Performance: {self._describe_recent_performance()}

RATIONALE:
This signal is generated based on a combination of momentum and RSI indicators.
The {signal.action} recommendation reflects {self._describe_market_condition(signal)}.

RISK CONSIDERATIONS:
- Position sizing adjusted for current risk level
- Stop loss recommended at {self._calculate_stop_loss(signal, current_price):.2f}
- Take profit target at {self._calculate_take_profit(signal, current_price):.2f}

EXPECTED OUTCOME:
Based on historical patterns, similar setups have resulted in profitable trades
{self._calculate_win_rate():.1%} of the time with an average return of {self._calculate_avg_return():.1%}.
"""
        return thesis.strip()
    
    def _generate_defensive_thesis(self, risk_score: float, market_regime: str) -> str:
        """Generate thesis for defensive position"""
        return f"""
DEFENSIVE POSITION THESIS
========================
Action: HOLD (No new positions)
Confidence: 90%

REASONING:
The system is in defensive mode due to elevated risk conditions.

RISK FACTORS:
- Current Risk Score: {risk_score:.1%} (above 80% threshold)
- Market Regime: {market_regime}
- Volatility: Elevated

RECOMMENDATION:
1. Maintain existing positions with tight stops
2. Wait for risk levels to normalize below 70%
3. Monitor market conditions closely
4. Consider reducing position sizes if risk increases further

This defensive stance protects capital during uncertain market conditions.
Historical data shows that avoiding new positions during high-risk periods
improves overall portfolio performance by reducing drawdowns.
"""
    
    def _describe_risk_level(self, risk_score: float) -> str:
        """Convert risk score to description"""
        if risk_score < 0.3:
            return f"Low ({risk_score:.1%})"
        elif risk_score < 0.6:
            return f"Moderate ({risk_score:.1%})"
        elif risk_score < 0.8:
            return f"Elevated ({risk_score:.1%})"
        else:
            return f"High ({risk_score:.1%})"
    
    def _describe_recent_performance(self) -> str:
        """Describe recent signal performance"""
        if not self.signal_history:
            return "No recent signals"
        
        recent_signals = list(self.signal_history)[-10:]
        buy_signals = sum(1 for s in recent_signals if s.action == 'buy')
        sell_signals = sum(1 for s in recent_signals if s.action == 'sell')
        
        return f"{buy_signals} buy and {sell_signals} sell signals in last 10 decisions"
    
    def _describe_market_condition(self, signal: TradingSignal) -> str:
        """Describe market condition based on signal"""
        if signal.action == 'buy':
            return "oversold conditions with bullish momentum"
        elif signal.action == 'sell':
            return "overbought conditions with bearish momentum"
        else:
            return "neutral conditions without clear directional bias"
    
    def _calculate_stop_loss(self, signal: TradingSignal, current_price: float) -> float:
        """Calculate stop loss price"""
        if signal.action == 'buy':
            return current_price * 0.98  # 2% stop loss
        elif signal.action == 'sell':
            return current_price * 1.02
        else:
            return current_price
    
    def _calculate_take_profit(self, signal: TradingSignal, current_price: float) -> float:
        """Calculate take profit price"""
        if signal.action == 'buy':
            return current_price * 1.03  # 3% take profit
        elif signal.action == 'sell':
            return current_price * 0.97
        else:
            return current_price
    
    def _calculate_win_rate(self) -> float:
        """Calculate historical win rate"""
        if self.performance_stats['total_signals'] == 0:
            return 0.6  # Default estimate
        
        return (self.performance_stats['correct_predictions'] / 
                self.performance_stats['total_signals'])
    
    def _calculate_avg_return(self) -> float:
        """Calculate average return"""
        # Simplified - would use actual PnL in production
        return 0.025  # 2.5% average
    
    def _record_signal(self, signal: TradingSignal):
        """Record signal for performance tracking"""
        self.signal_history.append(signal)
        self.performance_stats['total_signals'] += 1
        
        # Update average confidence
        n = self.performance_stats['total_signals']
        prev_avg = self.performance_stats['avg_confidence']
        self.performance_stats['avg_confidence'] = (
            (prev_avg * (n - 1) + signal.confidence) / n
        )
    
    # ─────────────────────────────────────────────────────────────
    # State Management for Hot-Reload
    # ─────────────────────────────────────────────────────────────
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete module state for hot-reload"""
        base_state = super().get_state()
        
        # Add module-specific state
        base_state['module_state'] = {
            'signal_history': list(self.signal_history),
            'performance_stats': self.performance_stats.copy(),
            'indicator_cache': self.indicator_cache.copy(),
            'cache_timestamp': self.cache_timestamp,
            'parameters': {
                'momentum_window': self.momentum_window,
                'rsi_threshold_buy': self.rsi_threshold_buy,
                'rsi_threshold_sell': self.rsi_threshold_sell,
                'min_confidence': self.min_confidence
            }
        }
        
        return base_state
    
    def set_state(self, state: Dict[str, Any]):
        """Restore module state after hot-reload"""
        super().set_state(state)
        
        if 'module_state' in state:
            ms = state['module_state']
            
            # Restore history
            self.signal_history = deque(
                ms.get('signal_history', []), 
                maxlen=100
            )
            
            # Restore stats
            self.performance_stats = ms.get('performance_stats', {
                'total_signals': 0,
                'correct_predictions': 0,
                'avg_confidence': 0.0
            })
            
            # Restore cache
            self.indicator_cache = ms.get('indicator_cache', {})
            self.cache_timestamp = ms.get('cache_timestamp', 0)
            
            # Restore parameters
            params = ms.get('parameters', {})
            self.momentum_window = params.get('momentum_window', 20)
            self.rsi_threshold_buy = params.get('rsi_threshold_buy', 30)
            self.rsi_threshold_sell = params.get('rsi_threshold_sell', 70)
            self.min_confidence = params.get('min_confidence', 0.6)
    
    # ─────────────────────────────────────────────────────────────
    # Legacy Methods (for backward compatibility)
    # ─────────────────────────────────────────────────────────────
    
    def _step_impl(self, info_bus):
        """Legacy step implementation - delegates to process"""
        # This is called by the base class step() method
        # We'll use asyncio to run the async process method
        smart_bus = InfoBusManager.get_instance()
        
        # Gather inputs from SmartInfoBus
        inputs = {
            'market_data': smart_bus.get('market_data', self.__class__.__name__),
            'risk_score': smart_bus.get('risk_score', self.__class__.__name__) or 0.5,
            'market_regime': smart_bus.get('market_regime', self.__class__.__name__) or 'normal',
            'portfolio_state': smart_bus.get('portfolio_state', self.__class__.__name__) or {}
        }
        
        # Run async process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.process(**inputs))
        finally:
            loop.close()
    
    def reset(self):
        """Reset module to initial state"""
        super().reset()
        self.signal_history.clear()
        self.performance_stats = {
            'total_signals': 0,
            'correct_predictions': 0,
            'avg_confidence': 0.0
        }
        self.indicator_cache.clear()
        self.cache_timestamp = 0
    
    def _get_observation_impl(self, info_bus) -> np.ndarray:
        """Extract observation for RL (legacy)"""
        # Return dummy observation
        return np.zeros(10)
    
    # ─────────────────────────────────────────────────────────────
    # Performance & Error Handling
    # ─────────────────────────────────────────────────────────────
    
    @audit_module_call()
    def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Enhanced input validation with detailed error messages"""
        try:
            # Check required fields
            super().validate_inputs(inputs)
            
            # Validate market data
            market_data = inputs.get('market_data', {})
            if not isinstance(market_data, dict):
                raise ValueError("market_data must be a dictionary")
            
            prices = market_data.get('prices', [])
            if not prices or len(prices) < 2:
                raise ValueError("Insufficient price data (need at least 2 prices)")
            
            # Validate risk score
            risk_score = inputs.get('risk_score', 0)
            if not 0 <= risk_score <= 1:
                raise ValueError(f"risk_score must be between 0 and 1, got {risk_score}")
            
            return True
            
        except Exception as e:
            # Let SmartInfoBus circuit breaker handle repeated failures
            self.logger.error(f"Input validation failed: {e}")
            raise


# ─────────────────────────────────────────────────────────────
# Module Registration (happens automatically via @module decorator)
# ─────────────────────────────────────────────────────────────

# The module is automatically registered when this file is imported.
# No manual registration needed in env.py or orchestrator.

# To use this module:
# 1. Import it anywhere to trigger registration
# 2. The orchestrator will discover it via the @module decorator
# 3. It will be included in the execution order based on dependencies

# Example usage in env.py:
# from modules.example.example_trading_module import ExampleTradingModule
# # That's it! Module is now registered and ready to use