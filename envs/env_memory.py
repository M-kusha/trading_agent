# envs/env_memory.py
"""
Enhanced memory integration with InfoBus support
Maintains backward compatibility while adding InfoBus infrastructure  
"""
from typing import List, Dict, Any
import numpy as np

from modules.utils.info_bus import InfoBus, create_info_bus, extract_standard_context
from modules.utils.audit_utils import format_operator_message


def _feed_memory_modules(self, info_bus: InfoBus):
    """Enhanced memory module feeding with InfoBus integration"""
    
    try:
        # Extract context and trades from InfoBus
        context = extract_standard_context(info_bus)
        recent_trades = info_bus.get('recent_trades', [])
        
        if not recent_trades and self.config.debug:
            self.logger.debug("No recent trades to feed memory modules")
            return
        
        # Feed memory modules that handle InfoBus directly
        self._feed_info_bus_memory_modules(info_bus, context)
        
        # Feed legacy memory modules with extracted data
        self._feed_legacy_memory_modules(recent_trades, context, info_bus)
        
        # Update memory performance tracking
        self._update_memory_performance_tracking(info_bus, len(recent_trades))
        
        self.logger.debug(
            format_operator_message(
                "🧠", "MEMORY_MODULES_FED",
                details=f"Fed {len(recent_trades)} trades to memory systems",
                context="memory_processing"
            )
        )
        
    except Exception as e:
        self.logger.error(
            format_operator_message(
                "💥", "MEMORY_FEED_ERROR",
                details=f"Memory module feeding failed: {e}",
                context="memory_processing"
            )
        )


def _feed_info_bus_memory_modules(self, info_bus: InfoBus, context: Dict[str, Any]):
    """Feed memory modules that support InfoBus directly"""
    
    # These modules use _step_impl with InfoBus
    info_bus_modules = [
        ('PlaybookMemory', self.playbook_memory),
        ('MemoryCompressor', self.memory_compressor),
        ('HistoricalReplayAnalyzer', self.replay_analyzer),
    ]
    
    for module_name, module in info_bus_modules:
        try:
            if hasattr(module, '_step_impl'):
                module._step_impl(info_bus=info_bus)
            elif hasattr(module, 'step'):
                module.step(info_bus=info_bus)
            
        except Exception as e:
            self.logger.error(
                format_operator_message(
                    "💥", "MODULE_FEED_ERROR",
                    instrument=module_name,
                    details=str(e),
                    context="memory_processing"
                )
            )


def _feed_legacy_memory_modules(self, trades: List[Dict], context: Dict[str, Any], info_bus: InfoBus):
    """Feed memory modules that still need legacy interface"""
    
    if not trades:
        return
    
    # Extract features for legacy modules
    features = self._extract_memory_features(info_bus, context)
    
    # Feed MistakeMemory with individual trades
    try:
        for trade in trades:
            trade_features = trade.get("features", features)
            pnl = trade.get("pnl", 0.0)
            
            # Use InfoBus step if available, otherwise legacy
            if hasattr(self.mistake_memory, '_step_impl'):
                trade_info_bus = info_bus.copy()
                trade_info_bus['current_trade'] = trade
                self.mistake_memory._step_impl(info_bus=trade_info_bus)
            else:
                self.mistake_memory.step(
                    features=trade_features,
                    pnl=pnl,
                    info=context
                )
            
            # Update budget optimizer
            if pnl != 0:
                self.memory_budget.step(
                    memory_used="mistakes",
                    profit=pnl,
                    source="mistakes"
                )
                
    except Exception as e:
        self.logger.error(f"Error feeding MistakeMemory: {e}")
    
    # Feed MemoryBudgetOptimizer
    try:
        if hasattr(self.memory_budget, '_step_impl'):
            self.memory_budget._step_impl(info_bus=info_bus)
        else:
            # Legacy interface
            total_profit = sum(t.get("pnl", 0.0) for t in trades)
            if total_profit != 0:
                self.memory_budget.step(
                    memory_used="general",
                    profit=total_profit,
                    source="trading"
                )
    except Exception as e:
        self.logger.error(f"Error feeding MemoryBudgetOptimizer: {e}")


def _extract_memory_features(self, info_bus: InfoBus, context: Dict[str, Any]) -> np.ndarray:
    """Extract features for memory modules from InfoBus"""
    
    try:
        features = []
        
        # Market context features
        features.extend([
            context.get('volatility_level', 0.5),  # Normalized volatility
            context.get('drawdown_pct', 0.0),
            context.get('exposure_pct', 0.0),
            float(context.get('position_count', 0)),
            context.get('risk_score', 0.5),
        ])
        
        # Price features if available
        prices = info_bus.get('prices', {})
        if prices and self.instruments:
            primary_price = prices.get(self.instruments[0], 0.0)
            features.append(primary_price / 10000.0)  # Normalized price
        else:
            features.append(0.0)
        
        # Session and regime features
        session_map = {'london': 0.2, 'new_york': 0.5, 'tokyo': 0.8, 'sydney': 0.1, 'unknown': 0.0}
        features.append(session_map.get(context.get('session', 'unknown'), 0.0))
        
        regime_map = {'trending': 0.8, 'volatile': 0.5, 'ranging': 0.2, 'unknown': 0.0}
        features.append(regime_map.get(context.get('regime', 'unknown'), 0.0))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        self.logger.warning(f"Feature extraction failed: {e}")
        return np.zeros(8, dtype=np.float32)  # Safe fallback


def _get_current_market_context(self) -> Dict[str, Any]:
    """Enhanced market context extraction with InfoBus support"""
    
    try:
        # Create InfoBus for current state
        info_bus = create_info_bus(self, step=self.market_state.current_step)
        
        # Extract standard context
        context = extract_standard_context(info_bus)
        
        # Add environment-specific context
        context.update({
            'balance': self.market_state.balance,
            'peak_balance': self.market_state.peak_balance,
            'session_pnl': getattr(self.market_state, 'session_pnl', 0.0),
            'trades_today': getattr(self.market_state, 'session_trades', 0),
            'step': self.market_state.current_step,
        })
        
        return context
        
    except Exception as e:
        self.logger.error(f"Error getting market context: {e}")
        
        # Fallback context
        return {
            'volatility': self._get_current_volatility(),
            'regime': 'unknown',
            'hour': 12,  # Default to midday
            'step': self.market_state.current_step,
            'balance': self.market_state.balance,
            'drawdown': self.market_state.current_drawdown,
            'session': 'unknown',
            'volatility_level': 'medium',
            'risk_score': 0.5,
        }


def _update_memory_compressor(self, episode_trades: List[Dict]):
    """Enhanced memory compressor update with InfoBus integration"""
    
    try:
        if not episode_trades:
            self.logger.debug("No episode trades for memory compressor")
            return
        
        # Create InfoBus for compression
        info_bus = create_info_bus(self, step=self.market_state.current_step)
        info_bus['episode_trades'] = episode_trades
        info_bus['episode_number'] = self.episode_count
        
        # Add enhanced trade features
        enhanced_trades = []
        for trade in episode_trades:
            enhanced_trade = self._enhance_trade_for_compression(trade, info_bus)
            enhanced_trades.append(enhanced_trade)
        
        # Use InfoBus compression if available
        if hasattr(self.memory_compressor, '_step_impl'):
            info_bus['trades_for_compression'] = enhanced_trades
            self.memory_compressor._step_impl(info_bus=info_bus)
        else:
            # Legacy compression
            self.memory_compressor.compress(
                episode=self.episode_count,
                trades=enhanced_trades
            )
        
        self.logger.info(
            format_operator_message(
                "🧠", "MEMORY_COMPRESSED",
                details=f"Compressed {len(enhanced_trades)} trades for episode {self.episode_count}",
                context="memory_management"
            )
        )
        
    except Exception as e:
        self.logger.error(
            format_operator_message(
                "💥", "COMPRESSION_ERROR", 
                details=f"Memory compression failed: {e}",
                context="memory_management"
            )
        )


def _enhance_trade_for_compression(self, trade: Dict[str, Any], info_bus: InfoBus) -> Dict[str, Any]:
    """Enhance trade data for memory compression"""
    
    enhanced_trade = trade.copy()
    
    # Ensure features exist
    if "features" not in enhanced_trade:
        enhanced_trade["features"] = np.array([
            trade.get("entry_price", 0.0),
            trade.get("exit_price", 0.0), 
            trade.get("size", 0.0),
            trade.get("pnl", 0.0),
            self._get_current_volatility(),
            self.market_state.current_drawdown,
            self.market_state.balance / 10000.0,
            float(self.market_state.current_step)
        ], dtype=np.float32)
    
    # Add market context
    context = extract_standard_context(info_bus)
    enhanced_trade["market_context"] = context
    
    # Add timing information
    enhanced_trade["episode"] = self.episode_count
    enhanced_trade["step"] = self.market_state.current_step
    
    return enhanced_trade


def _record_episode_in_replay_analyzer(self):
    """Enhanced episode recording with InfoBus integration"""
    
    try:
        total_pnl = sum(self.episode_metrics.pnls) if self.episode_metrics.pnls else 0.0
        
        # Create InfoBus for episode recording
        info_bus = create_info_bus(self, step=self.market_state.current_step)
        info_bus['episode_summary'] = {
            'episode_number': self.episode_count,
            'total_pnl': total_pnl,
            'trade_count': len(self.episode_metrics.trades),
            'max_drawdown': max(self.episode_metrics.drawdowns) if self.episode_metrics.drawdowns else 0.0,
            'final_balance': self.market_state.balance,
        }
        
        # Use InfoBus recording if available
        if hasattr(self.replay_analyzer, '_step_impl'):
            self.replay_analyzer._step_impl(info_bus=info_bus)
        else:
            # Legacy recording
            if len(self.episode_metrics.pnls) > 0:
                actions = getattr(self, '_last_actions', np.array([0, 0]))
                self.replay_analyzer.record_episode(
                    data={"episode": self.episode_count},
                    actions=actions,
                    pnl=total_pnl
                )
        
        self.logger.info(
            format_operator_message(
                "📊", "EPISODE_RECORDED",
                details=f"Episode {self.episode_count} recorded: PnL={total_pnl:+.2f}",
                context="episode_tracking"
            )
        )
        
    except Exception as e:
        self.logger.error(
            format_operator_message(
                "💥", "EPISODE_RECORDING_ERROR",
                details=f"Failed to record episode: {e}",
                context="episode_tracking"
            )
        )


def _update_memory_performance_tracking(self, info_bus: InfoBus, trade_count: int):
    """Track memory module performance"""
    
    try:
        # Update InfoBus with memory performance data
        memory_performance = {
            'trade_count': trade_count,
            'step': self.market_state.current_step,
            'episode': self.episode_count,
            'memory_usage': self._get_memory_usage_stats(),
        }
        
        from modules.utils.info_bus import InfoBusUpdater
        InfoBusUpdater.add_module_data(info_bus, 'MemorySystem', memory_performance)
        
    except Exception as e:
        self.logger.warning(f"Memory performance tracking failed: {e}")


def _get_memory_usage_stats(self) -> Dict[str, Any]:
    """Get memory usage statistics from all memory modules"""
    
    stats = {}
    
    try:
        # PlaybookMemory stats
        if hasattr(self.playbook_memory, 'memory_records'):
            stats['playbook_entries'] = len(self.playbook_memory.memory_records)
        
        # MistakeMemory stats  
        if hasattr(self.mistake_memory, '_loss_buf'):
            stats['mistake_entries'] = len(self.mistake_memory._loss_buf)
        
        # MemoryCompressor stats
        if hasattr(self.memory_compressor, 'profit_memory'):
            stats['compressed_entries'] = len(self.memory_compressor.profit_memory)
        
        # ReplayAnalyzer stats
        if hasattr(self.replay_analyzer, 'episode_buffer'):
            stats['replay_episodes'] = len(self.replay_analyzer.episode_buffer)
            
    except Exception as e:
        self.logger.warning(f"Memory stats collection failed: {e}")
    
    return stats


# Backward compatibility methods
def feed_memory_modules_legacy(self, trades: List[Dict], actions: np.ndarray, obs: np.ndarray):
    """Legacy method for feeding memory modules - converts to InfoBus"""
    
    # Create InfoBus from legacy parameters
    info_bus = create_info_bus(self, step=self.market_state.current_step)
    info_bus['recent_trades'] = trades
    info_bus['raw_actions'] = actions
    info_bus['observation'] = obs
    
    # Use enhanced method
    self._feed_memory_modules(info_bus)