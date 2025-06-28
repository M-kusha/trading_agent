# envs/env_memory.py
"""
Memory integration methods for the trading environment
"""
from typing import List, Dict, Any
import numpy as np


def _feed_memory_modules(self, trades: List[Dict], actions: np.ndarray, obs: np.ndarray):
    """FIXED: Properly feed data to memory modules"""
    try:
        # Extract features for memory modules
        features = obs[:32] if len(obs) >= 32 else np.pad(obs, (0, max(0, 32 - len(obs))))
        
        # Get current market context
        context = self._get_current_market_context()
        
        # Feed MistakeMemory with individual trades
        if hasattr(self.mistake_memory, 'step'):
            for trade in trades:
                try:
                    self.mistake_memory.step(
                        features=trade.get("features", features),
                        pnl=trade.get("pnl", 0.0),
                        info=context
                    )
                    
                    # Update budget optimizer with memory usage
                    if trade.get("pnl", 0) != 0:
                        self.memory_budget.step(
                            memory_used="mistakes",
                            profit=trade.get("pnl", 0.0),
                            source="mistakes"
                        )
                except Exception as e:
                    self.logger.error(f"Error feeding MistakeMemory: {e}")
        
        # Feed PlaybookMemory with trade records
        if hasattr(self.playbook_memory, 'record') and trades:
            for trade in trades:
                try:
                    self.playbook_memory.record(
                        features=trade.get("features", features),
                        actions=actions,
                        pnl=trade.get("pnl", 0.0),
                        context=context
                    )
                    
                    # Update budget optimizer
                    if trade.get("pnl", 0) != 0:
                        self.memory_budget.step(
                            memory_used="trades",
                            profit=trade.get("pnl", 0.0),
                            source="trades"
                        )
                except Exception as e:
                    self.logger.error(f"Error feeding PlaybookMemory: {e}")
        
        # FIXED: DON'T feed MemoryCompressor here - it gets episode data separately
        # MemoryCompressor accumulates over episodes and compresses at intervals
        self._feed_memory_compressor_step_by_step(trades)
        
        # Feed HistoricalReplayAnalyzer with action sequence
        if hasattr(self.replay_analyzer, 'step'):
            try:
                self.replay_analyzer.step(
                    action=actions,
                    features=features,
                    timestamp=self.market_state.current_step
                )
            except Exception as e:
                self.logger.error(f"Error feeding HistoricalReplayAnalyzer: {e}")
                
        self.logger.debug(f"Fed {len(trades)} trades to memory modules")
        
    except Exception as e:
        self.logger.error(f"Error in _feed_memory_modules: {e}")


def _feed_memory_compressor_step_by_step(self, trades: List[Dict]):
    """Feed MemoryCompressor with individual trades as they happen"""
    try:
        if hasattr(self.memory_compressor, 'step') and trades:
            for trade in trades:
                # Create enhanced trade data for compression
                enhanced_trade = {
                    "features": trade.get("features", np.array([
                        trade.get("entry_price", 0.0),
                        trade.get("exit_price", 0.0), 
                        trade.get("size", 0.0),
                        trade.get("pnl", 0.0),
                        self._get_current_volatility(),
                        self.market_state.current_drawdown,
                        self.market_state.balance / 10000.0,
                        float(self.market_state.current_step)
                    ], dtype=np.float32)),
                    "pnl": trade.get("pnl", 0.0)
                }
                
                # Feed individual trade to memory compressor
                # This will build up its internal memory between episodes
                self.memory_compressor.compress(
                    episode=self.episode_count, 
                    trades=[enhanced_trade]
                )
                
            self.logger.debug(f"MemoryCompressor: Fed {len(trades)} individual trades")
            
    except Exception as e:
        self.logger.error(f"Error feeding MemoryCompressor step-by-step: {e}")


def _get_current_market_context(self) -> Dict[str, Any]:
    """Get current market context for memory modules"""
    try:
        # Get volatility and regime info
        vol = self._get_current_volatility()
        
        # Get current hour (for session detection)
        import datetime
        current_hour = datetime.datetime.now().hour
        
        # Determine regime (simplified)
        regime = "trending"  # Could be enhanced with actual regime detection
        if vol > 0.03:
            regime = "volatile"
        elif vol < 0.01:
            regime = "noise"
            
        return {
            "volatility": vol,
            "regime": regime,
            "hour": current_hour,
            "step": self.market_state.current_step,
            "balance": self.market_state.balance,
            "drawdown": self.market_state.current_drawdown
        }
    except Exception as e:
        self.logger.error(f"Error getting market context: {e}")
        return {}


def _update_memory_compressor(self, episode_trades: List[Dict]):
    """FIXED: Force compression for testing (remove after verification)"""
    try:
        if hasattr(self.memory_compressor, 'compress') and episode_trades:
            # Add features to trades if missing
            enhanced_trades = []
            for trade in episode_trades:
                if "features" not in trade:
                    # Create features from trade data
                    trade_features = np.array([
                        trade.get("entry_price", 0.0),
                        trade.get("exit_price", 0.0),
                        trade.get("size", 0.0),
                        trade.get("duration", 0.0),
                        trade.get("pnl", 0.0),
                        self._get_current_volatility(),
                        self.market_state.current_drawdown,
                        self.market_state.balance / 10000.0
                    ], dtype=np.float32)
                    trade["features"] = trade_features
                enhanced_trades.append(trade)
                
            # FORCE compression every episode for testing
            self.memory_compressor.compress(
                episode=self.episode_count,
                trades=enhanced_trades
            )
            self.logger.info(f"MemoryCompressor: FORCED compression of {len(enhanced_trades)} trades for episode {self.episode_count}")
        else:
            self.logger.debug(f"MemoryCompressor: No trades to compress for episode {self.episode_count}")
    except Exception as e:
        self.logger.error(f"Error updating memory compressor: {e}")


def _record_episode_in_replay_analyzer(self):
    """FIXED: Record episode results in replay analyzer"""
    try:
        if hasattr(self.replay_analyzer, 'record_episode'):
            total_pnl = sum(self.episode_metrics.pnls)
            
            # Only record if we have meaningful data
            if len(self.episode_metrics.pnls) > 0:
                self.replay_analyzer.record_episode(
                    data={"episode": self.episode_count},
                    actions=np.array(self._last_actions if hasattr(self, '_last_actions') else [0, 0]),
                    pnl=total_pnl
                )
                self.logger.info(f"HistoricalReplayAnalyzer: Recorded episode {self.episode_count}, PnL=€{total_pnl:.2f}")
            else:
                self.logger.debug(f"HistoricalReplayAnalyzer: No PnL data for episode {self.episode_count}")
    except Exception as e:
        self.logger.error(f"Error recording episode in replay analyzer: {e}")