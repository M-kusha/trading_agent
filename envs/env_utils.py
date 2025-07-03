# envs/env_utils.py
"""
Enhanced utility methods with InfoBus integration
Maintains backward compatibility while adding InfoBus infrastructure
"""
from datetime import datetime
import random
import os
import pickle
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from modules.utils.info_bus import InfoBus, create_info_bus, extract_standard_context, InfoBusUpdater
from modules.utils.audit_utils import format_operator_message


def _get_initial_balance(self) -> float:
    """Enhanced initial balance with InfoBus logging"""
    
    if self.config.live_mode:
        try:
            info = mt5.account_info()
            if info and hasattr(info, "balance"):
                balance = float(info.balance)
                self.logger.info(
                    format_operator_message(
                        "💰", "LIVE_BALANCE_DETECTED",
                        details=f"MT5 Balance: ${balance:,.2f}",
                        context="live_trading"
                    )
                )
                return balance
        except Exception as e:
            self.logger.warning(f"Failed to get MT5 balance: {e}")
    
    return self.config.initial_balance


def _select_starting_step(self) -> int:
    """Enhanced starting step selection with InfoBus logging"""
    
    # Find minimum available bars across instruments
    min_bars = min(
        len(self.data[inst]["D1"])
        for inst in self.instruments
        if "D1" in self.data[inst]
    )
    
    # Ensure we have enough data
    min_start = 100  # Need history for indicators
    max_start = max(min_bars - self.config.max_steps - 1, min_start)
    
    if max_start <= min_start:
        self.logger.warning(
            format_operator_message(
                "⚠️", "INSUFFICIENT_DATA",
                details=f"Only {min_bars} bars available",
                result="Starting at step 0",
                context="data_validation"
            )
        )
        return 0
    
    start_step = random.randint(min_start, max_start)
    
    self.logger.info(
        format_operator_message(
            "🎯", "EPISODE_START_SELECTED",
            details=f"Step {start_step} (range: {min_start}-{max_start})",
            context="episode_initialization"
        )
    )
    
    return start_step


def _reset_all_modules(self):
    """Enhanced module reset with InfoBus coordination"""
    
    reset_count = 0
    failed_resets = []
    
    # Reset pipeline modules
    if hasattr(self, 'pipeline'):
        try:
            self.pipeline.reset()
            reset_count += len(self.pipeline.modules)
        except Exception as e:
            self.logger.error(f"Pipeline reset failed: {e}")
            failed_resets.append("Pipeline")
    
    # Reset other core modules
    modules_to_reset = [
        ('RewardShaper', self.reward_shaper),
        ('MetaRL', self.meta_rl),
        ('Arbiter', self.arbiter),
        ('Consensus', self.consensus),
        ('Collusion', self.collusion),
        ('HorizonAligner', self.haligner),
        ('StrategyPool', self.strategy_pool),
        ('MetaAgent', self.meta_agent),
        ('MetaPlanner', self.meta_planner),
        ('LongTermMemory', self.long_term_memory),
        ('WorldModel', self.world_model)
    ]
    
    for name, module in modules_to_reset:
        if module and hasattr(module, 'reset'):
            try:
                module.reset()
                reset_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to reset {name}: {e}")
                failed_resets.append(name)
    
    # Log reset summary
    if failed_resets:
        self.logger.warning(
            format_operator_message(
                "⚠️", "PARTIAL_RESET",
                details=f"Reset {reset_count} modules, {len(failed_resets)} failed",
                result=f"Failed: {', '.join(failed_resets)}",
                context="module_management"
            )
        )
    else:
        self.logger.info(
            format_operator_message(
                "✅", "MODULES_RESET",
                details=f"Successfully reset {reset_count} modules",
                context="module_management"
            )
        )


def _prime_risk_system(self):
    """Enhanced risk system priming with InfoBus integration"""
    
    try:
        price_dict = {}
        for inst in self.instruments:
            df = self.data[inst]["D1"]
            start_idx = max(0, self.market_state.current_step - self.risk_system.var_window)
            end_idx = self.market_state.current_step + 1
            prices = df["close"].iloc[start_idx:end_idx].values
            price_dict[inst] = prices
            
        if all(len(p) > 10 for p in price_dict.values()):
            self.risk_system.prime_returns_with_history(price_dict)
            method = "historical"
        else:
            self.risk_system.prime_returns_with_random()
            method = "random"
            
        self.logger.info(
            format_operator_message(
                "🛡️", "RISK_SYSTEM_PRIMED",
                details=f"Method: {method}, Instruments: {len(price_dict)}",
                context="risk_management"
            )
        )
        
    except Exception as e:
        self.logger.warning(
            format_operator_message(
                "⚠️", "RISK_PRIMING_FAILED",
                details=str(e),
                result="Using random priming",
                context="risk_management"
            )
        )
        self.risk_system.prime_returns_with_random()


def _select_strategy_genome(self):
    """Enhanced strategy genome selection with InfoBus logging"""
    
    try:
        self.strategy_pool.select_genome("random")
        self.current_genome = self.strategy_pool.active_genome.copy()
        
        genome_summary = {k: v for k, v in self.current_genome.items() if isinstance(v, (int, float, str))}
        
        self.logger.info(
            format_operator_message(
                "🧬", "GENOME_SELECTED",
                details=f"ID: {genome_summary.get('id', 'unknown')}",
                result=f"Fitness: {genome_summary.get('fitness', 0.0):.3f}",
                context="strategy_management"
            )
        )
        
    except Exception as e:
        self.logger.error(f"Genome selection failed: {e}")
        self.current_genome = {}


def _get_current_volatility(self) -> float:
    """Enhanced volatility calculation with InfoBus integration"""
    
    try:
        df = self.data[self.instruments[0]]["D1"]
        if self.market_state.current_step < len(df):
            vol = float(df.iloc[self.market_state.current_step]["volatility"])
            return max(vol, 0.001)  # Minimum volatility floor
    except Exception:
        pass
    return 0.01  # Safe default


def _get_instrument_volatility(self, instrument: str) -> float:
    """Enhanced instrument-specific volatility with InfoBus integration"""
    
    try:
        df = self.data[instrument]["D1"]
        if self.market_state.current_step < len(df):
            vol = df.iloc[self.market_state.current_step].get("volatility", 0.01)
            vol = float(np.nan_to_num(vol, nan=0.01))
            return max(vol, getattr(self.position_manager, 'min_volatility', 0.001))
    except Exception:
        pass
    return getattr(self.position_manager, 'min_volatility', 0.001)


def _get_price_history(self, instrument: str, timeframe: str) -> np.ndarray:
    """Enhanced price history with InfoBus integration"""
    
    try:
        df = self.data[instrument][timeframe]
        end_idx = min(self.market_state.current_step + 1, len(df))
        start_idx = max(0, end_idx - 7)
        
        if end_idx > start_idx:
            prices = df["close"].iloc[start_idx:end_idx].values
            # Pad if needed
            if len(prices) < 7:
                prices = np.pad(prices, (7 - len(prices), 0), mode='edge')
            return prices[-7:].astype(np.float32)
    except Exception as e:
        if self.config.debug:
            self.logger.warning(f"Failed to get price history for {instrument}/{timeframe}: {e}")
        
    return np.zeros(7, dtype=np.float32)


def _get_recent_returns(self) -> Dict[str, np.ndarray]:
    """Enhanced returns calculation with InfoBus integration"""
    
    returns = {}
    for inst in self.instruments:
        try:
            df = self.data[inst]["D1"]
            end_idx = self.market_state.current_step + 1
            start_idx = max(0, end_idx - 20)
            
            if end_idx > start_idx + 1:
                prices = df["close"].iloc[start_idx:end_idx].values
                ret = np.diff(np.log(prices))
                returns[inst] = ret[np.isfinite(ret)]  # Filter out NaN/Inf
            else:
                returns[inst] = np.array([])
        except Exception:
            returns[inst] = np.array([])
            
    return returns


def get_instrument_correlations(self) -> Dict[str, float]:
    """Enhanced correlation calculation with InfoBus integration"""
    
    correlations = {}
    
    try:
        # Get returns for all instruments
        returns_dict = self._get_recent_returns()
        
        # Calculate pairwise correlations
        for i, inst1 in enumerate(self.instruments):
            for j, inst2 in enumerate(self.instruments):
                if i >= j:
                    continue
                    
                ret1 = returns_dict.get(inst1, np.array([]))
                ret2 = returns_dict.get(inst2, np.array([]))
                
                if len(ret1) > 5 and len(ret2) > 5:
                    # Align lengths
                    min_len = min(len(ret1), len(ret2))
                    ret1 = ret1[-min_len:]
                    ret2 = ret2[-min_len:]
                    
                    # Calculate correlation
                    if np.std(ret1) > 0 and np.std(ret2) > 0:
                        corr = np.corrcoef(ret1, ret2)[0, 1]
                        correlations[f"{inst1}-{inst2}"] = float(np.nan_to_num(corr))
                    else:
                        correlations[f"{inst1}-{inst2}"] = 0.0
                else:
                    correlations[f"{inst1}-{inst2}"] = 0.0
                    
    except Exception as e:
        self.logger.warning(f"Failed to calculate correlations: {e}")
        
    return correlations


def _calculate_reward(
    self,
    trades: List[Dict],
    actions: np.ndarray,
    consensus: float,
    info_bus: Optional[InfoBus] = None
) -> float:
    """Enhanced reward calculation with InfoBus integration"""
    
    try:
        # Use InfoBus-enhanced reward calculation if available
        if info_bus and hasattr(self.reward_shaper, 'shape_reward_from_info_bus'):
            reward = self.reward_shaper.shape_reward_from_info_bus(
                info_bus=info_bus,
                trades=trades,
                balance=self.market_state.balance,
                drawdown=self.market_state.current_drawdown,
                consensus=consensus,
                actions=actions
            )
        else:
            # Legacy reward calculation
            reward = self.reward_shaper.shape_reward(
                trades=trades,
                balance=self.market_state.balance,
                drawdown=self.market_state.current_drawdown,
                consensus=consensus,
                actions=actions,
            )
        
        # Store for next step
        self._last_reward = reward
        
        # Log significant rewards
        if abs(reward) > 1.0:
            self.logger.debug(
                format_operator_message(
                    "🎯", "SIGNIFICANT_REWARD",
                    details=f"Reward: {reward:+.3f}",
                    result=f"Trades: {len(trades)}, Consensus: {consensus:.3f}",
                    context="reward_calculation"
                )
            )
        
        return reward
        
    except Exception as e:
        self.logger.error(f"Reward calculation failed: {e}")
        return 0.0


def _check_termination(self) -> bool:
    """Enhanced termination check with InfoBus logging"""
    
    termination_reasons = []
    
    # Max steps reached
    if self.market_state.current_step >= self.config.max_steps - 1:
        termination_reasons.append("max_steps")
        
    # Data exhausted
    for inst in self.instruments:
        if self.market_state.current_step >= len(self.data[inst]["D1"]) - 1:
            termination_reasons.append("data_exhausted")
            break
            
    # Catastrophic loss
    loss_threshold = self.config.initial_balance * 0.5
    if self.market_state.balance < loss_threshold:
        termination_reasons.append("catastrophic_loss")
        
    # Extreme drawdown
    if self.market_state.current_drawdown > 0.5:
        termination_reasons.append("extreme_drawdown")
    
    # Log termination reason
    if termination_reasons:
        primary_reason = termination_reasons[0]
        self.logger.warning(
            format_operator_message(
                "🛑", "EPISODE_TERMINATED",
                details=f"Reason: {primary_reason}",
                result=f"Step: {self.market_state.current_step}, Balance: ${self.market_state.balance:.2f}",
                context="episode_management"
            )
        )
        return True
        
    return False


def _finalize_step(
    self,
    trades: List[Dict],
    actions: np.ndarray,
    consensus: float,
    info_bus: Optional[InfoBus] = None
) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """Enhanced step finalization with comprehensive InfoBus integration"""
    
    # 1) Store trades and calculate PnL
    self.trades = trades
    self.episode_metrics.trades.extend(trades)
    step_pnl = sum(t.get("pnl", 0.0) for t in trades)
    self.episode_metrics.pnls.append(step_pnl)

    # 2) Update regime performance matrix with InfoBus
    try:
        vol = self._get_current_volatility()
        if hasattr(self.fractal_confirm, '_step_impl') and info_bus:
            self.fractal_confirm._step_impl(info_bus=info_bus)
            reg_label = info_bus.get('regime', 'unknown')
        else:
            reg_label, _ = self.fractal_confirm.step(
                data_dict=self.data,
                current_step=self.market_state.current_step,
                theme_detector=self.theme_detector
            )
        
        regime_map = {"noise": 0, "volatile": 1, "trending": 2, "unknown": 0}
        pred_idx = regime_map.get(reg_label, 0)
        
        if hasattr(self.regime_matrix, '_step_impl') and info_bus:
            info_bus['regime_prediction'] = pred_idx
            info_bus['step_pnl'] = step_pnl
            info_bus['volatility'] = vol
            self.regime_matrix._step_impl(info_bus=info_bus)
        else:
            self.regime_matrix.step(
                pnl=step_pnl,
                volatility=vol,
                predicted_regime=pred_idx
            )
    except Exception as e:
        self.logger.error(f"Error updating regime matrix: {e}")

    # 3) Calculate enhanced reward with InfoBus
    reward = self._calculate_reward(trades, actions, consensus, info_bus)

    # 4) Build next observation with InfoBus
    if info_bus is None:
        info_bus = create_info_bus(self, step=self.market_state.current_step)
        info_bus['recent_trades'] = trades
        info_bus['raw_actions'] = actions
        info_bus['consensus'] = consensus
    
    obs = self._get_next_observation(trades, actions)

    # 5) Feed memory modules with InfoBus
    self._feed_memory_modules(info_bus)

    # 6) Check termination
    terminated = self._check_termination()

    # 7) Pack enhanced info dict
    info = self._create_step_info(trades, step_pnl, consensus, info_bus)

    # 8) Update trading mode
    self._update_mode_manager(trades, step_pnl, consensus)

    # 9) Advance step counter
    self.market_state.current_step += 1
    self.current_step = self.market_state.current_step

    # 10) Handle end of episode with enhanced memory updates
    if terminated:
        self._handle_episode_end(step_pnl, info_bus)
        
        # Enhanced memory updates with InfoBus
        self._update_memory_compressor(self.episode_metrics.trades)
        self._record_episode_in_replay_analyzer()
        
        # Update memory budget optimizer
        try:
            if hasattr(self.memory_budget, '_step_impl') and info_bus:
                info_bus['episode_end'] = True
                info_bus['episode_number'] = self.episode_count
                self.memory_budget._step_impl(info_bus=info_bus)
            else:
                self.memory_budget.optimize_allocation(self.episode_count)
        except Exception as e:
            self.logger.error(f"Error optimizing memory allocation: {e}")

    return obs, float(reward), terminated, False, info


def _create_reset_info(self) -> Dict[str, Any]:
    """Enhanced reset info with InfoBus data"""
    
    info = {
        "episode": self.episode_count,
        "balance": self.market_state.balance,
        "genome": self.current_genome,
        "start_step": self.market_state.current_step,
        "info_bus_enabled": self.config.info_bus_enabled,
    }
    
    # Add module status
    if hasattr(self, 'pipeline'):
        info["module_count"] = len(self.pipeline.modules)
        info["expected_obs_size"] = getattr(self.pipeline, 'expected_size', None)
    
    return info


def _create_step_info(self, trades: List[Dict], step_pnl: float, consensus: float, info_bus: InfoBus) -> Dict[str, Any]:
    """Create step info with robust error handling"""
    try:
        # Validate InfoBus safely - FIXED VERSION
        from modules.utils.info_bus import safe_quality_check
        quality_metrics = safe_quality_check(info_bus)
        
        # Create safe step info
        step_info = {
            "step": self.market_state.current_step,
            "balance": float(self.market_state.balance),
            "trades": len(trades),
            "pnl": float(step_pnl),
            "consensus": float(consensus),
            "regime": info_bus.get('market_context', {}).get('regime', 'unknown'),
            "volatility": info_bus.get('market_context', {}).get('volatility', {}),
            "info_bus": quality_metrics,  # Use the safe quality check
            "timestamp": datetime.now().isoformat()
        }
        
        # Add position info safely
        positions = info_bus.get('positions', [])
        step_info["positions"] = len(positions)
        
        # Add alert info safely  
        alerts = info_bus.get('alerts', [])
        step_info["alerts"] = len(alerts)
        
        return step_info
        
    except Exception as e:
        self.logger.error(f"Failed to create step info: {e}")
        # Return minimal safe info
        return {
            "step": getattr(self.market_state, 'current_step', 0),
            "balance": getattr(self.market_state, 'balance', 3000.0),
            "trades": 0,
            "pnl": 0.0,
            "consensus": 0.0,
            "regime": "unknown",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def _update_mode_manager(
    self,
    trades: List[Dict],
    pnl: float,
    consensus: float
):
    """Enhanced mode manager update with InfoBus integration"""
    
    try:
        if hasattr(self.mode_manager, '_step_impl'):
            # Create InfoBus for mode manager if needed
            info_bus = create_info_bus(self, step=self.market_state.current_step)
            info_bus.update({
                'recent_trades': trades,
                'step_pnl': pnl,
                'consensus': consensus,
                'current_drawdown': self.market_state.current_drawdown,
                'trade_count': len(trades)
            })
            self.mode_manager._step_impl(info_bus=info_bus)
        else:
            # Legacy update
            self.mode_manager.update(
                pnl=pnl,
                drawdown=self.market_state.current_drawdown,
                consensus=consensus,
                trade_count=len(trades),
            )
    except Exception as e:
        self.logger.error(f"Mode manager update failed: {e}")


def _handle_episode_end(self, final_pnl: float, info_bus: Optional[InfoBus] = None):
    """Enhanced episode termination handling with InfoBus integration"""
    
    # Calculate episode statistics
    total_pnl = sum(self.episode_metrics.pnls) if self.episode_metrics.pnls else 0.0
    total_trades = len(self.episode_metrics.trades)
    max_dd = max(self.episode_metrics.drawdowns) if self.episode_metrics.drawdowns else 0.0
    
    # Log comprehensive episode summary
    self.logger.info(
        format_operator_message(
            "🏁", "EPISODE_COMPLETED",
            details=f"Episode {self.episode_count}",
            result=f"PnL: {total_pnl:+.2f}, Trades: {total_trades}, MaxDD: {max_dd:.2%}",
            context="episode_management"
        )
    )
    
    # Record audit event
    self.audit_tracker.record_event(
        "episode_completed",
        "Environment",
        {
            "episode": self.episode_count,
            "total_pnl": total_pnl,
            "trade_count": total_trades,
            "max_drawdown": max_dd,
            "final_balance": self.market_state.balance,
            "steps": self.market_state.current_step
        },
        severity="info"
    )
    
    # Update InfoBus with episode summary
    if info_bus:
        InfoBusUpdater.add_module_data(info_bus, 'EpisodeManager', {
            'episode_summary': {
                'episode': self.episode_count,
                'total_pnl': total_pnl,
                'trade_count': total_trades,
                'max_drawdown': max_dd,
                'final_balance': self.market_state.balance
            }
        })
    
    # Save checkpoints periodically
    if self.episode_count % 100 == 0:
        try:
            self._save_checkpoints()
        except Exception as e:
            self.logger.error(f"Failed to save checkpoints: {e}")


def _save_checkpoints(self):
    """Enhanced checkpoint saving with InfoBus state"""
    
    try:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Save enhanced environment state
        state = self.get_state()
        state['info_bus_config'] = self.config.get_info_bus_config()
        state['module_performance'] = getattr(self.pipeline, 'module_performance', {}) if hasattr(self, 'pipeline') else {}
        
        state_path = os.path.join(
            self.config.checkpoint_dir,
            f"enhanced_env_state_ep{self.episode_count}.pkl"
        )
        with open(state_path, "wb") as f:
            pickle.dump(state, f)
            
        # Save module states with enhanced error handling
        modules_to_save = [
            ('PositionManager', self.position_manager),
            ('RiskController', self.risk_controller),
            ('RiskSystem', self.risk_system),
            ('StrategyPool', self.strategy_pool),
            ('MistakeMemory', self.mistake_memory),
            ('MetaRL', self.meta_rl),
            ('Pipeline', self.pipeline),
        ]
        
        saved_count = 0
        failed_saves = []
        
        for name, module in modules_to_save:
            if module and hasattr(module, 'save_checkpoint'):
                try:
                    checkpoint_path = os.path.join(
                        self.config.checkpoint_dir,
                        f"{name}_ep{self.episode_count}.pkl"
                    )
                    module.save_checkpoint(checkpoint_path)
                    saved_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to save {name}: {e}")
                    failed_saves.append(name)
        
        # Log checkpoint summary
        self.logger.info(
            format_operator_message(
                "💾", "CHECKPOINTS_SAVED",
                details=f"Episode {self.episode_count}",
                result=f"Saved: {saved_count}, Failed: {len(failed_saves)}",
                context="checkpoint_management"
            )
        )
        
        if failed_saves:
            self.logger.warning(f"Failed to save: {', '.join(failed_saves)}")
                    
    except Exception as e:
        self.logger.error(
            format_operator_message(
                "💥", "CHECKPOINT_ERROR",
                details=str(e),
                context="checkpoint_management"
            )
        )


def set_module_enabled(self, name: str, enabled: bool):
    """Enhanced module enablement with InfoBus integration"""
    
    if name not in self.module_enabled:
        self.logger.warning(f"Unknown module: {name}")
        return
        
    self.module_enabled[name] = enabled
    
    self.logger.info(
        format_operator_message(
            "🔧", "MODULE_TOGGLED",
            instrument=name,
            details=f"{'Enabled' if enabled else 'Disabled'}",
            context="module_management"
        )
    )
    
    # Record audit event
    self.audit_tracker.record_event(
        "module_toggled",
        "Environment",
        {"module": name, "enabled": enabled},
        severity="info"
    )


def get_state(self) -> Dict[str, Any]:
    """Enhanced state serialization with InfoBus data"""
    
    state = {
        "market_state": {
            "balance": self.market_state.balance,
            "peak_balance": self.market_state.peak_balance,
            "current_step": self.market_state.current_step,
            "current_drawdown": self.market_state.current_drawdown,
            "last_trade_step": self.market_state.last_trade_step,
            "session_pnl": getattr(self.market_state, 'session_pnl', 0.0),
            "session_trades": getattr(self.market_state, 'session_trades', 0),
        },
        "episode_metrics": {
            "pnls": self.episode_metrics.pnls,
            "durations": self.episode_metrics.durations,
            "drawdowns": self.episode_metrics.drawdowns,
            "trades": self.episode_metrics.trades,
            "votes_log": self.episode_metrics.votes_log,
            "reasoning_trace": self.episode_metrics.reasoning_trace,
            "consensus_history": getattr(self.episode_metrics, 'consensus_history', []),
        },
        "episode_count": self.episode_count,
        "position_manager": self.position_manager.get_state() if hasattr(self.position_manager, 'get_state') else {},
        "module_enabled": dict(self.module_enabled),
        "config_summary": {
            "info_bus_enabled": self.config.info_bus_enabled,
            "live_mode": self.config.live_mode,
            "debug": self.config.debug,
        }
    }
    
    # Add InfoBus state if available
    if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'get_performance_summary'):
        state["pipeline_performance"] = self.pipeline.get_performance_summary()
    
    return state


def set_state(self, state: Dict[str, Any]):
    """Enhanced state restoration with InfoBus data"""
    
    # Restore market state
    ms = state.get("market_state", {})
    self.market_state.balance = ms.get("balance", self.config.initial_balance)
    self.market_state.peak_balance = ms.get("peak_balance", self.market_state.balance)
    self.market_state.current_step = ms.get("current_step", 0)
    self.market_state.current_drawdown = ms.get("current_drawdown", 0.0)
    self.market_state.last_trade_step = ms.get("last_trade_step", {})
    
    # Restore episode metrics
    em = state.get("episode_metrics", {})
    self.episode_metrics.pnls = em.get("pnls", [])
    self.episode_metrics.durations = em.get("durations", [])
    self.episode_metrics.drawdowns = em.get("drawdowns", [])
    self.episode_metrics.trades = em.get("trades", [])
    self.episode_metrics.votes_log = em.get("votes_log", [])
    self.episode_metrics.reasoning_trace = em.get("reasoning_trace", [])
    
    # Restore other state
    self.episode_count = state.get("episode_count", 0)
    self.current_step = self.market_state.current_step
    
    # Restore module enablement
    if "module_enabled" in state:
        self.module_enabled.update(state["module_enabled"])
    
    # Restore position manager
    if "position_manager" in state and hasattr(self.position_manager, 'set_state'):
        self.position_manager.set_state(state["position_manager"])
    
    self.logger.info(
        format_operator_message(
            "📥", "STATE_RESTORED",
            details=f"Episode {self.episode_count}, Step {self.current_step}",
            result=f"Balance: ${self.market_state.balance:.2f}",
            context="state_management"
        )
    )


def render(self, mode: str = "human"):
    """Enhanced rendering with InfoBus data"""
    
    if mode == "human":
        # Enhanced text output
        status_emoji = "🔥" if self.market_state.balance > self.market_state.peak_balance * 0.95 else "📉"
        mode_emoji = {"safe": "🛡️", "aggressive": "⚡", "balanced": "⚖️"}.get(self.mode_manager.get_mode(), "🤖")
        
        print(
            f"{status_emoji} Step {self.market_state.current_step} | "
            f"{mode_emoji} Mode: {self.mode_manager.get_mode().upper()} | "
            f"💰 Balance: ${self.market_state.balance:.2f} | "
            f"📉 DD: {self.market_state.current_drawdown:.2%} | "
            f"📊 Trades: {len(self.episode_metrics.trades)} | "
            f"🏛️ Committee: {len(self.committee)} members"
        )
        
        # Show InfoBus status if enabled
        if self.config.info_bus_enabled and hasattr(self, 'pipeline'):
            performance = getattr(self.pipeline, 'module_performance', {})
            active_modules = len([m for m in performance.values() if m.get('call_count', 0) > 0])
            print(f"🚌 InfoBus: {active_modules}/{len(self.pipeline.modules)} modules active")
            
    elif mode == "rgb_array":
        # Could implement visual rendering here
        return None
        
    return None


def close(self):
    """Enhanced cleanup with InfoBus state saving"""
    
    # Save final enhanced checkpoints
    try:
        self._save_checkpoints()
    except Exception as e:
        self.logger.error(f"Failed to save final checkpoints: {e}")
    
    # Record final audit event
    self.audit_tracker.record_event(
        "environment_closed",
        "Environment",
        {
            "episode": self.episode_count,
            "final_balance": self.market_state.balance,
            "total_steps": self.market_state.current_step
        },
        severity="info"
    )
    
    # Close enhanced loggers
    if hasattr(self, 'logger') and hasattr(self.logger, 'logger'):
        for handler in self.logger.logger.handlers:
            handler.close()
    
    # Close InfoBus logger if exists
    if hasattr(self, 'info_bus_logger'):
        for handler in self.info_bus_logger.logger.handlers:
            handler.close()
        
    self.logger.info(
        format_operator_message(
            "👋", "ENVIRONMENT_CLOSED",
            details=f"Episode {self.episode_count} completed",
            result=f"Final balance: ${self.market_state.balance:.2f}",
            context="system_shutdown"
        )
    )