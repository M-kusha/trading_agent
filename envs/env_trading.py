# envs/env_trading.py
"""
Trading execution methods for the trading environment
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import MetaTrader5 as mt5

from .shared_utils import profile_method


def _validate_actions(self, actions: np.ndarray) -> np.ndarray:
    """Validate and sanitize action array"""
    # Convert to numpy array
    actions = np.asarray(actions, dtype=np.float32)
    
    # Ensure correct shape
    if actions.shape != (self.action_dim,):
        self.logger.warning(
            f"Invalid action shape {actions.shape}, expected {(self.action_dim,)}"
        )
        actions = actions.reshape(-1)[:self.action_dim]
        
    # Clip to valid range
    actions = np.clip(actions, -1.0, 1.0)
    
    # Replace any NaN/Inf
    actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return actions


def _apply_meta_rl(self, actions: np.ndarray) -> np.ndarray:
    """Apply meta-RL modulation to actions"""
    if hasattr(self.meta_rl, 'modulate_action'):
        try:
            modulated = self.meta_rl.modulate_action(actions)
            return np.asarray(modulated, dtype=np.float32)
        except Exception as e:
            self.logger.warning(f"Meta-RL modulation failed: {e}")
    return actions


def _pass_risk_checks(self) -> bool:
    """FIXED: Simplified risk check using centralized manager"""
    context = {
        'drawdown': self.market_state.current_drawdown,
        'correlations': self.get_instrument_correlations(),
        'open_positions': self.position_manager.open_positions,
        'returns': self._get_recent_returns(),
    }
    
    passed, reason = self.risk_manager.pre_trade_check(context)
    if not passed:
        self.logger.info(f"Risk check failed: {reason}")
        
    # Also update legacy risk modules for compatibility
    vol = self._get_current_volatility()
    self.risk_controller.adjust_risk({
        "drawdown": self.market_state.current_drawdown,
        "volatility": vol
    })
    
    return passed


@profile_method
def _get_committee_decision(self, actions: np.ndarray) -> np.ndarray:
    """
    FIXED: Enhanced committee decision with proper data flow to regime modules
    """
    votes_by_sym_tf = {}
    blended_by_sym_tf = {}
    committee_names = [m.__class__.__name__ for m in self.arbiter.members]
    
    # FIXED: Get current market data for regime detection
    current_price = None
    volatility = 0.01
    
    try:
        inst = self.instruments[0]
        df = self.data[inst]["D1"]
        step = self.market_state.current_step
        
        if step < len(df):
            current_price = float(df.iloc[step]["close"])
            
            # Calculate volatility from recent prices
            if step >= 20:
                recent_prices = df["close"].iloc[max(0, step-20):step+1].values
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = float(np.std(returns[np.isfinite(returns)]))
                
    except Exception as e:
        self.logger.warning(f"Failed to extract market data: {e}")
        current_price = 2000.0
        volatility = 0.01
    
    # FIXED: Update regime switcher with current price before committee decision
    try:
        
        # Also update fractal confirmation
        self.fractal_confirm.step(
            data_dict=self.data,
            current_step=self.market_state.current_step,
            theme_detector=self.theme_detector
        )
        
    except Exception as e:
        self.logger.error(f"Failed to update regime modules: {e}")
    
    # Collect votes for each instrument/timeframe combination
    for inst in self.instruments:
        for tf in ["H1", "H4", "D1"]:
            # Get price history
            hist = self._get_price_history(inst, tf)
            
            # Create comprehensive observation for this timeframe
            obs_data = {
                "env": self,
                "price_h1": hist if tf == "H1" else np.zeros_like(hist),
                "price_h4": hist if tf == "H4" else np.zeros_like(hist),
                "price_d1": hist if tf == "D1" else np.zeros_like(hist),
                "actions": actions,
                "current_step": self.market_state.current_step,
                "data_dict": self.data,
                "price": current_price,
                "volatility": volatility,
                "balance": self.market_state.balance,
                "drawdown": self.market_state.current_drawdown,
            }
            obs = self._get_full_observation(obs_data)
            
            # Get arbiter's blended proposal
            blend = self.arbiter.propose(obs)
            
            # Store votes
            if self.arbiter.last_alpha is not None:
                alpha = self.arbiter.last_alpha.copy()
            else:
                alpha = np.zeros(len(self.arbiter.members))
                
            votes_by_sym_tf[(inst, tf)] = dict(zip(committee_names, alpha.tolist()))
            blended_by_sym_tf[(inst, tf)] = blend
            
    # Store votes for logging
    self.episode_metrics.votes_log.append(votes_by_sym_tf)
    
    # Blend across timeframes for final action
    final_action = np.zeros(self.action_dim, dtype=np.float32)
    weights = {"H1": 0.3, "H4": 0.4, "D1": 0.3}
    
    for i, inst in enumerate(self.instruments):
        intensity_sum = 0.0
        duration_sum = 0.0
        total_weight = 0.0
        
        for tf in ["H1", "H4", "D1"]:
            blend = blended_by_sym_tf[(inst, tf)]
            w = weights[tf]
            intensity_sum += blend[2*i] * w
            duration_sum += blend[2*i+1] * w
            total_weight += w
            
        if total_weight > 0:
            final_action[2*i] = intensity_sum / total_weight
            final_action[2*i+1] = duration_sum / total_weight
            
    return final_action


def _calculate_consensus(self) -> float:
    """Calculate consensus level from committee votes"""
    if not hasattr(self.arbiter, 'last_alpha') or self.arbiter.last_alpha is None:
        return 0.5
        
    # Use coefficient of variation as consensus measure
    alpha = self.arbiter.last_alpha
    if alpha.sum() > 0:
        normalized = alpha / (alpha.sum() + 1e-12)
        entropy = -np.sum(normalized * np.log(normalized + 1e-12))
        max_entropy = np.log(len(alpha))
        consensus = 1.0 - (entropy / max_entropy)
    else:
        consensus = 0.0
        
    return float(consensus)


def _pass_consensus_check(self, consensus: float) -> bool:
    """Check if consensus meets threshold"""
    # More lenient check
    if consensus < self.config.consensus_min:
        self.logger.debug(f"Low consensus: {consensus:.3f}")
        return np.random.random() < 0.3  # 30% chance to trade anyway
        
    if consensus > self.config.consensus_max:
        self.logger.debug(f"High consensus: {consensus:.3f}")
        
    return True


def _execute_trades(self, actions: np.ndarray) -> List[Dict[str, Any]]:
    """Execute trades based on actions"""
    trades = []
    
    for i, instrument in enumerate(self.instruments):
        intensity = float(actions[2*i])
        duration_norm = float(actions[2*i+1])
        
        # Skip if intensity too low
        if abs(intensity) < self.config.min_intensity:
            continue
            
        # Check rotation gap
        last_step = self.market_state.last_trade_step.get(instrument, -999)
        if self.market_state.current_step - last_step < self.config.rotation_gap:
            continue
            
        # Check instrument confidence
        inst_conf = self.position_manager.position_confidence.get(instrument, 1.0)
        if inst_conf < self.config.min_inst_confidence:
            self.logger.debug(
                f"Skipping {instrument}: confidence {inst_conf:.3f} < {self.config.min_inst_confidence}"
            )
            continue
            
        # Execute trade
        trade = self._execute_single_trade(instrument, intensity, duration_norm)
        if trade:
            trades.append(trade)
            self.market_state.last_trade_step[instrument] = self.market_state.current_step
            
            # FIXED: Update risk manager after trade
            self.risk_manager.post_trade_update(trade)
            
    return trades


def _execute_single_trade(
    self,
    instrument: str,
    intensity: float,
    duration_norm: float
) -> Optional[Dict[str, Any]]:
    """Execute a single trade (live or simulated)"""
    # Get current market data
    df = self.data[instrument]["D1"]
    if self.market_state.current_step >= len(df):
        return None
        
    bar = df.iloc[self.market_state.current_step]
    price = float(bar["close"])
    
    # Get volatility with safety checks
    vol = self._get_instrument_volatility(instrument)
    
    # Calculate position size
    size = self._calculate_position_size(instrument, intensity, vol)
    
    if size == 0.0:
        self.logger.debug(f"Zero position size for {instrument}")
        return None
        
    # Execute based on mode
    if self.config.live_mode:
        return self._execute_live_trade(instrument, size, intensity)
    else:
        return self._execute_simulated_trade(
            instrument, size, intensity, duration_norm, price, df
        )


def _execute_simulated_trade(
    self,
    instrument: str,
    size: float,
    intensity: float,
    duration_norm: float,
    entry_price: float,
    df
) -> Dict[str, Any]:
    """Execute a simulated trade"""
    side = "BUY" if intensity > 0 else "SELL"
    self.logger.info(
        f"[SIM] {side} {instrument} {size:.3f} lots @ {entry_price:.4f}"
    )
    
    # Calculate holding period
    hold_steps = max(
        int(duration_norm * self.config.max_steps), 
        1
    )
    exit_idx = min(
        self.market_state.current_step + hold_steps,
        len(df) - 1
    )
    
    # Get exit price
    exit_price = float(df.iloc[exit_idx]["close"])
    
    # Calculate PnL
    point_value = self.point_value.get(instrument, 1.0)
    if intensity > 0:
        pnl = (exit_price - entry_price) * size * point_value
    else:
        pnl = (entry_price - exit_price) * size * point_value
        
    # Sanitize PnL
    pnl = np.clip(
        np.nan_to_num(pnl, nan=0.0),
        -10 * self.config.initial_balance,
        10 * self.config.initial_balance
    )
    
    # Update balance
    self.market_state.balance += pnl
    self.market_state.peak_balance = max(
        self.market_state.peak_balance,
        self.market_state.balance
    )
    self.market_state.current_drawdown = max(
        (self.market_state.peak_balance - self.market_state.balance) / 
        (self.market_state.peak_balance + 1e-12),
        0.0
    )
    
    self.logger.info(
        f"[SIM] Trade closed: exit @ {exit_price:.4f}, PnL={pnl:.2f}"
    )
    
    return {
        "instrument": instrument,
        "pnl": pnl,
        "duration": hold_steps,
        "exit_reason": "timeout",
        "size": size if intensity > 0 else -size,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "features": np.array([exit_price, pnl, hold_steps], dtype=np.float32),
    }


def _execute_live_trade(
    self,
    instrument: str,
    size: float,
    intensity: float
) -> Optional[Dict[str, Any]]:
    """FIXED: Execute a live trade via MetaTrader5"""
    symbol = instrument.replace("/", "")
    
    # Select symbol
    if not mt5.symbol_select(symbol, True):
        self.logger.warning(f"Cannot select symbol {symbol}")
        return None
        
    # Get symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        self.logger.warning(f"No symbol info for {symbol}")
        return None
        
    # Round size to broker requirements
    size = self._round_lot_size(size, symbol_info)
    
    # Get current tick
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        self.logger.warning(f"No tick data for {symbol}")
        return None
        
    # Determine price based on side
    price = tick.ask if intensity > 0 else tick.bid
    
    # Create order request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": size,
        "type": mt5.ORDER_TYPE_BUY if intensity > 0 else mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": 20,
        "magic": 202406,
        "comment": f"AI trade ep{self.episode_count}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Send order
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        self.logger.warning(f"Order failed: {result.comment}")
        return None
        
    # Get the actual executed price
    executed_price = result.price if hasattr(result, 'price') else price
    
    # Register position
    self.position_manager.open_positions[instrument] = {
        "ticket": result.order,
        "side": 1 if intensity > 0 else -1,
        "lots": size,
        "price_open": executed_price,
        "peak_profit": 0.0,
        "entry_step": self.market_state.current_step,  # FIXED: Added for monitoring
        "instrument": instrument,  # FIXED: Added for standardization
    }
    
    self.logger.info(
        f"[LIVE] {'BUY' if intensity > 0 else 'SELL'} {instrument} "
        f"{size:.3f} lots @ {executed_price:.4f}, ticket={result.order}"
    )
    
    # FIXED: Return proper trade dictionary
    return {
        "instrument": instrument,
        "size": size,
        "entry_price": executed_price,
        "side": "BUY" if intensity > 0 else "SELL",
        "ticket": result.order,
        "pnl": 0.0,  # Initial PnL is zero
        "duration": 0,
        "exit_reason": "open",
        "features": np.array([executed_price, size, intensity], dtype=np.float32),
    }


def _calculate_position_size(
    self,
    instrument: str,
    intensity: float,
    volatility: float
) -> float:
    """Calculate position size with risk management - FIXED for better trading"""
    # Base size from balance - increased allocation
    risk_pct = 0.02  # Risk 2% per trade
    risk_capital = self.market_state.balance * risk_pct
    
    # Adjust for volatility
    vol_adj = min(0.02 / (volatility + 1e-12), 2.0)
    
    # Scale by intensity with minimum size
    base_size = max(
        risk_capital * abs(intensity) * vol_adj / 100000,
        0.01  # Minimum 0.01 lots
    )
    
    # Apply instrument-specific limits
    if "XAU" in instrument or "GOLD" in instrument:
        base_size = np.clip(base_size, 0.01, 1.0)  # 0.01-1.0 lots for gold
    else:
        base_size = np.clip(base_size, 0.01, 5.0)  # 0.01-5.0 lots for forex
        
    return round(base_size, 2)


def _round_lot_size(self, size: float, symbol_info) -> float:
    """Round lot size to broker requirements"""
    if hasattr(symbol_info, 'volume_step'):
        step = symbol_info.volume_step
        return round(size / step) * step
    return round(size, 2)


def _create_no_trade_step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """Create step output when no trades are executed"""
    return self._finalize_step([], actions, 0.0)


@profile_method
def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """Execute one environment step"""
    try:
        # Validate and sanitize actions
        actions = self._validate_actions(actions)
        
        # Clear previous step data
        self.trades = []
        
        # Update position manager
        self.position_manager.step()
        
        # Apply meta-RL overlay
        actions = self._apply_meta_rl(actions)
        
        # Perform risk checks
        if not self._pass_risk_checks():
            return self._create_no_trade_step(actions)
            
        # Get committee votes and blend actions
        actions = self._get_committee_decision(actions)
        
        # Check consensus threshold
        consensus = self._calculate_consensus()
        if not self._pass_consensus_check(consensus):
            return self._create_no_trade_step(actions)
            
        # Execute trades
        trades = self._execute_trades(actions)
        
        # Update state and calculate reward
        obs, reward, terminated, truncated, info = self._finalize_step(
            trades, actions, consensus
        )
        
        return obs, reward, terminated, truncated, info
        
    except Exception as e:
        self.logger.exception(f"Error in step(): {e}")
        # Return safe default values
        obs = self._get_full_observation(self._create_dummy_input())
        return obs, -1.0, True, False, {"error": str(e)}