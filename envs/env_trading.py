# envs/env_trading.py
"""
Enhanced trading execution with InfoBus integration
Maintains all existing method signatures while adding InfoBus infrastructure
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import MetaTrader5 as mt5

from modules.utils.info_bus import InfoBus, create_info_bus, extract_standard_context, InfoBusUpdater
from modules.utils.audit_utils import format_operator_message
from .shared_utils import profile_method


def _validate_actions(self, actions: np.ndarray) -> np.ndarray:
    """Enhanced action validation with InfoBus logging"""
    
    original_shape = actions.shape
    
    # Convert to numpy array
    actions = np.asarray(actions, dtype=np.float32)
    
    # Ensure correct shape
    if actions.shape != (self.action_dim,):
        self.logger.warning(
            format_operator_message(
                "⚠️", "ACTION_SHAPE_MISMATCH",
                details=f"Got {original_shape}, expected {(self.action_dim,)}",
                result="Reshaping actions",
                context="action_validation"
            )
        )
        actions = actions.reshape(-1)[:self.action_dim]
        
    # Clip to valid range
    actions = np.clip(actions, -1.0, 1.0)
    
    # Replace any NaN/Inf
    nan_count = np.sum(~np.isfinite(actions))
    if nan_count > 0:
        self.logger.warning(
            format_operator_message(
                "🧹", "ACTION_SANITIZED",
                details=f"Fixed {nan_count} invalid values",
                context="action_validation"
            )
        )
    
    actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return actions


def _apply_meta_rl(self, actions: np.ndarray, info_bus: Optional[InfoBus] = None) -> np.ndarray:
    """Enhanced meta-RL modulation with InfoBus integration"""
    
    if not hasattr(self.meta_rl, 'modulate_action'):
        return actions
        
    try:
        # Use InfoBus context if available
        if info_bus:
            context = extract_standard_context(info_bus)
            
            # Enhanced modulation with context
            if hasattr(self.meta_rl, 'modulate_action_with_context'):
                modulated = self.meta_rl.modulate_action_with_context(actions, context)
            else:
                modulated = self.meta_rl.modulate_action(actions)
        else:
            modulated = self.meta_rl.modulate_action(actions)
            
        modulated_actions = np.asarray(modulated, dtype=np.float32)
        
        # Log significant modulations
        modulation_strength = np.linalg.norm(modulated_actions - actions)
        if modulation_strength > 0.1:
            self.logger.debug(
                format_operator_message(
                    "🧠", "META_RL_MODULATION",
                    details=f"Strength: {modulation_strength:.3f}",
                    context="meta_learning"
                )
            )
            
        return modulated_actions
        
    except Exception as e:
        self.logger.warning(
            format_operator_message(
                "⚠️", "META_RL_ERROR",
                details=f"Modulation failed: {e}",
                result="Using original actions",
                context="meta_learning"
            )
        )
        return actions


def _pass_risk_checks(self, info_bus: Optional[InfoBus] = None) -> bool:
    """Enhanced risk checks with InfoBus integration"""
    
    try:
        # Use InfoBus for comprehensive risk checking
        if info_bus:
            passed, reason = self.risk_manager.pre_trade_check(info_bus)
        else:
            # Legacy risk check
            context = {
                'drawdown': self.market_state.current_drawdown,
                'correlations': self.get_instrument_correlations(),
                'open_positions': self.position_manager.open_positions,
                'returns': self._get_recent_returns(),
            }
            passed, reason = self.risk_manager.pre_trade_check(context)
        
        if not passed:
            self.logger.info(
                format_operator_message(
                    "🚫", "RISK_CHECK_FAILED",
                    details=reason,
                    result="Trade blocked",
                    context="risk_management"
                )
            )
        
        # Update legacy risk modules for compatibility
        vol = self._get_current_volatility()
        if hasattr(self.risk_controller, 'adjust_risk'):
            self.risk_controller.adjust_risk({
                "drawdown": self.market_state.current_drawdown,
                "volatility": vol
            })
        
        return passed
        
    except Exception as e:
        self.logger.error(
            format_operator_message(
                "💥", "RISK_CHECK_ERROR",
                details=f"Risk check failed: {e}",
                result="Blocking trade for safety",
                context="risk_management"
            )
        )
        return False


@profile_method
def _get_committee_decision(self, actions: np.ndarray, info_bus: Optional[InfoBus] = None) -> np.ndarray:
    """Enhanced committee decision with comprehensive InfoBus integration"""
    
    votes_by_sym_tf = {}
    blended_by_sym_tf = {}
    committee_names = [m.__class__.__name__ for m in self.arbiter.members]
    
    # Create or enhance InfoBus
    if info_bus is None:
        info_bus = create_info_bus(self, step=self.market_state.current_step)
    
    # Get current market data for regime detection
    current_price, volatility = self._extract_current_market_data(info_bus)
    
    # Update regime modules with current data
    self._update_regime_modules(info_bus, current_price, volatility)
    
    # Collect votes for each instrument/timeframe combination
    for inst in self.instruments:
        for tf in ["H1", "H4", "D1"]:
            try:
                # Get comprehensive voting context
                voting_context = self._create_voting_context(inst, tf, actions, info_bus)
                
                # Get arbiter's blended proposal with InfoBus
                if hasattr(self.arbiter, 'propose_with_info_bus'):
                    blend = self.arbiter.propose_with_info_bus(info_bus, voting_context)
                else:
                    # Legacy proposal method
                    obs = self._get_legacy_observation_for_voting(voting_context)
                    blend = self.arbiter.propose(obs)
                
                # Store votes
                if self.arbiter.last_alpha is not None:
                    alpha = self.arbiter.last_alpha.copy()
                else:
                    alpha = np.zeros(len(self.arbiter.members))
                    
                votes_by_sym_tf[(inst, tf)] = dict(zip(committee_names, alpha.tolist()))
                blended_by_sym_tf[(inst, tf)] = blend
                
            except Exception as e:
                self.logger.error(
                    format_operator_message(
                        "💥", "VOTING_ERROR",
                        instrument=f"{inst}/{tf}",
                        details=str(e),
                        context="committee_decision"
                    )
                )
                # Use neutral vote as fallback
                votes_by_sym_tf[(inst, tf)] = {name: 0.0 for name in committee_names}
                blended_by_sym_tf[(inst, tf)] = np.zeros(self.action_dim, dtype=np.float32)
    
    # Store votes for logging
    self.episode_metrics.votes_log.append(votes_by_sym_tf)
    
    # Update InfoBus with voting results
    InfoBusUpdater.add_module_data(info_bus, 'Committee', {
        'votes': votes_by_sym_tf,
        'blended_actions': blended_by_sym_tf,
        'committee_size': len(committee_names)
    })
    
    # Blend across timeframes for final action
    final_action = self._blend_timeframe_actions(blended_by_sym_tf)
    
    return final_action


def _extract_current_market_data(self, info_bus: InfoBus) -> Tuple[float, float]:
    """Extract current market data from InfoBus or environment"""
    
    current_price = 2000.0  # Default
    volatility = 0.01       # Default
    
    try:
        # Try InfoBus first
        prices = info_bus.get('prices', {})
        if prices and self.instruments:
            current_price = float(prices.get(self.instruments[0], current_price))
        
        # Try direct data access
        if current_price == 2000.0:  # Still default
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
    
    return current_price, volatility


def _update_regime_modules(self, info_bus: InfoBus, price: float, volatility: float):
    """Update regime detection modules with current market data"""
    
    try:
        # Update fractal confirmation
        if hasattr(self.fractal_confirm, '_step_impl'):
            self.fractal_confirm._step_impl(info_bus=info_bus)
        else:
            # Legacy update
            self.fractal_confirm.step(
                data_dict=self.data,
                current_step=self.market_state.current_step,
                theme_detector=self.theme_detector
            )
        
        # Update theme detector if it supports InfoBus
        if hasattr(self.theme_detector, '_step_impl'):
            self.theme_detector._step_impl(info_bus=info_bus)
            
    except Exception as e:
        self.logger.error(f"Failed to update regime modules: {e}")


def _create_voting_context(self, inst: str, tf: str, actions: np.ndarray, info_bus: InfoBus) -> Dict[str, Any]:
    """Create comprehensive voting context for instrument/timeframe"""
    
    # Get price history for this timeframe
    hist = self._get_price_history(inst, tf)
    
    # Create comprehensive context
    context = {
        "env": self,
        "instrument": inst,
        "timeframe": tf,
        "price_history": hist,
        "actions": actions,
        "current_step": self.market_state.current_step,
        "balance": self.market_state.balance,
        "drawdown": self.market_state.current_drawdown,
        "info_bus": info_bus,
    }
    
    # Add timeframe-specific price data
    context[f"price_{tf.lower()}"] = hist
    
    return context


def _get_legacy_observation_for_voting(self, voting_context: Dict[str, Any]) -> np.ndarray:
    """Create legacy observation for voting compatibility"""
    
    try:
        obs_data = {
            "env": voting_context["env"],
            "price_h1": voting_context.get("price_h1", np.zeros(7, dtype=np.float32)),
            "price_h4": voting_context.get("price_h4", np.zeros(7, dtype=np.float32)),
            "price_d1": voting_context.get("price_d1", np.zeros(7, dtype=np.float32)),
            "actions": voting_context["actions"],
            "current_step": voting_context["current_step"],
            "data_dict": self.data,
            "balance": voting_context["balance"],
            "drawdown": voting_context["drawdown"],
        }
        
        return self._get_full_observation(obs_data)
        
    except Exception as e:
        self.logger.error(f"Legacy observation creation failed: {e}")
        return np.zeros(100, dtype=np.float32)  # Safe fallback


def _blend_timeframe_actions(self, blended_by_sym_tf: Dict[Tuple[str, str], np.ndarray]) -> np.ndarray:
    """Blend actions across timeframes for final decision"""
    
    final_action = np.zeros(self.action_dim, dtype=np.float32)
    weights = {"H1": 0.3, "H4": 0.4, "D1": 0.3}
    
    for i, inst in enumerate(self.instruments):
        intensity_sum = 0.0
        duration_sum = 0.0
        total_weight = 0.0
        
        for tf in ["H1", "H4", "D1"]:
            try:
                blend = blended_by_sym_tf.get((inst, tf), np.zeros(self.action_dim))
                w = weights[tf]
                
                if len(blend) > 2*i+1:
                    intensity_sum += blend[2*i] * w
                    duration_sum += blend[2*i+1] * w
                    total_weight += w
            except Exception as e:
                self.logger.warning(f"Blending error for {inst}/{tf}: {e}")
        
        if total_weight > 0:
            final_action[2*i] = intensity_sum / total_weight
            final_action[2*i+1] = duration_sum / total_weight
            
    return final_action


def _calculate_consensus(self) -> float:
    """Enhanced consensus calculation with InfoBus logging"""
    
    if not hasattr(self.arbiter, 'last_alpha') or self.arbiter.last_alpha is None:
        return 0.5
        
    try:
        # Use coefficient of variation as consensus measure
        alpha = self.arbiter.last_alpha
        if alpha.sum() > 0:
            normalized = alpha / (alpha.sum() + 1e-12)
            entropy = -np.sum(normalized * np.log(normalized + 1e-12))
            max_entropy = np.log(len(alpha))
            consensus = 1.0 - (entropy / max_entropy)
        else:
            consensus = 0.0
            
        # Log very low or high consensus
        if consensus < 0.2 or consensus > 0.9:
            level = "HIGH" if consensus > 0.9 else "LOW"
            self.logger.debug(
                format_operator_message(
                    "📊", f"{level}_CONSENSUS",
                    details=f"Consensus: {consensus:.3f}",
                    context="committee_decision"
                )
            )
            
        return float(consensus)
        
    except Exception as e:
        self.logger.warning(f"Consensus calculation failed: {e}")
        return 0.5


def _pass_consensus_check(self, consensus: float, info_bus: Optional[InfoBus] = None) -> bool:
    """Enhanced consensus check with InfoBus integration"""
    
    try:
        # More lenient check with context awareness
        if consensus < self.config.consensus_min:
            self.logger.debug(f"Low consensus: {consensus:.3f}")
            
            # Context-aware consensus override
            if info_bus:
                context = extract_standard_context(info_bus)
                # Allow lower consensus in volatile markets
                if context.get('volatility_level') == 'high':
                    return np.random.random() < 0.5  # 50% chance in high vol
                # Allow lower consensus when risk is low
                elif context.get('risk_score', 0.5) < 0.3:
                    return np.random.random() < 0.4  # 40% chance in low risk
            
            return np.random.random() < 0.3  # 30% chance normally
            
        if consensus > self.config.consensus_max:
            self.logger.debug(f"High consensus: {consensus:.3f}")
            
        return True
        
    except Exception as e:
        self.logger.warning(f"Consensus check failed: {e}")
        return True  # Default to allowing trade


def _execute_trades(self, actions: np.ndarray, info_bus: Optional[InfoBus] = None) -> List[Dict[str, Any]]:
    """Enhanced trade execution with InfoBus integration"""
    
    trades = []
    
    for i, instrument in enumerate(self.instruments):
        try:
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
                    format_operator_message(
                        "⏭️", "CONFIDENCE_SKIP",
                        instrument=instrument,
                        details=f"Confidence {inst_conf:.3f} < {self.config.min_inst_confidence}",
                        context="trade_filtering"
                    )
                )
                continue
                
            # Execute trade
            trade = self._execute_single_trade(instrument, intensity, duration_norm, info_bus)
            if trade:
                trades.append(trade)
                self.market_state.last_trade_step[instrument] = self.market_state.current_step
                
                # Update risk manager after trade
                if info_bus:
                    self.risk_manager.post_trade_update(info_bus, trade)
                else:
                    self.risk_manager.post_trade_update(trade)
                    
        except Exception as e:
            self.logger.error(
                format_operator_message(
                    "💥", "TRADE_EXECUTION_ERROR",
                    instrument=instrument,
                    details=str(e),
                    context="trade_execution"
                )
            )
    
    return trades


def _execute_single_trade(
    self,
    instrument: str,
    intensity: float,
    duration_norm: float,
    info_bus: Optional[InfoBus] = None
) -> Optional[Dict[str, Any]]:
    """Enhanced single trade execution with InfoBus integration"""
    
    # Get current market data
    df = self.data[instrument]["D1"]
    if self.market_state.current_step >= len(df):
        return None
        
    bar = df.iloc[self.market_state.current_step]
    price = float(bar["close"])
    
    # Get volatility with safety checks
    vol = self._get_instrument_volatility(instrument)
    
    # Calculate position size with InfoBus context
    size = self._calculate_position_size(instrument, intensity, vol, info_bus)
    
    if size == 0.0:
        self.logger.debug(f"Zero position size for {instrument}")
        return None
        
    # Execute based on mode
    if self.config.live_mode:
        return self._execute_live_trade(instrument, size, intensity, info_bus)
    else:
        return self._execute_simulated_trade(
            instrument, size, intensity, duration_norm, price, df, info_bus
        )


def _execute_simulated_trade(
    self,
    instrument: str,
    size: float,
    intensity: float,
    duration_norm: float,
    entry_price: float,
    df,
    info_bus: Optional[InfoBus] = None
) -> Dict[str, Any]:
    """Enhanced simulated trade execution with InfoBus integration"""
    
    side = "BUY" if intensity > 0 else "SELL"
    
    self.logger.info(
        format_operator_message(
            "📈" if intensity > 0 else "📉", "SIMULATED_TRADE",
            instrument=instrument,
            details=f"{side} {size:.3f} lots @ {entry_price:.4f}",
            context="trade_execution"
        )
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
        format_operator_message(
            "✅", "TRADE_CLOSED",
            instrument=instrument,
            details=f"Exit @ {exit_price:.4f}",
            result=f"PnL: {pnl:+.2f}",
            context="trade_execution"
        )
    )
    
    # Create enhanced trade record
    trade = {
        "instrument": instrument,
        "pnl": pnl,
        "duration": hold_steps,
        "exit_reason": "timeout",
        "size": size if intensity > 0 else -size,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "features": np.array([exit_price, pnl, hold_steps], dtype=np.float32),
        "timestamp": self.market_state.current_step,
        "side": side,
    }
    
    # Add context if InfoBus available
    if info_bus:
        context = extract_standard_context(info_bus)
        trade["market_context"] = context
    
    return trade


def _execute_live_trade(
    self,
    instrument: str,
    size: float,
    intensity: float,
    info_bus: Optional[InfoBus] = None
) -> Optional[Dict[str, Any]]:
    """Enhanced live trade execution with InfoBus integration"""
    
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
        self.logger.warning(
            format_operator_message(
                "❌", "ORDER_FAILED",
                instrument=instrument,
                details=result.comment,
                context="live_trading"
            )
        )
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
        "entry_step": self.market_state.current_step,
        "instrument": instrument,
    }
    
    self.logger.info(
        format_operator_message(
            "🔥", "LIVE_TRADE_EXECUTED",
            instrument=instrument,
            details=f"{'BUY' if intensity > 0 else 'SELL'} {size:.3f} @ {executed_price:.4f}",
            result=f"Ticket: {result.order}",
            context="live_trading"
        )
    )
    
    # Create enhanced trade record
    trade = {
        "instrument": instrument,
        "size": size,
        "entry_price": executed_price,
        "side": "BUY" if intensity > 0 else "SELL",
        "ticket": result.order,
        "pnl": 0.0,  # Initial PnL is zero
        "duration": 0,
        "exit_reason": "open",
        "features": np.array([executed_price, size, intensity], dtype=np.float32),
        "timestamp": self.market_state.current_step,
    }
    
    # Add context if InfoBus available
    if info_bus:
        context = extract_standard_context(info_bus)
        trade["market_context"] = context
    
    return trade


def _calculate_position_size(
    self,
    instrument: str,
    intensity: float,
    volatility: float,
    info_bus: Optional[InfoBus] = None
) -> float:
    """Enhanced position sizing with InfoBus context"""
    
    # Base size from balance - enhanced allocation
    risk_pct = 0.02  # Risk 2% per trade
    
    # Adjust risk based on InfoBus context
    if info_bus:
        context = extract_standard_context(info_bus)
        risk_score = context.get('risk_score', 0.5)
        
        # Lower risk in high-risk environments
        if risk_score > 0.7:
            risk_pct *= 0.5
        elif risk_score < 0.3:
            risk_pct *= 1.5
    
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
    """Create step output when no trades are executed with InfoBus integration"""
    return self._finalize_step([], actions, 0.0)


@profile_method
def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """Enhanced environment step with comprehensive InfoBus integration"""
    
    try:
        # Validate and sanitize actions
        actions = self._validate_actions(actions)
        
        # Clear previous step data
        self.trades = []
        
        # Create comprehensive InfoBus for this step
        info_bus = create_info_bus(self, step=self.market_state.current_step)
        info_bus['input_actions'] = actions.copy()
        
        # Update position manager
        self.position_manager.step()
        
        # Apply meta-RL overlay with InfoBus
        actions = self._apply_meta_rl(actions, info_bus)
        info_bus['modulated_actions'] = actions.copy()
        
        # Perform risk checks with InfoBus
        if not self._pass_risk_checks(info_bus):
            return self._create_no_trade_step(actions)
            
        # Get committee votes and blend actions with InfoBus
        actions = self._get_committee_decision(actions, info_bus)
        info_bus['final_actions'] = actions.copy()
        
        # Check consensus threshold
        consensus = self._calculate_consensus()
        if not self._pass_consensus_check(consensus, info_bus):
            return self._create_no_trade_step(actions)
            
        # Execute trades with InfoBus
        trades = self._execute_trades(actions, info_bus)
        
        # Update InfoBus with trade results
        info_bus['executed_trades'] = trades
        info_bus['consensus'] = consensus
        
        # Update state and calculate reward with InfoBus
        obs, reward, terminated, truncated, info = self._finalize_step(
            trades, actions, consensus, info_bus
        )
        
        return obs, reward, terminated, truncated, info
        
    except Exception as e:
        self.logger.exception(
            format_operator_message(
                "💥", "STEP_ERROR",
                details=f"Step execution failed: {e}",
                result="Returning safe defaults",
                context="environment_step"
            )
        )
        
        # Return safe default values
        obs = self._get_fallback_observation()
        return obs, -1.0, True, False, {"error": str(e)}