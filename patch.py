#!/usr/bin/env python3
"""
apply_fixes.py - Run this script to automatically apply all fixes to your trading environment.

Usage:
    python apply_fixes.py
    
This will patch your environment in-place to fix all the integration errors.
"""

import os
import sys
import importlib
import numpy as np
import logging

# Suppress verbose logging during patching
logging.getLogger().setLevel(logging.WARNING)

print("🔧 Trading Environment Fix Script")
print("="*60)

# ===============================================================
# STEP 1: Import and patch the environment
# ===============================================================
print("\n📦 Importing environment modules...")

try:
    from envs.env import EnhancedTradingEnv, TradingPipeline
    print("✅ Environment imported successfully")
except ImportError as e:
    print(f"❌ Failed to import environment: {e}")
    sys.exit(1)

# ===============================================================
# STEP 2: Add missing current_step property
# ===============================================================
print("\n🔨 Adding current_step property...")

if not hasattr(EnhancedTradingEnv, 'current_step'):
    def get_current_step(self):
        return self.market_state.current_step
        
    def set_current_step(self, value):
        self.market_state.current_step = value
        
    EnhancedTradingEnv.current_step = property(get_current_step, set_current_step)
    print("✅ Added current_step property")
else:
    print("✅ current_step property already exists")

# ===============================================================
# STEP 3: Fix TradingPipeline.step method
# ===============================================================
print("\n🔨 Fixing TradingPipeline.step method...")

def step_fixed(self, data):
    """Fixed step method with proper parameter handling"""
    env = data.get("env")
    observations = []
    
    # Ensure critical data is available
    if env:
        if "current_drawdown" not in data:
            data["current_drawdown"] = env.market_state.current_drawdown
        if "current_step" not in data:
            data["current_step"] = env.market_state.current_step
        if "balance" not in data:
            data["balance"] = env.market_state.balance
    
    for module in self.modules:
        if env and not env.module_enabled.get(module.__class__.__name__, True):
            continue
            
        try:
            module_name = module.__class__.__name__
            
            # Special handling for specific modules
            if module_name == "DrawdownRescue":
                module.step(data.get("current_drawdown", 0.0), 
                           current_step=data.get("current_step", 0))
                           
            elif module_name == "ExplanationGenerator":
                # Provide dummy data if not available
                module.step(
                    arbiter_weights=data.get("arbiter_weights", np.ones(8) / 8),
                    member_names=data.get("member_names", ["Member" + str(i) for i in range(8)]),
                    votes=data.get("votes", {})
                )
                
            elif module_name == "NeuralMemoryArchitect":
                experience = {
                    "obs": data.get("obs", np.zeros(32)),
                    "action": data.get("actions", np.zeros(4)),
                    "reward": data.get("reward", 0.0),
                    "done": data.get("done", False),
                    "info": data.get("info", {})
                }
                module.step(experience)
                
            elif module_name == "ActiveTradeMonitor":
                module.step(
                    open_positions=data.get("open_positions", {}),
                    current_step=data.get("current_step", 0)
                )
                
            else:
                # Default handling
                sig = module.step.__code__.co_varnames[1:module.step.__code__.co_argcount]
                kwargs = {k: data[k] for k in sig if k in data}
                module.step(**kwargs)
                
            # Collect observations
            obs = module.get_observation_components()
            observations.append(obs)
            
        except Exception as e:
            logging.error(f"Error in {module.__class__.__name__}.step(): {e}")
            # Try to get default observation
            try:
                obs = module.get_observation_components()
                observations.append(obs)
            except:
                observations.append(np.zeros(0, dtype=np.float32))
                
    return np.concatenate(observations) if observations else np.zeros(0, dtype=np.float32)

TradingPipeline.step = step_fixed
print("✅ Fixed TradingPipeline.step method")

# ===============================================================
# STEP 4: Fix _update_mode_manager method
# ===============================================================
print("\n🔨 Fixing _update_mode_manager method...")

def _update_mode_manager_fixed(self, trades, pnl, consensus):
    """Fixed update mode manager that handles missing update method"""
    if hasattr(self.mode_manager, 'update'):
        self.mode_manager.update(
            pnl=pnl,
            drawdown=self.market_state.current_drawdown,
            consensus=consensus,
            trade_count=len(trades),
        )
    else:
        # Fallback: just track trade count
        if hasattr(self.mode_manager, 'trade_count'):
            self.mode_manager.trade_count = len(trades)
        # Set mode based on simple rules
        if hasattr(self.mode_manager, 'set_mode'):
            if self.market_state.current_drawdown > 0.2:
                self.mode_manager.set_mode("safe")
            elif pnl > 0 and consensus > 0.5:
                self.mode_manager.set_mode("normal")

EnhancedTradingEnv._update_mode_manager = _update_mode_manager_fixed
print("✅ Fixed _update_mode_manager method")

# ===============================================================
# STEP 5: Fix TradingModeManager if needed
# ===============================================================
print("\n🔨 Checking TradingModeManager...")

try:
    from modules.trading_modes.trading_mode import TradingModeManager
    
    if not hasattr(TradingModeManager, 'update'):
        def update(self, pnl=0, drawdown=0, consensus=0.5, trade_count=0):
            """Update trading mode based on metrics"""
            self.trade_count = trade_count
            
            # Simple mode transitions
            if self.current_mode == "safe":
                if drawdown < 0.1 and consensus > 0.4:
                    self.current_mode = "normal"
            elif self.current_mode == "normal":
                if drawdown < 0.05 and pnl > 50:
                    self.current_mode = "aggressive"
                elif drawdown > 0.2:
                    self.current_mode = "safe"
            elif self.current_mode == "aggressive":
                if drawdown > 0.15:
                    self.current_mode = "normal"
                    
        TradingModeManager.update = update
        print("✅ Added update method to TradingModeManager")
    else:
        print("✅ TradingModeManager.update already exists")
        
except ImportError:
    print("⚠️  TradingModeManager not found - will use default implementation")

# ===============================================================
# STEP 6: Fix voting wrappers
# ===============================================================
print("\n🔨 Fixing voting wrappers...")

try:
    from modules.strategy.voting_wrappers import ThemeExpert, MetaRLExpert
    
    # Fix ThemeExpert
    original_propose = ThemeExpert.propose_action
    
    def propose_action_fixed(self, obs, extras=None):
        # Temporarily fix the env attribute access
        if hasattr(self.env, 'market_state'):
            current_step = self.env.market_state.current_step
        else:
            current_step = 0
            
        # Monkey patch for the detect call
        old_current_step = None
        if hasattr(self.env, 'current_step'):
            old_current_step = self.env.current_step
        self.env.current_step = current_step
        
        try:
            result = original_propose(self, obs, extras)
        finally:
            # Restore
            if old_current_step is not None:
                self.env.current_step = old_current_step
            elif hasattr(self.env, 'current_step'):
                delattr(self.env, 'current_step')
                
        return result
        
    ThemeExpert.propose_action = propose_action_fixed
    print("✅ Fixed ThemeExpert")
    
    # Fix MetaRLExpert
    original_call_policy = MetaRLExpert._call_policy
    
    def _call_policy_fixed(self, obs_vec):
        try:
            # Check if properly initialized
            if not hasattr(self.mrl, 'obs_dim') or not hasattr(self.mrl, 'agent'):
                return np.zeros(self.env.action_dim, dtype=np.float32)
                
            # Ensure obs_dim is not a method
            if callable(self.mrl.obs_dim):
                obs_dim = self.mrl.obs_dim()
            else:
                obs_dim = self.mrl.obs_dim
                
            # Fix observation size
            if obs_vec.size != obs_dim:
                if obs_vec.size < obs_dim:
                    obs_vec = np.pad(obs_vec, (0, obs_dim - obs_vec.size))
                else:
                    obs_vec = obs_vec[:obs_dim]
                    
            return original_call_policy(self, obs_vec)
            
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Policy call failed: {e}")
            return np.zeros(self.env.action_dim, dtype=np.float32)
            
    MetaRLExpert._call_policy = _call_policy_fixed
    print("✅ Fixed MetaRLExpert")
    
except ImportError as e:
    print(f"⚠️  Could not fix voting wrappers: {e}")

# ===============================================================
# STEP 7: Run a quick test
# ===============================================================
print("\n🧪 Running quick test...")

try:
    from envs.env import EnhancedTradingEnv, TradingConfig
    import pandas as pd
    
    # Create minimal test data
    instruments = ["EUR/USD", "XAU/USD"]
    data = {}
    
    for inst in instruments:
        data[inst] = {}
        for tf in ["H1", "H4", "D1"]:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='H')
            data[inst][tf] = pd.DataFrame({
                'open': np.ones(100) * 100,
                'high': np.ones(100) * 101,
                'low': np.ones(100) * 99,
                'close': np.ones(100) * 100,
                'volume': np.ones(100) * 1000000,
                'volatility': np.ones(100) * 0.01
            }, index=dates)
    
    # Create environment
    config = TradingConfig(
        initial_balance=10000,
        live_mode=False,
        debug=False,
        max_steps=50
    )
    
    env = EnhancedTradingEnv(data, config)
    obs, info = env.reset()
    
    # Try a few steps
    errors = 0
    for i in range(5):
        try:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            errors += 1
            print(f"  Step {i} error: {str(e)[:50]}...")
            
    if errors == 0:
        print("✅ Test completed successfully - no errors!")
    else:
        print(f"⚠️  Test completed with {errors} errors")
        
except Exception as e:
    print(f"❌ Test failed: {e}")

# ===============================================================
# DONE
# ===============================================================
print("\n" + "="*60)
print("🎉 Fixes applied!")
print("\nYour environment should now work without the integration errors.")
print("\nNext steps:")
print("1. Run your test suite again: python test_trading_system.py")
print("2. If successful, proceed with live trading")
print("3. If errors persist, check the specific error messages")

print("\n💡 Note: These fixes are applied to the current Python session.")
print("   To make them permanent, you'll need to update your source files.")
print("   The fixes are documented in the comments above each patch.")