#!/usr/bin/env python3
"""
Test script to verify live trading integration without executing real trades.
Run this before going live to ensure everything is properly connected.
"""

import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("IntegrationTest")

def test_imports():
    """Test all required imports"""
    logger.info("Testing imports...")
    
    try:
        import MetaTrader5 as mt5
        logger.info("✓ MetaTrader5 imported")
    except ImportError:
        logger.error("✗ MetaTrader5 not installed")
        return False
        
    try:
        from stable_baselines3 import PPO, SAC, TD3
        logger.info("✓ stable-baselines3 imported")
    except ImportError:
        logger.error("✗ stable-baselines3 not installed")
        return False
        
    try:
        from live.live_connector import LiveDataConnector
        logger.info("✓ LiveDataConnector imported")
    except ImportError:
        logger.error("✗ LiveDataConnector not found")
        return False
        
    try:
        from envs.env import EnhancedTradingEnv, TradingConfig
        logger.info("✓ EnhancedTradingEnv imported")
    except ImportError:
        logger.error("✗ EnhancedTradingEnv not found")
        return False
        
    return True

def test_mt5_connection():
    """Test MT5 connection without credentials"""
    logger.info("\nTesting MT5 connection...")
    
    try:
        import MetaTrader5 as mt5
        
        # Just test initialization
        if mt5.initialize():
            logger.info("✓ MT5 initialized")
            
            # Check version
            version = mt5.version()
            if version:
                logger.info(f"  MT5 version: {version}")
                
            mt5.shutdown()
            return True
        else:
            logger.warning("✗ MT5 initialization failed - terminal may not be running")
            return False
            
    except Exception as e:
        logger.error(f"✗ MT5 test failed: {e}")
        return False

def test_environment_creation():
    """Test environment creation with dummy data"""
    logger.info("\nTesting environment creation...")
    
    try:
        from envs.env import EnhancedTradingEnv, TradingConfig
        
        # Create dummy data
        instruments = ["EUR/USD", "XAU/USD"]
        timeframes = ["H1", "H4", "D1"]
        n_bars = 100
        
        data = {}
        for inst in instruments:
            data[inst] = {}
            for tf in timeframes:
                # Create dummy OHLCV data
                dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='H')
                dummy_df = pd.DataFrame({
                    'open': np.random.rand(n_bars) * 0.01 + 1.1,
                    'high': np.random.rand(n_bars) * 0.01 + 1.11,
                    'low': np.random.rand(n_bars) * 0.01 + 1.09,
                    'close': np.random.rand(n_bars) * 0.01 + 1.1,
                    'volume': np.random.rand(n_bars) * 1000000,
                    'volatility': np.random.rand(n_bars) * 0.001
                }, index=dates)
                data[inst][tf] = dummy_df
                
        # Create config
        config = TradingConfig(
            initial_balance=10000,
            live_mode=False,  # Test mode
            debug=False
        )
        
        # Create environment
        env = EnhancedTradingEnv(data, config)
        logger.info("✓ Environment created successfully")
        
        # Test committee
        if hasattr(env, 'arbiter'):
            members = len(env.arbiter.members)
            logger.info(f"✓ Committee has {members} members")
            
            # List members
            for i, member in enumerate(env.arbiter.members):
                logger.info(f"  {i+1}. {member.__class__.__name__}")
        else:
            logger.error("✗ No arbiter found in environment")
            return False
            
        # Test observation
        obs, info = env.reset()
        logger.info(f"✓ Observation shape: {obs.shape}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info("✓ Environment step executed")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading"""
    logger.info("\nTesting model loading...")
    
    models_to_test = [
        ("SAC", "models/sac_final_model.zip"),
        ("PPO", "models/ppo_final_model.zip"),
        ("TD3", "models/td3_final_model.zip"),
    ]
    
    import os
    from stable_baselines3 import PPO, SAC, TD3
    
    found_any = False
    for model_name, model_path in models_to_test:
        if os.path.exists(model_path):
            try:
                if model_name == "SAC":
                    model = SAC.load(model_path, device="cpu")
                elif model_name == "PPO":
                    model = PPO.load(model_path, device="cpu")
                elif model_name == "TD3":
                    model = TD3.load(model_path, device="cpu")
                    
                logger.info(f"✓ {model_name} model loaded from {model_path}")
                found_any = True
            except Exception as e:
                logger.error(f"✗ Failed to load {model_name}: {e}")
        else:
            logger.warning(f"  {model_name} model not found at {model_path}")
            
    return found_any

def test_risk_systems():
    """Test risk management systems"""
    logger.info("\nTesting risk systems...")
    
    try:
        from modules.risk.risk_controller import DynamicRiskController
        from modules.risk.portfolio import PortfolioRiskSystem
        from modules.risk.risk_monitor import ActiveTradeMonitor
        
        # Test DynamicRiskController
        risk_controller = DynamicRiskController(debug=False)
        risk_controller.adjust_risk({"volatility": 0.01, "drawdown": 0.1})
        risk_scale = risk_controller.calculate_risk_scale()
        logger.info(f"✓ DynamicRiskController - risk scale: {risk_scale:.3f}")
        
        # Test PortfolioRiskSystem
        portfolio_risk = PortfolioRiskSystem(debug=False)
        limits = portfolio_risk.get_position_limits()
        logger.info(f"✓ PortfolioRiskSystem - position limits: {limits}")
        
        # Test ActiveTradeMonitor
        monitor = ActiveTradeMonitor(debug=False)
        monitor.step(open_positions={}, current_step=100)
        logger.info(f"✓ ActiveTradeMonitor - risk score: {monitor.risk_score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Risk systems test failed: {e}")
        return False

def test_committee_voting():
    """Test committee voting mechanism"""
    logger.info("\nTesting committee voting...")
    
    try:
        from modules.strategy.voting import StrategyArbiter
        from modules.core.core import Module
        
        # Create dummy members
        class DummyMember(Module):
            def __init__(self, name):
                self.name = name
                
            def propose_action(self, obs):
                return np.random.randn(4)
                
            def confidence(self, obs):
                return np.random.rand()
                
        # Create arbiter
        members = [DummyMember(f"Member{i}") for i in range(8)]
        arbiter = StrategyArbiter(
            members=members,
            init_weights=np.ones(8) / 8,
            action_dim=4,
            debug=False
        )
        
        # Test voting
        obs = np.random.randn(100)
        action = arbiter.propose(obs)
        
        logger.info(f"✓ StrategyArbiter created with {len(members)} members")
        logger.info(f"✓ Committee voted - action shape: {action.shape}")
        logger.info(f"✓ Gate statistics: {arbiter.get_diagnostics()['gate_stats']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Committee voting test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    logger.info("="*60)
    logger.info("LIVE TRADING INTEGRATION TEST")
    logger.info("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("MT5 Connection", test_mt5_connection),
        ("Environment Creation", test_environment_creation),
        ("Model Loading", test_model_loading),
        ("Risk Systems", test_risk_systems),
        ("Committee Voting", test_committee_voting),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
            
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
            
    logger.info("-"*60)
    logger.info(f"Total: {passed}/{len(tests)} tests passed")
    logger.info("="*60)
    
    if passed == len(tests):
        logger.info("\n✅ All tests passed! System is ready for live trading.")
        logger.info("\nNext steps:")
        logger.info("1. Ensure MT5 terminal is running")
        logger.info("2. Create live/mt5_credentials.py with your credentials")
        logger.info("3. Run: python run.py --debug")
        return True
    else:
        logger.warning("\n⚠️  Some tests failed. Please fix issues before live trading.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)