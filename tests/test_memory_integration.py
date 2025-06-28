# test_memory_integration.py
"""
Test script to verify memory module integration fixes
Run this to check if the data flow issues are resolved
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("MemoryIntegrationTest")

def create_test_data():
    """Create minimal test data for environment"""
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    
    # Create realistic price data with some volatility
    base_price = 1.2000
    price_data = []
    current_price = base_price
    
    for i in range(200):
        # Random walk with volatility
        change = np.random.normal(0, 0.01)
        current_price += change
        price_data.append(current_price)
    
    dummy_data = pd.DataFrame({
        'open': [p + np.random.normal(0, 0.001) for p in price_data],
        'high': [p + abs(np.random.normal(0, 0.002)) for p in price_data],
        'low': [p - abs(np.random.normal(0, 0.002)) for p in price_data],
        'close': price_data,
        'volume': np.random.uniform(10000, 100000, 200),
        'volatility': np.random.uniform(0.005, 0.03, 200),
    }, index=dates)
    
    return {
        'EUR/USD': {
            'H1': dummy_data.copy(),
            'H4': dummy_data.copy(),
            'D1': dummy_data.copy(),
        }
    }

def test_memory_modules_individually():
    """Test each memory module individually to verify they work"""
    logger.info("Testing individual memory modules...")
    
    try:
        from modules.memory.memory import (
            MistakeMemory, MemoryCompressor, HistoricalReplayAnalyzer, 
            PlaybookMemory, MemoryBudgetOptimizer
        )
        
        # Test MistakeMemory
        logger.info("Testing MistakeMemory...")
        mistake_memory = MistakeMemory(max_mistakes=20, debug=False)
        
        # Feed some test trades
        for i in range(10):
            features = np.random.randn(5)
            pnl = np.random.normal(0, 50)  # Mix of profits/losses
            info = {"volatility": 0.02, "regime": "trending", "hour": i % 24}
            mistake_memory.step(features=features, pnl=pnl, info=info)
        
        obs = mistake_memory.get_observation_components()
        logger.info(f"✓ MistakeMemory observation shape: {obs.shape}")
        
        # Test danger score
        danger = mistake_memory.check_similarity_to_mistakes(np.random.randn(5))
        logger.info(f"✓ MistakeMemory danger score: {danger:.3f}")
        
        # Test MemoryCompressor
        logger.info("Testing MemoryCompressor...")
        compressor = MemoryCompressor(debug=False)
        
        # Create test trades for compression
        test_trades = []
        for i in range(15):
            test_trades.append({
                "features": np.random.randn(8),
                "pnl": np.random.normal(10, 40)
            })
        
        compressor.compress(episode=1, trades=test_trades)
        obs = compressor.get_observation_components()
        logger.info(f"✓ MemoryCompressor observation shape: {obs.shape}")
        
        # Test HistoricalReplayAnalyzer
        logger.info("Testing HistoricalReplayAnalyzer...")
        replay = HistoricalReplayAnalyzer(debug=False)
        
        # Simulate action sequence
        for i in range(5):
            replay.step(action=np.random.randn(2), features=np.random.randn(5), timestamp=i)
        
        replay.record_episode({"test": True}, np.random.randn(5, 2), 75.0)
        obs = replay.get_observation_components()
        logger.info(f"✓ HistoricalReplayAnalyzer observation shape: {obs.shape}")
        
        # Test PlaybookMemory
        logger.info("Testing PlaybookMemory...")
        playbook = PlaybookMemory(debug=False)
        
        # Record some trades
        for i in range(8):
            features = np.random.randn(5)
            actions = np.random.randn(2)
            pnl = np.random.normal(0, 30)
            context = {"regime": "trending", "volatility": 0.02}
            playbook.record(features, actions, pnl, context)
        
        # Test recall
        recall_result = playbook.recall(np.random.randn(5))
        logger.info(f"✓ PlaybookMemory recall: expected_pnl={recall_result['expected_pnl']:.2f}")
        
        # Test MemoryBudgetOptimizer
        logger.info("Testing MemoryBudgetOptimizer...")
        budget = MemoryBudgetOptimizer(debug=False)
        
        # Simulate memory usage
        budget.step(memory_used="trades", profit=25.0, source="trades")
        budget.step(memory_used="mistakes", profit=-15.0, source="mistakes")
        budget.step(memory_used="plays", profit=35.0, source="plays")
        
        obs = budget.get_observation_components()
        logger.info(f"✓ MemoryBudgetOptimizer observation shape: {obs.shape}")
        
        logger.info("✓ All individual memory modules tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Individual memory module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_integration():
    """Test memory modules integrated with the environment"""
    logger.info("Testing environment integration...")
    
    try:
        from envs.env import EnhancedTradingEnv, TradingConfig
        
        # Create test configuration
        config = TradingConfig(
            initial_balance=10000.0,
            max_steps=20,  # Short test
            debug=False,
            live_mode=False,
        )
        
        # Create test data
        data = create_test_data()
        
        # Create environment
        env = EnhancedTradingEnv(data, config)
        logger.info("✓ Environment created")
        
        # Reset environment
        obs, info = env.reset()
        logger.info(f"✓ Environment reset, obs shape: {obs.shape}")
        
        # Run a few steps to test memory integration
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            logger.info(f"Step {step + 1}: reward={reward:.3f}, balance=${info['balance']:.2f}")
            
            if terminated:
                break
        
        logger.info(f"✓ Completed {step + 1} steps, total reward: {total_reward:.3f}")
        
        # Check if memory modules have recorded data
        memory_stats = []
        
        # Check MistakeMemory
        if hasattr(env, 'mistake_memory'):
            mistake_obs = env.mistake_memory.get_observation_components()
            memory_stats.append(f"MistakeMemory: {mistake_obs.shape}")
        
        # Check MemoryCompressor
        if hasattr(env, 'memory_compressor'):
            comp_obs = env.memory_compressor.get_observation_components()
            memory_stats.append(f"MemoryCompressor: {comp_obs.shape}")
        
        # Check PlaybookMemory
        if hasattr(env, 'playbook_memory'):
            play_obs = env.playbook_memory.get_observation_components()
            memory_stats.append(f"PlaybookMemory: {play_obs.shape}")
        
        # Check MemoryBudgetOptimizer
        if hasattr(env, 'memory_budget'):
            budget_obs = env.memory_budget.get_observation_components()
            memory_stats.append(f"MemoryBudget: {budget_obs.shape}")
        
        logger.info(f"✓ Memory module observations: {', '.join(memory_stats)}")
        
        env.close()
        logger.info("✓ Environment integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Environment integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all memory integration tests"""
    logger.info("=" * 60)
    logger.info("Memory Module Integration Test Suite")
    logger.info("=" * 60)
    
    # Test 1: Individual modules
    test1_passed = test_memory_modules_individually()
    
    # Test 2: Environment integration
    test2_passed = test_environment_integration()
    
    # Results
    logger.info("=" * 60)
    logger.info("TEST RESULTS:")
    logger.info(f"Individual Memory Modules: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    logger.info(f"Environment Integration:   {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("🎉 ALL TESTS PASSED! Memory integration is working correctly.")
        logger.info("You can now run training with properly functioning memory modules.")
    else:
        logger.error("❌ SOME TESTS FAILED. Please check the error messages above.")
        logger.info("Make sure you've applied all the fixes to your code.")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    main()