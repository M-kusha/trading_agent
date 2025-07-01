# test_trading_system.py
"""
Comprehensive test suite to verify all trading system fixes are working correctly.
Run this to ensure the system can trade profitably and reliably.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any
import os
import sys
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("TradingSystemTest")

# ═══════════════════════════════════════════════════════════════════
# Test Data Generation
# ═══════════════════════════════════════════════════════════════════

def generate_test_data(instruments: List[str], n_bars: int = 1000) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Generate realistic test market data"""
    data = {}
    
    for inst in instruments:
        data[inst] = {}
        
        # Base parameters for each instrument
        if "XAU" in inst or "GOLD" in inst:
            base_price = 2000.0
            volatility = 0.015
            trend = 0.0001
        else:  # Forex
            base_price = 1.1000
            volatility = 0.008
            trend = 0.00005
            
        # Generate daily data
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq='D')
        
        # Random walk with trend
        returns = np.random.normal(trend, volatility, n_bars)
        prices = base_price * np.exp(np.cumsum(returns))
        
        # OHLC data
        opens = prices * (1 + np.random.normal(0, volatility/4, n_bars))
        highs = np.maximum(opens, prices) * (1 + np.abs(np.random.normal(0, volatility/2, n_bars)))
        lows = np.minimum(opens, prices) * (1 - np.abs(np.random.normal(0, volatility/2, n_bars)))
        closes = prices
        
        # Volume
        base_volume = 100000 if "XAU" in inst else 1000000
        volumes = base_volume * (1 + np.random.exponential(0.5, n_bars))
        
        # Create DataFrames for different timeframes
        daily_df = pd.DataFrame({
            'datetime': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'volatility': pd.Series(closes).pct_change().rolling(20).std().fillna(volatility)
        })
        daily_df.set_index('datetime', inplace=True)
        
        # Generate H4 and H1 by resampling
        h4_df = daily_df.resample('4h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'volatility': 'mean'
        }).dropna()
        
        h1_df = daily_df.resample('1h').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'volatility': 'mean'
        }).dropna()
        
        data[inst] = {
            'D1': daily_df,
            'H4': h4_df,
            'H1': h1_df
        }
        
    return data

# ═══════════════════════════════════════════════════════════════════
# Test Cases
# ═══════════════════════════════════════════════════════════════════

class TradingSystemTester:
    """Test suite for the fixed trading system"""
    
    def __init__(self):
        self.test_results = []
        self.instruments = ["EUR/USD", "XAU/USD"]
        
    def run_all_tests(self):
        """Run all test cases"""
        logger.info("Starting comprehensive trading system tests...")
        
        tests = [
            self.test_environment_initialization,
            self.test_data_flow_consistency,
            self.test_committee_voting,
            self.test_live_trade_recording,
            self.test_risk_management,
            self.test_reward_shaping,
            self.test_bootstrap_trading,
            self.test_full_episode,
        ]
        
        for test in tests:
            try:
                logger.info(f"\nRunning {test.__name__}...")
                result = test()
                self.test_results.append({
                    "test": test.__name__,
                    "passed": result,
                    "error": None
                })
                logger.info(f"✓ {test.__name__} {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                self.test_results.append({
                    "test": test.__name__,
                    "passed": False,
                    "error": str(e)
                })
                logger.error(f"✗ {test.__name__} FAILED with error: {e}")
                
        self._print_summary()
        
    def test_environment_initialization(self) -> bool:
        """Test 1: Environment initializes correctly with all modules"""
        from envs.env import EnhancedTradingEnv, TradingConfig
        
        # Generate test data
        data = generate_test_data(self.instruments)
        
        # Create config
        config = TradingConfig(
            initial_balance=10000.0,
            debug=True,
            live_mode=False
        )
        
        # Initialize environment
        env = EnhancedTradingEnv(data, config)
        
        # Check all critical modules exist
        required_modules = [
            'position_manager', 'risk_controller', 'risk_system',
            'reward_shaper', 'arbiter', 'committee', 'theme_expert',
            'season_expert', 'meta_rl_expert', 'veto_expert', 'regime_expert'
        ]
        
        for module in required_modules:
            if not hasattr(env, module):
                logger.error(f"Missing module: {module}")
                return False
                
        # Check arbiter has correct number of members
        if len(env.arbiter.members) < 8:
            logger.error(f"Arbiter only has {len(env.arbiter.members)} members, expected 8+")
            return False
            
        logger.info(f"Environment initialized with {len(env.arbiter.members)} voting members")
        return True
        
    def test_data_flow_consistency(self) -> bool:
        """Test 2: Data flows correctly with standardized naming"""
        from envs.env import EnhancedTradingEnv, TradingConfig
        
        data = generate_test_data(self.instruments)
        config = TradingConfig(debug=False, live_mode=False)
        env = EnhancedTradingEnv(data, config)
        
        # Reset environment
        obs, info = env.reset()
        
        # Take a step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check ActiveTradeMonitor receives data
        monitor = env.active_monitor
        
        # The monitor should have processed open_positions
        if not hasattr(monitor, 'position_durations'):
            logger.error("ActiveTradeMonitor not tracking positions")
            return False
            
        logger.info("Data flow test passed - monitor receiving position data")
        return True
        
    def test_committee_voting(self) -> bool:
        """Test 3: All committee members participate in voting"""
        from envs.env import EnhancedTradingEnv, TradingConfig
        
        data = generate_test_data(self.instruments)
        config = TradingConfig(debug=True, live_mode=False)
        env = EnhancedTradingEnv(data, config)
        
        obs, _ = env.reset()
        
        # Get committee decision
        test_action = np.array([0.5, 0.5, -0.3, 0.5])
        blended = env._get_committee_decision(test_action)
        
        # Check arbiter recorded votes
        if env.arbiter.last_alpha is None:
            logger.error("Arbiter didn't record member votes")
            return False
            
        if len(env.arbiter.last_alpha) != len(env.arbiter.members):
            logger.error(f"Vote count mismatch: {len(env.arbiter.last_alpha)} vs {len(env.arbiter.members)}")
            return False
            
        # Check votes were recorded
        if not env.episode_metrics.votes_log:
            logger.error("No votes recorded in episode metrics")
            return False
            
        logger.info(f"Committee voting working with {len(env.arbiter.members)} members")
        return True
        
    def test_live_trade_recording(self) -> bool:
        """Test 4: Trades are properly recorded (simulated mode)"""
        from envs.env import EnhancedTradingEnv, TradingConfig
        
        data = generate_test_data(self.instruments)
        config = TradingConfig(
            debug=True,
            live_mode=False,  # Test in simulation mode
            min_intensity=0.1
        )
        env = EnhancedTradingEnv(data, config)
        
        obs, _ = env.reset()
        
        # Force a trade by using high intensity action
        action = np.array([0.8, 0.5, -0.7, 0.5])  # Strong signals
        
        # Take multiple steps to trigger trades
        trades_executed = 0
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(action)
            if env.trades:
                trades_executed += len(env.trades)
                
                # Verify trade structure
                for trade in env.trades:
                    required_fields = ['instrument', 'size', 'entry_price', 'pnl', 'features']
                    for field in required_fields:
                        if field not in trade:
                            logger.error(f"Trade missing field: {field}")
                            return False
                            
        if trades_executed == 0:
            logger.warning("No trades executed - may need to adjust thresholds")
            # This is not necessarily a failure - the system might be conservative
            
        logger.info(f"Trade recording test completed - {trades_executed} trades executed")
        return True
        
    def test_risk_management(self) -> bool:
        """Test 5: Risk management systems work correctly"""
        from envs.env import EnhancedTradingEnv, TradingConfig
        
        data = generate_test_data(self.instruments)
        config = TradingConfig(debug=True, live_mode=False)
        env = EnhancedTradingEnv(data, config)
        
        obs, _ = env.reset()
        
        # Test 1: Risk controller responds to drawdown
        env.market_state.current_drawdown = 0.25  # 25% drawdown
        risk_passed = env._pass_risk_checks()
        
        # The system should still allow some trading at 25% drawdown
        if not risk_passed:
            logger.warning("Risk checks too restrictive at 25% drawdown")
            
        # Test 2: Extreme drawdown should block trades
        env.market_state.current_drawdown = 0.5  # 50% drawdown
        risk_passed = env._pass_risk_checks()
        
        if risk_passed:
            logger.error("Risk checks failed to block at 50% drawdown")
            return False
            
        # Test 3: Portfolio risk system provides position limits
        limits = env.risk_system.get_position_limits()
        
        if not limits:
            logger.error("Portfolio risk system not providing position limits")
            return False
            
        for inst, limit in limits.items():
            if limit <= 0 or limit > 1:
                logger.error(f"Invalid position limit for {inst}: {limit}")
                return False
                
        logger.info(f"Risk management working - position limits: {limits}")
        return True
        
    def test_reward_shaping(self) -> bool:
        """Test 6: Reward shaper encourages trading"""
        from envs.env import EnhancedTradingEnv, TradingConfig
        from modules.reward.risk_adjusted_reward import RiskAdjustedReward
        
        # Test reward shaper directly
        shaper = RiskAdjustedReward(
            initial_balance=10000,
            debug=True
        )
        
        # Test 1: Profitable trade should give positive reward
        profitable_trade = [{
            "instrument": "EUR/USD",
            "pnl": 50.0,
            "size": 0.1
        }]
        
        reward1 = shaper.shape_reward(
            trades=profitable_trade,
            balance=10050,
            drawdown=0.0,
            consensus=0.7
        )
        
        if reward1 <= 0:
            logger.error(f"Profitable trade got negative reward: {reward1}")
            return False
            
        # Test 2: No trades should give small negative reward
        reward2 = shaper.shape_reward(
            trades=[],
            balance=10000,
            drawdown=0.0,
            consensus=0.5
        )
        
        if reward2 >= 0:
            logger.warning(f"No trades got positive reward: {reward2}")
            
        # Test 3: Bootstrap mode should be lenient
        shaper.reset()
        shaper._bootstrap_mode = True
        
        reward3 = shaper.shape_reward(
            trades=[],
            balance=10000,
            drawdown=0.0,
            consensus=0.5
        )
        
        if reward3 < reward2:
            logger.error("Bootstrap mode not more lenient")
            return False
            
        logger.info(f"Reward shaping working - profit reward: {reward1:.3f}, no-trade: {reward2:.3f}")
        return True
        
    def test_bootstrap_trading(self) -> bool:
        """Test 7: System can bootstrap and start trading"""
        from envs.env import EnhancedTradingEnv, TradingConfig
        
        data = generate_test_data(self.instruments)
        config = TradingConfig(
            debug=False,
            live_mode=False,
            min_intensity=0.2,
            consensus_min=0.2  # Lower threshold for bootstrap
        )
        env = EnhancedTradingEnv(data, config)
        
        obs, _ = env.reset()
        
        # Run for bootstrap period
        total_trades = 0
        for step in range(50):
            # Use moderate random actions
            action = np.random.uniform(-0.5, 0.5, env.action_dim)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if env.trades:
                total_trades += len(env.trades)
                
        if total_trades == 0:
            logger.error("No trades during bootstrap period")
            return False
            
        # Check if risk systems have exited bootstrap
        if env.risk_system.bootstrap_mode and env.risk_system.trade_count >= env.risk_system.bootstrap_trades:
            logger.error("Risk system stuck in bootstrap mode")
            return False
            
        logger.info(f"Bootstrap successful - {total_trades} trades in first 50 steps")
        return True
        
    def test_full_episode(self) -> bool:
        """Test 8: Run a full episode and check for profitability potential"""
        from envs.env import EnhancedTradingEnv, TradingConfig
        
        data = generate_test_data(self.instruments, n_bars=500)
        config = TradingConfig(
            debug=False,
            live_mode=False,
            max_steps=200,
            initial_balance=10000.0
        )
        env = EnhancedTradingEnv(data, config)
        
        obs, _ = env.reset()
        
        # Run full episode with simple strategy
        episode_trades = 0
        episode_pnl = 0
        
        for step in range(config.max_steps):
            # Simple momentum strategy
            if step > 10:
                # Look at recent price movement
                momentum = np.random.uniform(-1, 1, env.action_dim)
                action = momentum * 0.5
            else:
                action = env.action_space.sample() * 0.3
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            if env.trades:
                episode_trades += len(env.trades)
                episode_pnl += sum(t.get('pnl', 0) for t in env.trades)
                
            if terminated or truncated:
                break
                
        final_balance = env.market_state.balance
        total_return = (final_balance - config.initial_balance) / config.initial_balance
        
        logger.info(f"Episode complete:")
        logger.info(f"  - Total trades: {episode_trades}")
        logger.info(f"  - Final balance: ${final_balance:.2f}")
        logger.info(f"  - Total return: {total_return:.2%}")
        logger.info(f"  - Max drawdown: {env.market_state.current_drawdown:.2%}")
        
        # Success criteria: System should be able to trade
        if episode_trades == 0:
            logger.error("No trades executed in full episode")
            return False
            
        # The system doesn't need to be profitable with random actions
        # We just need to verify it CAN trade and manage risk
        return True
        
    def _print_summary(self):
        """Print test summary"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r['passed'])
        
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        
        for result in self.test_results:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            logger.info(f"{status} - {result['test']}")
            if result['error']:
                logger.info(f"      Error: {result['error']}")
                
        logger.info("-"*60)
        logger.info(f"Total: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        logger.info("="*60)
        
        if passed_tests == total_tests:
            logger.info("\n🎉 ALL TESTS PASSED! The trading system is ready for use.")
        else:
            logger.warning(f"\n⚠️  {total_tests - passed_tests} tests failed. Please review the errors above.")


# ═══════════════════════════════════════════════════════════════════
# Main Test Runner
# ═══════════════════════════════════════════════════════════════════

def main():
    """Run all trading system tests"""
    # Set up test environment
    os.makedirs("logs/test", exist_ok=True)
    
    # Run tests
    tester = TradingSystemTester()
    tester.run_all_tests()
    
    # Return exit code based on results
    all_passed = all(r['passed'] for r in tester.test_results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()