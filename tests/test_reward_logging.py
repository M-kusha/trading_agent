#!/usr/bin/env python3
"""
Quick test to verify reward logging is working
"""

import os
import sys

def test_reward_logging():
    """Test the fixed reward logging"""
    print("🧪 Testing Fixed Reward Logging...")
    
    try:
        # Import the fixed reward class
        from modules.reward.risk_adjusted_reward import RiskAdjustedReward
        
        # Create instance
        reward_shaper = RiskAdjustedReward(
            initial_balance=3000.0,
            debug=True
        )
        
        print("✅ RiskAdjustedReward created successfully")
        print(f"✅ Logger: {reward_shaper.logger.name}")
        print(f"✅ Handlers: {len(reward_shaper.logger.handlers)}")
        
        # Test shape_reward (the method your environment calls)
        print("\n Testing shape_reward method...")
        
        # Test with profitable trades
        test_trades = [
            {"pnl": 15.0},
            {"pnl": -5.0},
            {"pnl": 8.0}
        ]
        
        reward = reward_shaper.shape_reward(
            trades=test_trades,
            balance=3000.0,
            drawdown=0.02,
            consensus=0.7
        )
        print(f"✅ Reward with trades: {reward:.4f}")
        
        # Test without trades
        reward_no_trades = reward_shaper.shape_reward(
            trades=[],
            balance=3000.0,
            drawdown=0.02,
            consensus=0.7
        )
        print(f"✅ Reward without trades: {reward_no_trades:.4f}")
        
        # Check log file
        log_file = "logs/reward/risk_adjusted_reward.log"
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            print(f"✅ Log file exists: {log_file} ({size} bytes)")
            
            if size > 0:
                print("📝 Recent log entries:")
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-3:]:  # Show last 3 lines
                        print(f"    {line.strip()}")
            else:
                print("⚠️  Log file is empty")
                return False
        else:
            print(f"❌ Log file not found: {log_file}")
            return False
        
        # Test debug functionality
        print("\n🔍 Running debug check...")
        reward_shaper.debug_reward_usage()
        
        print("\n🎉 All tests passed! Reward logging is working.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_reward_logging()
    if success:
        print("\n✅ Your reward logging should now work in training!")
        print("💡 The shape_reward method now has proper logging.")
    else:
        print("\n❌ There's still an issue. Check the error above.")
    
    sys.exit(0 if success else 1)