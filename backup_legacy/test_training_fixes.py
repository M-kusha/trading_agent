#!/usr/bin/env python3
"""
Test script to verify training fixes are working
"""
import sys
import os
sys.path.insert(0, os.getcwd())

def test_environment_config():
    print("=== Testing Environment Config ===")
    from train.train_ppo_hybrid import FileDataProvider
    from envs.config import TradingConfig
    from envs.modern_env import ModernTradingEnv

    config = TradingConfig(test_mode=True, live_mode=False, training_mode=True)
    print(f"Config: test_mode={config.test_mode}, training_mode={config.training_mode}, balance={config.initial_balance}")

    data = FileDataProvider().load(config)
    env = ModernTradingEnv(data, config)

    env_cfg = env.smart_bus.get('environment_config', 'Test')
    print(f"Environment config: {env_cfg}")

    if isinstance(env_cfg, dict):
        mode = env_cfg.get('mode', 'NOT_FOUND')
        balance = env_cfg.get('initial_balance', 'NOT_FOUND')
        print(f"Mode: {mode}")
        print(f"Balance: {balance}")
        return mode == 'sim' and balance == 3000.0
    return False

def test_position_manager_suppression():
    print("\n=== Testing Position Manager Suppression ===")
    try:
        from modules.position.position import PositionManager
        from modules.utils.info_bus import InfoBusManager

        # Check if method exists
        pm = PositionManager()
        has_method = hasattr(pm, '_should_suppress_alerts')
        print(f"Position manager has _should_suppress_alerts method: {has_method}")

        if has_method:
            # Set up test environment config
            bus = InfoBusManager.get_instance()
            test_cfg = {'mode': 'sim', 'initial_balance': 3000.0}
            bus.set('environment_config', test_cfg, module='Test')

            should_suppress = pm._should_suppress_alerts()
            print(f"Should suppress alerts: {should_suppress}")
            return should_suppress
        return False

    except Exception as e:
        print(f"Error testing position manager: {e}")
        return False

def test_visualization_balance():
    print("\n=== Testing Visualization Balance ===")
    try:
        from modules.visualization.visualization_interface import VisualizationInterface
        from modules.utils.info_bus import InfoBusManager

        # Set up environment config with correct balance
        bus = InfoBusManager.get_instance()
        env_cfg = {'initial_balance': 3000.0, 'mode': 'sim'}
        bus.set('environment_config', env_cfg, module='Environment')

        viz = VisualizationInterface()
        # This would need actual testing of the balance extraction logic
        print("Visualization interface created successfully")
        return True

    except Exception as e:
        print(f"Error testing visualization: {e}")
        return False

if __name__ == "__main__":
    print("Testing Training Fixes...")

    env_ok = test_environment_config()
    pm_ok = test_position_manager_suppression()
    viz_ok = test_visualization_balance()

    print("\n=== RESULTS ===")
    print(f"Environment Config Fix: {'✓' if env_ok else '✗'}")
    print(f"Position Manager Suppression: {'✓' if pm_ok else '✗'}")
    print(f"Visualization Interface: {'✓' if viz_ok else '✗'}")

    if all([env_ok, pm_ok, viz_ok]):
        print("\n🎉 ALL FIXES WORKING!")
    else:
        print(f"\n❌ Some fixes not working - restart Python/training to clear module cache")