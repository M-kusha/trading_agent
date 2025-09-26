#!/usr/bin/env python3
"""
Simple test to verify system integration after fixes
"""
import sys
import os
import time
sys.path.insert(0, os.getcwd())

print("SYSTEM INTEGRATION TEST")
print("="*40)

# Test 1: Training with modules
print("\n1. Testing Training Integration...")
try:
    from train.train_ppo_hybrid import FileDataProvider
    from envs.config import TradingConfig
    from envs.modern_env import ModernTradingEnv

    config = TradingConfig(test_mode=True, live_mode=False, training_mode=True)
    data = FileDataProvider().load(config)
    env = ModernTradingEnv(data, config)

    env_cfg = env.smart_bus.get('environment_config', 'Test')
    if isinstance(env_cfg, dict) and env_cfg.get('mode') == 'sim':
        print(f"  PASS: Environment config mode={env_cfg['mode']}, balance={env_cfg.get('initial_balance')}")
        test1 = True
    else:
        print(f"  FAIL: Environment config issue: {env_cfg}")
        test1 = False

except Exception as e:
    print(f"  FAIL: {e}")
    test1 = False

# Test 2: Position manager suppression
print("\n2. Testing Position Manager Alert Suppression...")
try:
    from modules.position.position import PositionManager
    from modules.utils.info_bus import InfoBusManager

    pm = PositionManager()
    has_method = hasattr(pm, '_should_suppress_alerts')

    if has_method:
        bus = InfoBusManager.get_instance()
        bus.set('environment_config', {'mode': 'sim'}, module='Test')
        should_suppress = pm._should_suppress_alerts()
        if should_suppress:
            print(f"  PASS: Alert suppression working in sim mode")
            test2 = True
        else:
            print(f"  FAIL: Alert suppression not working")
            test2 = False
    else:
        print(f"  FAIL: Alert suppression method missing")
        test2 = False

except Exception as e:
    print(f"  FAIL: {e}")
    test2 = False

# Test 3: Module count and system health
print("\n3. Testing Module System...")
try:
    from modules.core.module_system import ModuleOrchestrator

    orch = ModuleOrchestrator.get_instance()
    print(f"  INFO: {len(orch.modules)} modules loaded")

    # Check if any voting modules exist
    voting_modules = [name for name in orch.modules.keys()
                     if any(keyword in name.lower() for keyword in ['voting', 'enhanced', 'committee'])]
    print(f"  INFO: {len(voting_modules)} voting/enhanced modules")

    if len(orch.modules) >= 30:  # Expect reasonable number of modules
        print("  PASS: Module system healthy")
        test3 = True
    else:
        print("  FAIL: Too few modules loaded")
        test3 = False

except Exception as e:
    print(f"  FAIL: {e}")
    test3 = False

# Summary
print("\n" + "="*40)
print("RESULTS:")
print(f"Training Integration:     {'PASS' if test1 else 'FAIL'}")
print(f"Alert Suppression:        {'PASS' if test2 else 'FAIL'}")
print(f"Module System Health:     {'PASS' if test3 else 'FAIL'}")

if all([test1, test2, test3]):
    print("\nOVERALL: ALL TESTS PASS")
    print("System integration fixes are working!")
else:
    print(f"\nOVERALL: SOME TESTS FAILED")
    print("System may need restart or additional fixes")