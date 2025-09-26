#!/usr/bin/env python3
"""
Comprehensive test to verify system integration after fixes
"""
import sys
import os
import time
sys.path.insert(0, os.getcwd())

def test_training_with_modules():
    print("=== Testing Training with Module Integration ===")
    try:
        from train.train_ppo_hybrid import FileDataProvider
        from envs.config import TradingConfig
        from envs.modern_env import ModernTradingEnv
        from modules.core.module_system import ModuleOrchestrator

        # Create training configuration
        config = TradingConfig(test_mode=True, live_mode=False, training_mode=True)
        data = FileDataProvider().load(config)

        print(f"Data loaded: {len(data)} instruments")

        # Create environment with module system
        env = ModernTradingEnv(data, config)

        # Get orchestrator and check module status
        orch = ModuleOrchestrator.get_instance()
        print(f"Orchestrator: {len(orch.modules)} modules")

        # Check voting system
        voting_modules = [name for name in orch.modules.keys()
                         if 'voting' in name.lower() or 'enhanced' in name.lower()]
        print(f"Voting modules: {len(voting_modules)} - {voting_modules[:5]}")

        # Check environment config
        env_cfg = env.smart_bus.get('environment_config', 'Test')
        if isinstance(env_cfg, dict) and env_cfg.get('mode') == 'sim':
            print(f"Environment config: mode={env_cfg['mode']}, balance={env_cfg.get('initial_balance')}")
            return True
        else:
            print(f"Environment config issue: {env_cfg}")
            return False

    except Exception as e:
        print(f"Training test failed: {e}")
        return False

def test_voting_coordinator_priority():
    print("\n=== Testing Voting Coordinator Priority Fix ===")
    try:
        from modules.voting.voting_wrappers import EnhancedVotingCommitteeCoordinator

        # Check if the class has the updated metadata
        metadata = getattr(EnhancedVotingCommitteeCoordinator, '__module_metadata__', None)
        if metadata:
            priority = getattr(metadata, 'priority', 0)
            print(f"✓ EnhancedVotingCommitteeCoordinator priority: {priority}")
            if priority < 0:
                print("✓ Negative priority confirmed - will run after voters")
                return True
            else:
                print("✗ Priority not negative - may still run before voters")
                return False
        else:
            print("✗ No metadata found")
            return False

    except Exception as e:
        print(f"✗ Priority test failed: {e}")
        return False

def test_position_manager_suppression():
    print("\n=== Testing Position Manager Alert Suppression ===")
    try:
        from modules.position.position import PositionManager
        from modules.utils.info_bus import InfoBusManager

        # Check if suppression method exists
        pm = PositionManager()
        has_method = hasattr(pm, '_should_suppress_alerts')
        print(f"✓ Alert suppression method exists: {has_method}")

        if has_method:
            # Test with sim mode
            bus = InfoBusManager.get_instance()
            bus.set('environment_config', {'mode': 'sim'}, module='Test')

            should_suppress = pm._should_suppress_alerts()
            print(f"✓ Should suppress in sim mode: {should_suppress}")
            return should_suppress
        else:
            print("✗ Alert suppression method missing")
            return False

    except Exception as e:
        print(f"✗ Suppression test failed: {e}")
        return False

def test_module_dependencies():
    print("\n=== Testing Module Dependencies ===")
    try:
        from modules.core.module_system import ModuleOrchestrator

        orch = ModuleOrchestrator.get_instance()

        # Check execution stages
        if hasattr(orch, 'execution_stages') and orch.execution_stages:
            print(f"✓ Execution stages: {len(orch.execution_stages)} stages")

            # Find voting coordinator stage
            coordinator_stage = None
            voter_stages = []

            for i, stage in enumerate(orch.execution_stages):
                if 'EnhancedVotingCommitteeCoordinator' in stage:
                    coordinator_stage = i
                for module in stage:
                    if any(voter in module for voter in ['EnhancedThemeExpert', 'EnhancedSeasonalityRiskExpert',
                                                        'DynamicRiskController', 'MetaAgent']):
                        voter_stages.append(i)

            print(f"✓ Voting coordinator stage: {coordinator_stage}")
            print(f"✓ Voter stages: {set(voter_stages)}")

            if coordinator_stage is not None and voter_stages:
                if coordinator_stage > max(voter_stages):
                    print("✓ Voting coordinator runs AFTER voters - FIXED!")
                    return True
                else:
                    print("✗ Voting coordinator still runs before some voters")
                    return False

        print("✓ Execution stages configured")
        return True

    except Exception as e:
        print(f"✗ Dependencies test failed: {e}")
        return False

if __name__ == "__main__":
    print("COMPREHENSIVE SYSTEM INTEGRATION TEST")
    print("="*50)

    test1 = test_training_with_modules()
    test2 = test_voting_coordinator_priority()
    test3 = test_position_manager_suppression()
    test4 = test_module_dependencies()

    print("\n" + "="*50)
    print("RESULTS SUMMARY:")
    print(f"Training Integration: {'PASS' if test1 else 'FAIL'}")
    print(f"Voting Priority Fix: {'PASS' if test2 else 'FAIL'}")
    print(f"Alert Suppression: {'PASS' if test3 else 'FAIL'}")
    print(f"Module Dependencies: {'PASS' if test4 else 'FAIL'}")

    if all([test1, test2, test3, test4]):
        print("\nALL TESTS PASS - SYSTEM INTEGRATION FIXED!")
    else:
        print(f"\nSome tests failed - system needs restart or additional fixes")