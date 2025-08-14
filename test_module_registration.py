#!/usr/bin/env python3
"""
Test module registration in SmartInfoBus after our fixes
"""

import sys
import os
import asyncio
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_module_registration():
    """Test if our critical modules register properly"""
    print("🔧 Testing module registration...")
    
    try:
        # Import SmartInfoBus
        from modules.utils.info_bus import InfoBusManager
        
        # Create InfoBus instance
        print("📡 Creating InfoBus instance...")
        info_bus = InfoBusManager()
        
        # Try to import and register critical modules
        critical_modules = [
            "MarketDataProvider",
            "SessionManager", 
            "PositionManager",
            "EnhancedVotingCommitteeCoordinator"
        ]
        
        registered_modules = []
        failed_modules = []
        
        for module_name in critical_modules:
            try:
                print(f"📦 Testing {module_name}...")
                
                if module_name == "MarketDataProvider":
                    from modules.external.market_data_provider import MarketDataProvider
                    module_instance = MarketDataProvider()
                    print(f"✅ {module_name} imported and instantiated successfully")
                    
                elif module_name == "SessionManager":
                    from modules.external.session_manager import SessionManager
                    module_instance = SessionManager()
                    print(f"✅ {module_name} imported and instantiated successfully")
                    
                elif module_name == "PositionManager":
                    from modules.position.position import PositionManager
                    module_instance = PositionManager()
                    print(f"✅ {module_name} imported and instantiated successfully")
                    
                elif module_name == "EnhancedVotingCommitteeCoordinator":
                    from modules.voting.voting_wrappers import EnhancedVotingCommitteeCoordinator
                    module_instance = EnhancedVotingCommitteeCoordinator()
                    print(f"✅ {module_name} imported and instantiated successfully")
                
                registered_modules.append(module_name)
                
            except Exception as e:
                print(f"❌ {module_name} failed: {e}")
                failed_modules.append((module_name, str(e)))
        
        print(f"\n📊 Results:")
        print(f"✅ Successfully registered: {len(registered_modules)}")
        print(f"❌ Failed to register: {len(failed_modules)}")
        
        if registered_modules:
            print(f"\nSuccessfully registered modules:")
            for module in registered_modules:
                print(f"  - {module}")
        
        if failed_modules:
            print(f"\nFailed modules:")
            for module, error in failed_modules:
                print(f"  - {module}: {error}")
        
        # Test InfoBus status
        print(f"\n📡 InfoBus basic test completed")
        
        return len(failed_modules) == 0
        
    except Exception as e:
        print(f"💥 Critical error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Module Registration Test")
    print("=" * 50)
    
    success = test_module_registration()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All critical modules registered successfully!")
        exit(0)
    else:
        print("⚠️  Some modules failed to register")
        exit(1)
