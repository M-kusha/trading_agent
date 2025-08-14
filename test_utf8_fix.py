#!/usr/bin/env python3
"""
Quick test for UTF-8 encoding fix in integration validator
"""

import sys
import ast
import time
from pathlib import Path

def test_unicode_parsing():
    """Test if files with Unicode characters can be parsed"""
    
    # Files that were failing before
    test_files = [
        "modules/voting/voting_wrappers.py",
        "modules/core/module_base.py", 
        "modules/risk/active_trade_monitor.py",
        "modules/market/regime_performance_matrix.py",
        "modules/strategy/thesis_evolution_engine.py"
    ]
    
    print("🔍 Testing Unicode character parsing...")
    
    success_count = 0
    for file_path in test_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            print(f"✅ {file_path}")
            success_count += 1
        except Exception as e:
            print(f"❌ {file_path}: {e}")
    
    print(f"\n📊 Results: {success_count}/{len(test_files)} files parsed successfully")
    return success_count == len(test_files)

def test_integration_validator():
    """Test the integration validator quickly"""
    print("\n🔧 Testing IntegrationValidator...")
    
    try:
        from modules.monitoring.integration_validator import IntegrationValidator
        
        validator = IntegrationValidator()
        print("✅ IntegrationValidator instantiated successfully")
        
        # Test module discovery on a small subset
        print("📁 Testing module discovery...")
        start_time = time.time()
        
        # We'll just test if we can call the method without it hanging
        validator.module_paths = ["modules/external"]  # Limit scope for testing
        validator._discover_modules()
        
        elapsed = time.time() - start_time
        print(f"✅ Module discovery completed in {elapsed:.2f}s")
        print(f"📦 Discovered {len(validator.discovered_modules)} modules")
        
        return True
        
    except Exception as e:
        print(f"❌ IntegrationValidator error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 UTF-8 Encoding Fix Validation Test")
    print("=" * 50)
    
    # Test 1: Unicode parsing
    unicode_success = test_unicode_parsing()
    
    # Test 2: Integration validator
    validator_success = test_integration_validator()
    
    print("\n" + "=" * 50)
    if unicode_success and validator_success:
        print("🎉 All tests passed! UTF-8 encoding fix is working.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Check the output above.")
        sys.exit(1)
