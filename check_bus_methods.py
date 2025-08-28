#!/usr/bin/env python3
"""
Script to check if SmartInfoBus has the find_circular_dependencies method.
"""

from modules.utils.info_bus import InfoBusManager

def main():
    print("Checking SmartInfoBus methods...")
    
    # Get the SmartInfoBus instance
    smart_bus = InfoBusManager.get_instance()
    
    # Check if find_circular_dependencies method exists
    has_method = hasattr(smart_bus, 'find_circular_dependencies')
    print(f"Has find_circular_dependencies method: {has_method}")
    
    if has_method:
        try:
            # Try to call the method
            result = smart_bus.find_circular_dependencies()
            print(f"Method returned: {result}")
            print(f"Number of circular dependencies: {len(result)}")
        except Exception as e:
            print(f"Error calling method: {e}")
    else:
        print("Method not available on SmartInfoBus instance.")

if __name__ == "__main__":
    main()
