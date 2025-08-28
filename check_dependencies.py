import sys
print(f"Python path: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import pywt
    print("✓ pywt is available")
except ImportError as e:
    print(f"✗ pywt not available: {e}")

try:
    import sklearn
    print("✓ sklearn is available")
except ImportError as e:
    print(f"✗ sklearn not available: {e}")

try:
    import psutil
    print("✓ psutil is available")
except ImportError as e:
    print(f"✗ psutil not available: {e}")

try:
    import numpy
    print("✓ numpy is available")
except ImportError as e:
    print(f"✗ numpy not available: {e}")
