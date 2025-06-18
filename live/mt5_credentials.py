# mt5_credentials.py
#!/usr/bin/env python3
import os

class MT5Credentials:
    """
    MetaTrader5 credentials loaded from environment variables,
    with safe fallbacks if you haven’t set them.
    """
    ACCOUNT  = int(os.getenv("MT5_ACCOUNT",  "10006700145"))
    PASSWORD = os.getenv("MT5_PASSWORD", "NcFl-nE0")
    SERVER   = os.getenv("MT5_SERVER",   "MetaQuotes-Demo")
