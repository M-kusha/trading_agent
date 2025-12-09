# mt5_credentials.py
#!/usr/bin/env python3
import os

class MT5Credentials:
    """
    MetaTrader5 credentials loaded from environment variables,
    with safe fallbacks if you haven’t set them.
    """
    ACCOUNT  = int(os.getenv("MT5_ACCOUNT",  "5043450868"))
    PASSWORD = os.getenv("MT5_PASSWORD", "@7TqLtGb")
    SERVER   = os.getenv("MT5_SERVER",   "MetaQuotes-Demo")

