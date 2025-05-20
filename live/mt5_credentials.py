# mt5_credentials.py
#!/usr/bin/env python3
import os

class MT5Credentials:
    """
    MetaTrader5 credentials loaded from environment variables,
    with safe fallbacks if you haven’t set them.
    """
    ACCOUNT  = int(os.getenv("MT5_ACCOUNT",  "92666977"))
    PASSWORD = os.getenv("MT5_PASSWORD", "@5TkYjBq")
    SERVER   = os.getenv("MT5_SERVER",   "MetaQuotes-Demo")
