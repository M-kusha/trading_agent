#!/usr/bin/env python3
"""MT5 credentials sourced from central config + environment overrides."""

from typing import Any, Dict, Optional

from config import get_config


class MT5Credentials:
    """
    MetaTrader5 credentials loaded from config/app_config.yaml (live mode)
    with environment variable overrides (MT5_ACCOUNT/MT5_PASSWORD/MT5_SERVER).
    """

    _cfg = get_config(mode="live")

    ACCOUNT: Optional[int] = _cfg.mt5.account
    PASSWORD: Optional[str] = _cfg.mt5.password
    SERVER: str = _cfg.mt5.server
    PATH: Optional[str] = getattr(_cfg.mt5, 'path', None)  # Path to specific MT5 terminal (e.g., FTMO)

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        result = {"login": cls.ACCOUNT, "password": cls.PASSWORD, "server": cls.SERVER}
        if cls.PATH:
            result["path"] = cls.PATH
        return result
