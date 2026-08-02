#!/usr/bin/env python3

from typing import Any, Dict, Optional

from config import get_config


class MT5Credentials:

    _cfg = get_config(mode="live")

    ACCOUNT: Optional[int] = _cfg.mt5.account
    PASSWORD: Optional[str] = _cfg.mt5.password
    SERVER: str = _cfg.mt5.server
    PATH: Optional[str] = getattr(_cfg.mt5, 'path', None)

    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        result = {"login": cls.ACCOUNT, "password": cls.PASSWORD, "server": cls.SERVER}
        if cls.PATH:
            result["path"] = cls.PATH
        return result
