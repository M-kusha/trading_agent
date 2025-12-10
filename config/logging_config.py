from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import LoggingConfig

_configured = False
_active_config: Optional[LoggingConfig] = None


def setup_logging(cfg: Optional[LoggingConfig] = None) -> None:
    """
    Configure global logging using a shared schema.
    Uses a rotating file handler plus console output with a global debug switch.
    """
    global _configured, _active_config

    cfg = cfg or LoggingConfig()
    if _configured and _active_config == cfg:
        return

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / cfg.filename

    level = cfg.effective_level
    handlers: List[str] = ["console"]
    handler_defs: Dict[str, Dict[str, Any]] = {
        "console": {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "standard",
        },
    }

    if cfg.filename:
        handlers.append("file")
        handler_defs["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "detailed",
            "filename": str(log_file),
            "maxBytes": int(cfg.max_bytes),
            "backupCount": int(cfg.backup_count),
            "encoding": "utf-8",
        }

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {"format": "%(asctime)s [%(levelname)s] %(message)s"},
                "detailed": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                },
            },
            "handlers": handler_defs,
            "root": {"level": level, "handlers": handlers},
        }
    )

    _configured = True
    _active_config = cfg


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured via the shared logging setup."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
