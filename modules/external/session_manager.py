# ─────────────────────────────────────────────────────────────
# File: modules/external/session_manager.py
# PRODUCTION-READY Session Manager (Pure, No Simulation)
# Tracks session timing & system health without fabricating metrics
# Pylance-clean: typed self.cfg (dataclass), pass dict to BaseModule
# Provides only session/health context (no duplication with risk/data modules)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time
import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Deque
from collections import deque

import numpy as np

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.utils.audit_utils import RotatingLogger


@dataclass
class SessionConfig:
    """Configuration for Session Manager."""
    session_duration: int = 3600        # seconds before suggesting a reset/roll
    performance_window: int = 500       # history length for internal counters
    enable_health_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_error_pinpointing: bool = True


@module(
    name="SessionManager",
    version="2.0.0",
    category="external",
    provides=[
        "session_metrics",
        "session_context",
        "system_performance",
        "system_health",
        "system_alerts",
        "trading_session",
        "session_type",
        "session_canonical",
        "time_of_day",
    ],
    requires=[],               # root provider; does not depend on others
    description="Session timing and health context (no fabricated metrics, no duplication with risk/data modules).",
    thesis_required=False,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    is_voting_member=False,
    explainable=False,
)
class SessionManager(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Responsibilities:
      - Track session lifecycle (start, duration).
      - Expose human & canonical session labels aligned with TimeAwareRiskScaling.
      - Maintain minimal health/performance counters (real counts only).
      - Never generates synthetic PnL, votes, or risk/theme data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = RotatingLogger("SessionManager", log_path="logs/external/session_manager.log")
        self.cfg = SessionConfig(**(config or {}))
        super().__init__(config=asdict(self.cfg))

        # Session state
        self.session_start_ts: float = time.time()
        self.session_id: str = f"session_{int(self.session_start_ts)}"
        self.session_status: str = "active"

        # Performance counters (real)
        self._success: int = 0
        self._fail: int = 0
        self._proc_times: Deque[float] = deque(maxlen=self.cfg.performance_window)

        # Health / alerts
        self.system_alerts = []  # list of {"level","message","timestamp"}
        self._last_health_check: float = time.time()

        # Labels
        self.trading_session: str = "london"
        self.session_type: str = "normal"

    # BaseModule hook
    def _initialize(self) -> None:
        # Nothing heavy: no fabricated metrics
        self._update_session_labels()
        self.logger.info("[OK] SessionManager initialized.")

    # ─────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────
    def _update_session_labels(self) -> None:
        """Human session labels (UI only); canonical label is derived separately."""
        hour = datetime.datetime.utcnow().hour
        if 8 <= hour < 16:
            self.trading_session = "london"
        elif 13 <= hour < 21:
            self.trading_session = "new_york"
        elif 21 <= hour or hour < 6:
            self.trading_session = "sydney"
        else:
            self.trading_session = "tokyo"

        if 9 <= hour < 17:
            self.session_type = "main"
        elif 17 <= hour < 21:
            self.session_type = "overlap"
        else:
            self.session_type = "overnight"

    def _session_canonical(self) -> str:
        h = datetime.datetime.utcnow().hour
        if 0 <= h < 8:
            return "asian"
        if 8 <= h < 16:
            return "european"
        if 16 <= h < 22:
            return "us"
        return "closed"

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    async def calculate_confidence(self, action: Optional[Dict[str, Any]] = None, **inputs) -> float:
        """
        Confidence here reflects recency of health checks and availability of basic counters.
        No fabricated signals are used.
        """
        now = time.time()
        freshness = max(0.0, 1.0 - (now - self._last_health_check) / 60.0)
        counters_ok = 1.0 if (self._success + self._fail) >= 0 else 0.5
        return float(min(1.0, 0.6 * freshness + 0.4 * counters_ok))

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose session maintenance (e.g., roll session after configured duration)."""
        duration = time.time() - self.session_start_ts
        return {
            "update_session": True,
            "reset_session": bool(duration >= self.cfg.session_duration),
            "reason": "roll session after configured duration" if duration >= self.cfg.session_duration else None,
        }

    async def process(self, **inputs) -> Dict[str, Any]:
        """Update session timestamps/labels and return a compact snapshot."""
        t0 = time.time()
        try:
            # Update labels/time
            self._update_session_labels()
            now = time.time()
            self._last_health_check = now

            # Build outputs
            duration = now - self.session_start_ts
            session_metrics = {
                "session_id": self.session_id,
                "duration": float(duration),
                "status": self.session_status,
                "start_time": datetime.datetime.utcfromtimestamp(self.session_start_ts).isoformat(),
            }

            system_performance = {
                "success_count": int(self._success),
                "failure_count": int(self._fail),
                "success_rate": float(self._success / max(1, self._success + self._fail)),
                "avg_processing_time_ms": float(np.mean(self._proc_times)) if self._proc_times else 0.0,
                "last_check": datetime.datetime.utcnow().isoformat(),
            }

            system_health = {
                "status": "healthy" if len(self.system_alerts) == 0 else "degraded",
                "alerts": list(self.system_alerts[-25:]),
            }

            snapshot = {
                "session_metrics": session_metrics,
                "session_context": {
                    "session_canonical": self._session_canonical(),
                    "trading_session": self.trading_session,
                    "session_type": self.session_type,
                },
                "system_performance": system_performance,
                "system_health": system_health,
                "system_alerts": list(self.system_alerts[-25:]),
                "trading_session": self.trading_session,
                "session_type": self.session_type,
                "session_canonical": self._session_canonical(),
                "time_of_day": datetime.datetime.utcnow().strftime("%H:%M:%S"),
            }

            # Bookkeeping (real)
            self._success += 1
            self._proc_times.append((time.time() - t0) * 1000.0)
            return snapshot

        except Exception as e:
            # Record a real failure; do not fabricate
            self._fail += 1
            self.system_alerts.append({
                "level": "error",
                "message": f"process() exception: {str(e)[:200]}",
                "timestamp": time.time(),
            })
            return {
                "session_metrics": {
                    "session_id": self.session_id,
                    "duration": float(time.time() - self.session_start_ts),
                    "status": "error",
                    "start_time": datetime.datetime.utcfromtimestamp(self.session_start_ts).isoformat(),
                },
                "system_performance": {
                    "success_count": int(self._success),
                    "failure_count": int(self._fail),
                    "success_rate": float(self._success / max(1, self._success + self._fail)),
                    "avg_processing_time_ms": float(np.mean(self._proc_times)) if self._proc_times else 0.0,
                    "last_check": datetime.datetime.utcnow().isoformat(),
                },
                "system_health": {
                    "status": "degraded",
                    "alerts": list(self.system_alerts[-25:]),
                },
                "system_alerts": list(self.system_alerts[-25:]),
                "trading_session": self.trading_session,
                "session_type": self.session_type,
                "session_canonical": self._session_canonical(),
                "time_of_day": datetime.datetime.utcnow().strftime("%H:%M:%S"),
            }
