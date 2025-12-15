#!/usr/bin/env python3
"""
Simulation Time Provider - Unified Time Abstraction for Training vs Live
=========================================================================

This module provides a clean abstraction for time-dependent logic that works
correctly in both:
- TRAINING MODE: Uses bar timestamps from historical data (simulation time)
- LIVE MODE: Uses wall-clock time (real time)

Key Problem Solved:
-------------------
Many modules (SeasonalityRiskExpert, timing features, cooldowns) use datetime.now()
which causes incorrect behavior during training:
- Training at night = all signals see "market closed"
- Cooldowns based on wall-clock = never expire during fast simulation
- Session state is wrong because real time != historical bar time

Architecture:
-------------
1. SimulationClock singleton tracks current simulation time
2. Env publishes bar timestamp to bus via "simulation_time" key on each step
3. Modules call get_simulation_time() which reads from bus (training) or wall-clock (live)
4. All session/timing logic uses this abstraction instead of datetime.now()

Usage:
------
    from modules.utils.simulation_time import get_simulation_time, get_session_info

    # Get current simulation time (bar time in training, wall-clock in live)
    current_time = get_simulation_time(smart_bus, module_name="MyModule")

    # Get session info dict (hour, minute, weekday) for timing features
    session_info = get_session_info(smart_bus, module_name="MyModule")

    # Get step-based cooldown tracking
    cooldown_tracker = get_step_cooldown_tracker(smart_bus, module_name="MyModule")
    if cooldown_tracker.can_act(action_key="entry", min_steps=10):
        # Do action
        cooldown_tracker.record_action("entry")

Version: 1.0.0
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Union
import warnings

# Global flag to detect training vs live mode
_SIMULATION_MODE: bool = True
_SIMULATION_MODE_LOCK = threading.Lock()

# Bus key for simulation time
SIMULATION_TIME_KEY = "simulation_time"
SIMULATION_STEP_KEY = "step_idx"


def set_simulation_mode(enabled: bool) -> None:
    """
    Set whether the system is in simulation mode (training) or live mode.
    
    Call this at startup:
    - Training script: set_simulation_mode(True) - uses historical bar timestamps
    - Live trading: set_simulation_mode(False) - uses wall-clock time
    """
    global _SIMULATION_MODE
    with _SIMULATION_MODE_LOCK:
        _SIMULATION_MODE = enabled


def is_simulation_mode() -> bool:
    """Check if the system is in simulation mode."""
    with _SIMULATION_MODE_LOCK:
        return _SIMULATION_MODE


@dataclass
class SimulationTime:
    """
    Container for simulation time state.
    
    Published to bus by the environment on each step.
    Consumed by modules that need time-aware logic.
    """
    # Core timestamp
    timestamp: datetime
    
    # Step counter (for step-based cooldowns)
    step: int = 0
    
    # Pre-computed session info for efficiency
    hour: int = 12
    minute: int = 0
    weekday: int = 2  # 0=Monday, 6=Sunday
    
    # Source indicator
    source: str = "simulation"  # "simulation" | "wall_clock" | "historical"
    
    # Simulation day counter (for per-day trade limits)
    simulation_day: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for bus publishing."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "hour": self.hour,
            "minute": self.minute,
            "weekday": self.weekday,
            "source": self.source,
            "simulation_day": self.simulation_day,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationTime":
        """Create from dictionary (bus retrieval)."""
        ts_str = data.get("timestamp")
        if isinstance(ts_str, str):
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.now(timezone.utc)
        elif isinstance(ts_str, datetime):
            ts = ts_str
        else:
            ts = datetime.now(timezone.utc)
        
        return cls(
            timestamp=ts,
            step=int(data.get("step", 0)),
            hour=int(data.get("hour", ts.hour)),
            minute=int(data.get("minute", ts.minute)),
            weekday=int(data.get("weekday", ts.weekday())),
            source=str(data.get("source", "unknown")),
            simulation_day=int(data.get("simulation_day", 0)),
        )
    
    @classmethod
    def from_datetime(cls, dt: datetime, step: int = 0, simulation_day: int = 0) -> "SimulationTime":
        """Create from a datetime object."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return cls(
            timestamp=dt,
            step=step,
            hour=dt.hour,
            minute=dt.minute,
            weekday=dt.weekday(),
            source="historical",
            simulation_day=simulation_day,
        )
    
    @classmethod
    def now(cls, step: int = 0) -> "SimulationTime":
        """Create from current wall-clock time."""
        dt = datetime.now(timezone.utc)
        return cls(
            timestamp=dt,
            step=step,
            hour=dt.hour,
            minute=dt.minute,
            weekday=dt.weekday(),
            source="wall_clock",
            simulation_day=0,
        )


def get_simulation_time(
    smart_bus: Optional[Any] = None,
    module_name: str = "Unknown",
    fallback_to_wall_clock: bool = True,
) -> datetime:
    """
    Get current simulation time.
    
    In TRAINING mode:
        Returns the timestamp of the current historical bar from the bus.
        This ensures session logic sees the correct historical time.
    
    In LIVE mode:
        Returns current wall-clock UTC time.
    
    Args:
        smart_bus: SmartInfoBus instance (optional, but recommended)
        module_name: Calling module name for bus access logging
        fallback_to_wall_clock: If True, return wall-clock if bus time unavailable
    
    Returns:
        Timezone-aware datetime representing current simulation/real time.
    """
    # In live mode, always use wall-clock
    if not is_simulation_mode():
        return datetime.now(timezone.utc)
    
    # In simulation mode, try to get time from bus
    if smart_bus is not None:
        try:
            sim_time_data = smart_bus.get(SIMULATION_TIME_KEY, module_name, default=None)
            if sim_time_data is not None:
                if isinstance(sim_time_data, dict):
                    sim_time = SimulationTime.from_dict(sim_time_data)
                    return sim_time.timestamp
                elif isinstance(sim_time_data, datetime):
                    return sim_time_data if sim_time_data.tzinfo else sim_time_data.replace(tzinfo=timezone.utc)
                elif isinstance(sim_time_data, str):
                    try:
                        dt = datetime.fromisoformat(sim_time_data.replace("Z", "+00:00"))
                        return dt
                    except ValueError:
                        pass
        except Exception:
            pass
    
    # Fallback to wall-clock if requested
    if fallback_to_wall_clock:
        return datetime.now(timezone.utc)
    
    raise RuntimeError(
        f"Simulation time not available on bus and fallback disabled. "
        f"Ensure environment publishes '{SIMULATION_TIME_KEY}' on each step."
    )


def get_session_info(
    smart_bus: Optional[Any] = None,
    module_name: str = "Unknown",
) -> Dict[str, Any]:
    """
    Get session info dict for timing features.
    
    Returns dict with:
        - hour: int (0-23)
        - minute: int (0-59)
        - weekday: int (0=Monday, 6=Sunday)
        - step: int (current simulation step)
        - simulation_day: int (day counter within simulation)
    
    This is the format expected by compute_timing_features() and similar functions.
    """
    # First try to get full SimulationTime from bus
    if smart_bus is not None:
        try:
            sim_time_data = smart_bus.get(SIMULATION_TIME_KEY, module_name, default=None)
            if isinstance(sim_time_data, dict):
                return {
                    "hour": int(sim_time_data.get("hour", 12)),
                    "minute": int(sim_time_data.get("minute", 0)),
                    "weekday": int(sim_time_data.get("weekday", 2)),
                    "step": int(sim_time_data.get("step", 0)),
                    "simulation_day": int(sim_time_data.get("simulation_day", 0)),
                }
        except Exception:
            pass
    
    # Fall back to computed from simulation time
    dt = get_simulation_time(smart_bus, module_name, fallback_to_wall_clock=True)
    
    # Get step from bus if available
    step = 0
    if smart_bus is not None:
        try:
            step = int(smart_bus.get(SIMULATION_STEP_KEY, module_name, default=0) or 0)
        except Exception:
            pass
    
    return {
        "hour": dt.hour,
        "minute": dt.minute,
        "weekday": dt.weekday(),
        "step": step,
        "simulation_day": 0,
    }


def get_current_step(
    smart_bus: Optional[Any] = None,
    module_name: str = "Unknown",
) -> int:
    """Get current simulation step from bus."""
    if smart_bus is not None:
        try:
            step = smart_bus.get(SIMULATION_STEP_KEY, module_name, default=0)
            if step is not None:
                return int(step)
        except Exception:
            pass
    return 0


@dataclass
class StepCooldownTracker:
    """
    Track cooldowns based on simulation steps rather than wall-clock time.
    
    This ensures cooldowns work correctly during fast training:
    - Wall-clock cooldown of 15 minutes would never expire in fast simulation
    - Step-based cooldown of 60 steps (60 M15 bars = 15 hours) is realistic
    """
    # Last action step for each action type
    _last_action_steps: Dict[str, int] = field(default_factory=dict)
    
    # Reference to bus for step tracking
    _smart_bus: Optional[Any] = field(default=None, repr=False)
    _module_name: str = "CooldownTracker"
    
    def can_act(self, action_key: str, min_steps: int = 1) -> bool:
        """
        Check if enough steps have passed since last action.
        
        Args:
            action_key: Identifier for the action type (e.g., "entry", "loss_recovery")
            min_steps: Minimum steps required between actions
        
        Returns:
            True if action is allowed, False if still in cooldown.
        """
        current_step = get_current_step(self._smart_bus, self._module_name)
        last_step = self._last_action_steps.get(action_key, -min_steps - 1)
        return (current_step - last_step) >= min_steps
    
    def record_action(self, action_key: str) -> None:
        """Record that an action was taken at the current step."""
        current_step = get_current_step(self._smart_bus, self._module_name)
        self._last_action_steps[action_key] = current_step
    
    def steps_since_action(self, action_key: str) -> int:
        """Get number of steps since last action of this type."""
        current_step = get_current_step(self._smart_bus, self._module_name)
        last_step = self._last_action_steps.get(action_key, 0)
        return current_step - last_step
    
    def reset(self) -> None:
        """Reset all cooldown tracking."""
        self._last_action_steps.clear()


def get_step_cooldown_tracker(
    smart_bus: Optional[Any] = None,
    module_name: str = "Unknown",
) -> StepCooldownTracker:
    """
    Create a step-based cooldown tracker.
    
    Use this instead of time-based cooldowns during training.
    """
    return StepCooldownTracker(_smart_bus=smart_bus, _module_name=module_name)


# Global cooldown trackers by module (singleton pattern for training)
_GLOBAL_COOLDOWN_TRACKERS: Dict[str, StepCooldownTracker] = {}
_GLOBAL_COOLDOWN_LOCK = threading.Lock()


def check_cooldown_elapsed(
    smart_bus: Optional[Any],
    module_name: str,
    cooldown_key: str,
    time_based_seconds: float = 60.0,
    step_based_count: int = 4,
    last_time_value: Optional[float] = None,
) -> bool:
    """
    Check if a cooldown has elapsed - uses steps in training, wall-clock in live.
    
    This is a drop-in replacement for time-based cooldown checks like:
        if time.time() - last_action_time > cooldown_seconds:
    
    Args:
        smart_bus: SmartInfoBus instance
        module_name: Calling module name
        cooldown_key: Unique key for this cooldown type (e.g., "circuit_breaker")
        time_based_seconds: Cooldown in seconds (used in live mode)
        step_based_count: Cooldown in steps (used in training mode)
        last_time_value: For live mode, the last action timestamp (time.time() result)
    
    Returns:
        True if cooldown has elapsed, False if still in cooldown.
    
    Usage:
        # Instead of:
        # if time.time() - self.circuit_breaker["last_failure"] > 60:
        
        # Use:
        # if check_cooldown_elapsed(self.smart_bus, "RiskController", "cb_failure", 
        #                           time_based_seconds=60, step_based_count=4,
        #                           last_time_value=self.circuit_breaker.get("last_failure")):
    """
    if is_simulation_mode():
        # TRAINING MODE: Use step-based cooldowns
        with _GLOBAL_COOLDOWN_LOCK:
            if module_name not in _GLOBAL_COOLDOWN_TRACKERS:
                _GLOBAL_COOLDOWN_TRACKERS[module_name] = StepCooldownTracker(
                    _smart_bus=smart_bus, _module_name=module_name
                )
            tracker = _GLOBAL_COOLDOWN_TRACKERS[module_name]
        
        return tracker.can_act(cooldown_key, min_steps=step_based_count)
    else:
        # LIVE MODE: Use time-based cooldowns
        if last_time_value is None:
            return True  # No last action recorded
        
        import time
        return (time.time() - last_time_value) >= time_based_seconds


def record_cooldown_action(
    smart_bus: Optional[Any],
    module_name: str,
    cooldown_key: str,
) -> float:
    """
    Record that a cooldown-triggering action occurred.
    
    Returns:
        Current time value (for storing in module state for live mode).
    
    Usage:
        # Instead of:
        # self.circuit_breaker["last_failure"] = time.time()
        
        # Use:
        # self.circuit_breaker["last_failure"] = record_cooldown_action(
        #     self.smart_bus, "RiskController", "cb_failure"
        # )
    """
    import time
    current_time = time.time()
    
    if is_simulation_mode():
        # TRAINING MODE: Record step-based action
        with _GLOBAL_COOLDOWN_LOCK:
            if module_name not in _GLOBAL_COOLDOWN_TRACKERS:
                _GLOBAL_COOLDOWN_TRACKERS[module_name] = StepCooldownTracker(
                    _smart_bus=smart_bus, _module_name=module_name
                )
            tracker = _GLOBAL_COOLDOWN_TRACKERS[module_name]
        
        tracker.record_action(cooldown_key)
    
    return current_time


def reset_global_cooldown_trackers() -> None:
    """Reset all global cooldown trackers (call at episode start)."""
    with _GLOBAL_COOLDOWN_LOCK:
        for tracker in _GLOBAL_COOLDOWN_TRACKERS.values():
            tracker.reset()
        _GLOBAL_COOLDOWN_TRACKERS.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT INTEGRATION - Functions for ModernTradingEnv to publish simulation time
# ═══════════════════════════════════════════════════════════════════════════════

def publish_simulation_time(
    smart_bus: Any,
    bar_timestamp: datetime,
    current_step: int,
    simulation_day: int = 0,
    module_name: str = "Environment",
) -> None:
    """
    Publish current simulation time to the bus.
    
    Called by ModernTradingEnv on each step to update simulation time.
    
    Args:
        smart_bus: SmartInfoBus instance
        bar_timestamp: Timestamp of the current historical bar
        current_step: Current simulation step
        simulation_day: Day counter within simulation (for per-day limits)
        module_name: Publishing module name
    """
    if bar_timestamp.tzinfo is None:
        bar_timestamp = bar_timestamp.replace(tzinfo=timezone.utc)
    
    sim_time = SimulationTime.from_datetime(bar_timestamp, current_step, simulation_day)
    
    try:
        smart_bus.set(
            SIMULATION_TIME_KEY,
            sim_time.to_dict(),
            module=module_name,
            thesis=f"Simulation time: {bar_timestamp.isoformat()} (step {current_step})",
        )
    except Exception:
        pass  # Silent fail - non-critical


def compute_simulation_day(
    current_timestamp: datetime,
    episode_start_timestamp: Optional[datetime] = None,
    step: int = 0,
) -> int:
    """
    Compute the simulation day number.
    
    Options:
    1. If episode_start_timestamp provided: days since episode start
    2. Otherwise: use step count (assuming M15 bars, ~96 bars per day)
    """
    if episode_start_timestamp is not None:
        delta = current_timestamp - episode_start_timestamp
        return max(0, delta.days)
    
    # Fallback: estimate from step count (M15 = 96 bars per day)
    return step // 96


def extract_bar_timestamp(
    data_dict: Dict[str, Any],
    current_step: int,
    instrument: str,
    timeframe: str = "M15",
) -> Optional[datetime]:
    """
    Extract bar timestamp from historical data.
    
    Looks for timestamp column in the DataFrame for the given instrument/timeframe.
    """
    try:
        inst_data = data_dict.get(instrument, {})
        tf_data = inst_data.get(timeframe)
        
        if tf_data is None:
            return None
        
        # Handle DataFrame
        if hasattr(tf_data, "iloc") and hasattr(tf_data, "columns"):
            # Check for timestamp columns
            ts_cols = ["timestamp", "time", "datetime", "date", "Timestamp", "Time", "Date"]
            for col in ts_cols:
                if col in tf_data.columns:
                    if current_step < len(tf_data):
                        ts_val = tf_data[col].iloc[current_step]
                        if hasattr(ts_val, "to_pydatetime"):
                            return ts_val.to_pydatetime()
                        elif isinstance(ts_val, datetime):
                            return ts_val
                        elif isinstance(ts_val, str):
                            try:
                                return datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
                            except ValueError:
                                pass
            
            # Check index
            if hasattr(tf_data.index, "to_pydatetime") or hasattr(tf_data.index[0], "to_pydatetime"):
                if current_step < len(tf_data):
                    idx_val = tf_data.index[current_step]
                    if hasattr(idx_val, "to_pydatetime"):
                        return idx_val.to_pydatetime()
                    elif isinstance(idx_val, datetime):
                        return idx_val
        
        # Handle dict with arrays
        elif isinstance(tf_data, dict):
            for key in ["timestamp", "time", "datetime"]:
                if key in tf_data:
                    ts_arr = tf_data[key]
                    if current_step < len(ts_arr):
                        ts_val = ts_arr[current_step]
                        if isinstance(ts_val, datetime):
                            return ts_val
                        elif isinstance(ts_val, str):
                            try:
                                return datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
                            except ValueError:
                                pass
    except Exception:
        pass
    
    return None


def estimate_bar_timestamp(
    current_step: int,
    base_timestamp: Optional[datetime] = None,
    timeframe: str = "M15",
) -> datetime:
    """
    Estimate bar timestamp when not available in data.
    
    Uses step count and timeframe to compute approximate historical time.
    Assumes data starts at base_timestamp and progresses forward.
    """
    if base_timestamp is None:
        # Default to a reasonable trading day start (random Wednesday)
        base_timestamp = datetime(2024, 6, 12, 9, 0, 0, tzinfo=timezone.utc)
    
    # Timeframe to minutes mapping
    tf_minutes = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H2": 120, "H4": 240, "H8": 480,
        "D1": 1440, "W1": 10080, "MN1": 43200,
    }
    
    minutes_per_bar = tf_minutes.get(timeframe.upper(), 15)
    total_minutes = current_step * minutes_per_bar
    
    return base_timestamp + timedelta(minutes=total_minutes)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE COMPATIBILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def get_trading_window_time(
    smart_bus: Optional[Any] = None,
    module_name: str = "Unknown",
    local_timezone: str = "Europe/Berlin",
) -> datetime:
    """
    Get simulation time converted to local timezone for trading window checks.
    
    This is a drop-in replacement for datetime.now(local_tz) in session logic.
    """
    utc_time = get_simulation_time(smart_bus, module_name)
    
    try:
        import zoneinfo
        local_tz = zoneinfo.ZoneInfo(local_timezone)
        return utc_time.astimezone(local_tz)
    except ImportError:
        try:
            import pytz
            local_tz = pytz.timezone(local_timezone)
            return utc_time.astimezone(local_tz)
        except ImportError:
            # Fallback: assume Europe/Berlin is UTC+1 or UTC+2
            # This is imprecise but better than nothing
            return utc_time + timedelta(hours=1)


__all__ = [
    "set_simulation_mode",
    "is_simulation_mode",
    "get_simulation_time",
    "get_session_info",
    "get_current_step",
    "get_step_cooldown_tracker",
    "check_cooldown_elapsed",
    "record_cooldown_action",
    "reset_global_cooldown_trackers",
    "SimulationTime",
    "StepCooldownTracker",
    "publish_simulation_time",
    "compute_simulation_day",
    "extract_bar_timestamp",
    "estimate_bar_timestamp",
    "get_trading_window_time",
    "SIMULATION_TIME_KEY",
    "SIMULATION_STEP_KEY",
]
