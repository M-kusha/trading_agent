"""Unified time source for training and live execution.

THE PROBLEM
-----------
The modules/ subsystems were written as real-time services. Across that tree
there are 401 `datetime.now()` calls, 610 `time.time()` calls, 36 `time.sleep()`
calls and 178 threading references. In a training loop that replays years of
history in minutes this breaks in ways that are silent rather than loud:

  * Session and seasonality logic reads the wall clock, so training at 02:00
    local time makes every module believe the market is closed - regardless of
    the timestamp on the bar being replayed.
  * Cooldowns expressed in seconds never expire, because thousands of simulated
    bars pass inside one real second.
  * Bus staleness checks (`max_data_age_seconds`) compare against time.time(),
    so data is either permanently fresh or permanently stale, never correct.

That is why the training environment ended up with its own parallel, time-pure
reimplementation of the experts in envs/prop_firm/signals/. The workaround was
correct; making it permanent was not.

THE APPROACH
------------
A single process-wide clock with an explicit mode.

  LIVE mode       now() -> wall clock, unchanged behaviour.
  SIMULATION mode now() -> the timestamp of the bar currently being replayed,
                  published by the environment on every step.

Modules do not have to be rewritten to benefit. `install_clock(module)` swaps
the `datetime` class and `time` module inside a module's own namespace, so every
existing `datetime.now()` call inside it resolves to simulation time without
touching a single call site. Editing the 362 inline call sites individually was
never viable; this is one line per module.

Deliberately NOT bus-coupled: the training path does not use SmartInfoBus at
all (0 files under envs/ or train/ import it), so a bus-delivered clock could
never reach the environment. The bus can still mirror the clock for live
consumers via publish_to_bus().

USAGE
-----
    from modules.utils import simulation_time as simclock

    simclock.set_mode(simclock.TimeMode.SIMULATION)   # training entrypoint

    # environment, once per step:
    simclock.set_bar_time(bar_timestamp, step=self.current_step)

    # any module:
    now = simclock.now()
    session = simclock.get_session_info()

    # step-based cooldowns instead of wall-clock ones:
    if simclock.cooldown_elapsed("entry", min_steps=10):
        simclock.record_cooldown_action("entry")
"""

from __future__ import annotations

import datetime as _datetime_module
import threading
import types
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional

__all__ = [
    "SimulationTime",
    "StepCooldownTracker",
    "TimeMode",
    "cooldown_elapsed",
    "get_bar_time",
    "get_current_step",
    "get_mode",
    "get_session_info",
    "install_clock",
    "is_simulation_mode",
    "now",
    "publish_to_bus",
    "record_cooldown_action",
    "reset",
    "set_bar_time",
    "set_mode",
    "simulation_mode",
    "utcnow",
]

SIMULATION_TIME_KEY = "simulation_time"
SIMULATION_STEP_KEY = "step_idx"


class TimeMode(Enum):
    """Where `now()` gets its answer."""

    LIVE = "live"
    SIMULATION = "simulation"


@dataclass
class SimulationTime:
    """Snapshot of simulated time, cheap to copy and to serialise."""

    timestamp: datetime
    step: int = 0
    simulation_day: int = 0

    @property
    def hour(self) -> int:
        return self.timestamp.hour

    @property
    def minute(self) -> int:
        return self.timestamp.minute

    @property
    def weekday(self) -> int:
        """0 = Monday ... 6 = Sunday."""
        return self.timestamp.weekday()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "simulation_day": self.simulation_day,
            "hour": self.hour,
            "minute": self.minute,
            "weekday": self.weekday,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulationTime":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            step=int(data.get("step", 0)),
            simulation_day=int(data.get("simulation_day", 0)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Process-wide clock state
# ─────────────────────────────────────────────────────────────────────────────

_LOCK = threading.RLock()
_MODE: TimeMode = TimeMode.LIVE
_BAR_TIME: Optional[datetime] = None
_STEP: int = 0
_EPOCH_DAY: Optional[_datetime_module.date] = None


def set_mode(mode: TimeMode) -> None:
    """Select the time source. Call once, at the process entrypoint."""
    global _MODE
    with _LOCK:
        _MODE = TimeMode(mode)


def get_mode() -> TimeMode:
    with _LOCK:
        return _MODE


def is_simulation_mode() -> bool:
    return get_mode() is TimeMode.SIMULATION


def reset() -> None:
    """Clear simulated time. Used between episodes and by tests."""
    global _BAR_TIME, _STEP, _EPOCH_DAY
    with _LOCK:
        _BAR_TIME = None
        _STEP = 0
        _EPOCH_DAY = None
    _reset_cooldowns()


def set_bar_time(timestamp: Optional[datetime], step: Optional[int] = None) -> None:
    """Publish the timestamp of the bar currently being replayed.

    The environment calls this once per step. Naive timestamps are treated as
    UTC so that comparisons against timezone-aware values do not raise.
    """
    global _BAR_TIME, _STEP, _EPOCH_DAY
    if timestamp is None:
        return
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    with _LOCK:
        _BAR_TIME = timestamp
        if step is not None:
            _STEP = int(step)
        if _EPOCH_DAY is None:
            _EPOCH_DAY = timestamp.date()


def now(tz: Optional[_datetime_module.tzinfo] = None) -> datetime:
    """Current time: the replayed bar in simulation, the wall clock in live."""
    with _LOCK:
        mode, bar = _MODE, _BAR_TIME

    if mode is TimeMode.SIMULATION and bar is not None:
        return bar.astimezone(tz) if tz is not None else bar.replace(tzinfo=None)
    return datetime.now(tz)


def utcnow() -> datetime:
    """Naive UTC, mirroring the semantics of the deprecated datetime.utcnow()."""
    return now(timezone.utc).replace(tzinfo=None)


def get_bar_time() -> Optional[datetime]:
    with _LOCK:
        return _BAR_TIME


def get_current_step() -> int:
    with _LOCK:
        return _STEP


def get_session_info() -> Dict[str, Any]:
    """Session context derived from the active clock, never from `datetime.now()`."""
    current = now(timezone.utc)
    with _LOCK:
        step, epoch_day = _STEP, _EPOCH_DAY

    simulation_day = 0
    if epoch_day is not None:
        simulation_day = (current.date() - epoch_day).days

    return {
        "timestamp": current,
        "hour": current.hour,
        "minute": current.minute,
        "weekday": current.weekday(),
        "is_weekend": current.weekday() >= 5,
        "step": step,
        "simulation_day": simulation_day,
        "source": get_mode().value,
    }


def snapshot() -> SimulationTime:
    info = get_session_info()
    return SimulationTime(
        timestamp=info["timestamp"],
        step=int(info["step"]),
        simulation_day=int(info["simulation_day"]),
    )


def publish_to_bus(bus: Any) -> None:
    """Mirror the clock onto a SmartInfoBus for live consumers.

    Optional: the training path has no bus, so the clock never depends on one.
    """
    if bus is None:
        return
    try:
        snap = snapshot()
        bus.set(SIMULATION_TIME_KEY, snap.to_dict(), module="SimulationClock")
        bus.set(SIMULATION_STEP_KEY, snap.step, module="SimulationClock")
    except AttributeError:
        # A bus without .set() is a programming error, but publishing time must
        # never take down a trading loop.
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Step-based cooldowns
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StepCooldownTracker:
    """Cooldowns counted in steps rather than seconds.

    A cooldown of "300 seconds" never expires when 10,000 bars replay inside one
    real second. Counting steps behaves identically in training and live, given
    a fixed bar interval.
    """

    _last_action_step: Dict[str, int] = field(default_factory=dict)

    def can_act(self, action_key: str, min_steps: int = 1) -> bool:
        last = self._last_action_step.get(action_key)
        if last is None:
            return True
        return (get_current_step() - last) >= int(min_steps)

    def record_action(self, action_key: str) -> None:
        self._last_action_step[action_key] = get_current_step()

    def steps_since_action(self, action_key: str) -> int:
        last = self._last_action_step.get(action_key)
        if last is None:
            return 1 << 30  # effectively "never"
        return get_current_step() - last

    def reset(self) -> None:
        self._last_action_step.clear()


_GLOBAL_COOLDOWNS: Dict[str, StepCooldownTracker] = {}


def _tracker(namespace: str = "default") -> StepCooldownTracker:
    with _LOCK:
        return _GLOBAL_COOLDOWNS.setdefault(namespace, StepCooldownTracker())


def cooldown_elapsed(action_key: str, min_steps: int = 1, namespace: str = "default") -> bool:
    return _tracker(namespace).can_act(action_key, min_steps)


def record_cooldown_action(action_key: str, namespace: str = "default") -> None:
    _tracker(namespace).record_action(action_key)


def _reset_cooldowns() -> None:
    with _LOCK:
        for tracker in _GLOBAL_COOLDOWNS.values():
            tracker.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Namespace injection
# ─────────────────────────────────────────────────────────────────────────────


class _ClockDateTime(datetime):
    """`datetime` subclass whose now()/utcnow()/today() follow the clock."""

    @classmethod
    def now(cls, tz: Optional[_datetime_module.tzinfo] = None) -> datetime:  # type: ignore[override]
        return now(tz)

    @classmethod
    def utcnow(cls) -> datetime:  # type: ignore[override]
        return utcnow()

    @classmethod
    def today(cls) -> datetime:  # type: ignore[override]
        return now()


_ClockDateTime.__name__ = "datetime"
_ClockDateTime.__qualname__ = "datetime"


def _clock_time_module(original: types.ModuleType) -> types.ModuleType:
    """A stand-in for `time` whose time() follows the clock and whose sleep()
    is a no-op under simulation."""
    shim = types.ModuleType("time")
    for attr in dir(original):
        try:
            setattr(shim, attr, getattr(original, attr))
        except (AttributeError, TypeError):
            continue

    def _time() -> float:
        return now(timezone.utc).timestamp()

    def _sleep(seconds: float) -> None:
        if is_simulation_mode():
            return  # a sleeping module cannot participate in a replay loop
        original.sleep(seconds)

    shim.time = _time  # type: ignore[attr-defined]
    shim.sleep = _sleep  # type: ignore[attr-defined]
    return shim


def install_clock(module: types.ModuleType) -> list[str]:
    """Point a module's time primitives at this clock.

    Returns the names that were patched, for logging and tests.

    Handles both import styles found in this codebase:
      `from datetime import datetime`  -> module.datetime is the class
      `import datetime`               -> module.datetime is the module
    """
    patched: list[str] = []

    attr = getattr(module, "datetime", None)
    if isinstance(attr, type) and issubclass(attr, datetime) and attr is not _ClockDateTime:
        module.datetime = _ClockDateTime  # type: ignore[attr-defined]
        patched.append("datetime-class")
    elif isinstance(attr, types.ModuleType) and attr.__name__ == "datetime":
        shim = types.ModuleType("datetime")
        for name in dir(_datetime_module):
            setattr(shim, name, getattr(_datetime_module, name))
        shim.datetime = _ClockDateTime  # type: ignore[attr-defined]
        module.datetime = shim  # type: ignore[attr-defined]
        patched.append("datetime-module")

    time_attr = getattr(module, "time", None)
    if isinstance(time_attr, types.ModuleType) and time_attr.__name__ == "time":
        module.time = _clock_time_module(time_attr)  # type: ignore[attr-defined]
        patched.append("time-module")

    return patched


class simulation_mode:
    """Context manager that enables simulation mode and restores the previous
    mode and clock afterwards. Primarily for tests."""

    def __init__(self, start: Optional[datetime] = None) -> None:
        self._start = start
        self._previous_mode: Optional[TimeMode] = None
        self._previous_bar: Optional[datetime] = None
        self._previous_step: int = 0

    def __enter__(self) -> "simulation_mode":
        global _BAR_TIME, _STEP
        with _LOCK:
            self._previous_mode = _MODE
            self._previous_bar = _BAR_TIME
            self._previous_step = _STEP
        set_mode(TimeMode.SIMULATION)
        if self._start is not None:
            set_bar_time(self._start, step=0)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        global _BAR_TIME, _STEP
        if self._previous_mode is not None:
            set_mode(self._previous_mode)
        with _LOCK:
            _BAR_TIME = self._previous_bar
            _STEP = self._previous_step


def advance(delta: timedelta, steps: int = 1) -> None:
    """Move simulated time forward. Convenience for tests and replay tools."""
    current = get_bar_time()
    if current is None:
        return
    set_bar_time(current + delta, step=get_current_step() + int(steps))
