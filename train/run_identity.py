"""Identity and liveness for a training run's telemetry.

A dashboard read a metrics file and reported `training_active=true`, `healthy`,
and 100.079% progress with an ETA of -38 seconds, while no training process
existed. Nothing in the payload said which run produced it, when, or whether the
producer was still alive - so a file left on disk from a finished run was
indistinguishable from a live one.

Three facts fix that, and they must travel with every write:

  run_id      which run produced this. A new run invalidates the old history
              rather than appending to it.
  sequence    monotonic per run. A consumer can tell a genuinely new snapshot
              from the same one re-read, and must not advance charts or alerts
              on a repeat.
  produced_at the producer's own UTC clock, so staleness is measured against
              when the data was made rather than when it was read.

`state` separates "the producer is running" from "the file exists". Connection
health and training health are different questions and a dashboard that
conflates them will always answer the easy one.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

# A snapshot older than this has almost certainly outlived its producer: the
# callbacks write far more often than this under any normal configuration.
STALE_AFTER_SECONDS = 120.0

RUN_STATES = ("starting", "running", "completed", "failed")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunIdentity:
    """Stamps telemetry so a consumer can tell whose it is and whether it lives."""

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started_at: str = field(default_factory=_utc_now_iso)
    pid: int = field(default_factory=os.getpid)
    state: str = "starting"
    _sequence: int = 0

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    @property
    def sequence(self) -> int:
        return self._sequence

    def set_state(self, state: str) -> None:
        if state not in RUN_STATES:
            raise ValueError(f"unknown run state {state!r}; expected one of {RUN_STATES}")
        self.state = state

    def stamp(self) -> Dict[str, Any]:
        """The block that must accompany every metrics write."""
        return {
            "run_id": self.run_id,
            "sequence": self.next_sequence(),
            "produced_at_utc": _utc_now_iso(),
            "started_at_utc": self.started_at,
            "state": self.state,
            "pid": self.pid,
        }


def snapshot_age_seconds(stamp: Dict[str, Any], *, now: Any = None) -> float:
    """Seconds since the producer wrote this, by the producer's own clock."""
    produced = (stamp or {}).get("produced_at_utc")
    if not produced:
        return float("inf")
    try:
        made = datetime.fromisoformat(str(produced))
    except ValueError:
        return float("inf")
    if made.tzinfo is None:
        made = made.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if getattr(current, "tzinfo", None) is None:
        current = current.replace(tzinfo=timezone.utc)
    return max(0.0, (current - made).total_seconds())


def is_training_live(stamp: Dict[str, Any], *, now: Any = None,
                     stale_after: float = STALE_AFTER_SECONDS) -> bool:
    """True only if the producer says it is running AND said so recently.

    Both halves matter. A finished run leaves `completed`; a killed one leaves
    `running` and simply stops updating, which only the age can reveal.
    """
    if not stamp:
        return False
    if str(stamp.get("state")) != "running":
        return False
    return snapshot_age_seconds(stamp, now=now) <= stale_after


def describe_liveness(stamp: Dict[str, Any], *, now: Any = None,
                      stale_after: float = STALE_AFTER_SECONDS) -> Dict[str, Any]:
    """Everything a consumer needs to render honestly, including 'unknown'."""
    if not stamp:
        return {
            "run_id": None,
            "sequence": None,
            "state": "unknown",
            "training_live": False,
            "age_seconds": None,
            "stale": True,
            "reason": "no run identity in telemetry",
        }

    age = snapshot_age_seconds(stamp, now=now)
    stale = age > stale_after
    live = is_training_live(stamp, now=now, stale_after=stale_after)

    if live:
        reason = "producer running and current"
    elif str(stamp.get("state")) != "running":
        reason = f"producer state is {stamp.get('state')!r}"
    else:
        reason = f"no update for {age:.0f}s (stale after {stale_after:.0f}s)"

    return {
        "run_id": stamp.get("run_id"),
        "sequence": stamp.get("sequence"),
        "state": stamp.get("state"),
        "training_live": live,
        "age_seconds": None if age == float("inf") else age,
        "stale": stale,
        "reason": reason,
    }
