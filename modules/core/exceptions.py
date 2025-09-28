# modules/core/exceptions.py
from __future__ import annotations

class InputsNotReady(RuntimeError):
    """Raised when a module's required inputs are not yet available on the bus."""
    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__(f"Inputs not ready: {missing}")

class ModuleTimeout(TimeoutError):
    """Raised when a module exceeds its per-module timeout budget."""
    pass

class StageTimeout(TimeoutError):
    """Raised when a whole stage exceeds its timeout budget."""
    pass

class ExecutionSkipped(RuntimeError):
    """Raised to indicate intentional skip (e.g., low confidence threshold)."""
    pass
