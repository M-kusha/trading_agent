
from __future__ import annotations


class InputsNotReady(RuntimeError):
    def __init__(self, missing: list[str]):
        self.missing = list(missing)
        super().__init__(f"Inputs not ready: {missing}")

class ModuleTimeout(TimeoutError):
    pass

class StageTimeout(TimeoutError):
    pass

class ExecutionSkipped(RuntimeError):
    pass
