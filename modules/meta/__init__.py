"""
Meta Module - PPO Agent and Related Components
==============================================

Architecture (v3.0):
- PPOCore: Pure RL engine (no SmartInfoBus knowledge)
- ArbiterLogic: Domain logic for per-instrument decisions
- PPOAgentShell: SmartInfoBus gateway

Observation Builder (v5.8):
- PPOObservationBuilder: builds the canonical 106-dim observation
- build_ppo_observation / build_ppo_observation_for_instrument

LAZY IMPORTS (PEP 562)
----------------------
This package previously imported every submodule eagerly. That made
`from modules.meta.ppo_observation_builder import ...` pull in ppo_core ->
torch and ppo_agent_shell -> the whole live stack, even though the observation
builder itself only needs numpy. The training environment therefore wrapped its
import in `except Exception`, which silently degraded the agent to an all-zero
observation whenever anything in that chain failed.

Attributes are now resolved on first access, so importing the observation builder
costs exactly the observation builder. The public API is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# name -> submodule that provides it
_EXPORTS: dict[str, str] = {
    # Types
    "InstrumentDecision": "ppo_types",
    "ArbiterMultiDecision": "ppo_types",
    "MemoryGateInfo": "ppo_types",
    "RiskInfo": "ppo_types",
    "GatingResult": "ppo_types",
    "InstrumentStats": "ppo_types",
    "InstrumentStatsTracker": "ppo_types",
    "DEFAULT_INSTRUMENTS": "ppo_types",
    "PRIMARY_INSTRUMENT": "ppo_types",
    # Core (requires torch)
    "PPOCore": "ppo_core",
    "PPOCoreConfig": "ppo_core",
    "EnhancedPPONetwork": "ppo_core",
    # Arbiter
    "ArbiterLogic": "arbiter_logic",
    # Shell (requires torch)
    "PPOAgentShell": "ppo_agent_shell",
    "PPOShellConfig": "ppo_agent_shell",
    # Observation (numpy only)
    "PPOObservationBuilder": "ppo_observation_builder",
    "PPOObservationConfig": "ppo_observation_builder",
    "PPO_OBS_SIZE": "ppo_observation_builder",
    "PPO_OBS_VERSION": "ppo_observation_builder",
    "FEATURE_GROUPS": "ppo_observation_builder",
    "get_ppo_observation_builder": "ppo_observation_builder",
    "build_ppo_observation": "ppo_observation_builder",
    "build_ppo_observation_for_instrument": "ppo_observation_builder",
}

# Spelled out literally rather than `list(_EXPORTS)` so static analysers can
# verify the export list (pyright reportUnsupportedDunderAll).
__all__ = [
    "DEFAULT_INSTRUMENTS",
    "FEATURE_GROUPS",
    "PPO_OBS_SIZE",
    "PPO_OBS_VERSION",
    "PRIMARY_INSTRUMENT",
    "ArbiterLogic",
    "ArbiterMultiDecision",
    "EnhancedPPONetwork",
    "GatingResult",
    "InstrumentDecision",
    "InstrumentStats",
    "InstrumentStatsTracker",
    "MemoryGateInfo",
    "PPOAgentShell",
    "PPOCore",
    "PPOCoreConfig",
    "PPOObservationBuilder",
    "PPOObservationConfig",
    "PPOShellConfig",
    "RiskInfo",
    "build_ppo_observation",
    "build_ppo_observation_for_instrument",
    "get_ppo_observation_builder",
]

assert set(__all__) == set(_EXPORTS), "__all__ and _EXPORTS have drifted apart"


def __getattr__(name: str) -> Any:
    """Resolve exported names on first access (PEP 562)."""
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(mod, name)
    globals()[name] = value  # cache so later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:  # static analysers still see the real symbols
    from modules.meta.arbiter_logic import ArbiterLogic
    from modules.meta.ppo_agent_shell import PPOAgentShell, PPOShellConfig
    from modules.meta.ppo_core import EnhancedPPONetwork, PPOCore, PPOCoreConfig
    from modules.meta.ppo_observation_builder import (
        FEATURE_GROUPS,
        PPO_OBS_SIZE,
        PPO_OBS_VERSION,
        PPOObservationBuilder,
        PPOObservationConfig,
        build_ppo_observation,
        build_ppo_observation_for_instrument,
        get_ppo_observation_builder,
    )
    from modules.meta.ppo_types import (
        DEFAULT_INSTRUMENTS,
        PRIMARY_INSTRUMENT,
        ArbiterMultiDecision,
        GatingResult,
        InstrumentDecision,
        InstrumentStats,
        InstrumentStatsTracker,
        MemoryGateInfo,
        RiskInfo,
    )
