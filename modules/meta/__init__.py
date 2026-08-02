from __future__ import annotations

from typing import TYPE_CHECKING, Any

_EXPORTS: dict[str, str] = {
    "PPOCore": "ppo_core",
    "PPOCoreConfig": "ppo_core",
    "EnhancedPPONetwork": "ppo_core",
    "LiveActionMaskBuilder": "live_action_mask",
    "LiveMaskConfig": "live_action_mask",
    "PPOObservationBuilder": "ppo_observation_builder",
    "PPOObservationConfig": "ppo_observation_builder",
    "ObservationContractError": "ppo_observation_builder",
    "PPO_OBS_SIZE": "ppo_observation_builder",
    "PPO_OBS_VERSION": "ppo_observation_builder",
    "PPO_OBS_FEATURE_NAMES": "ppo_observation_builder",
    "FEATURE_GROUPS": "ppo_observation_builder",
    "get_ppo_observation_builder": "ppo_observation_builder",
    "build_ppo_observation": "ppo_observation_builder",
    "build_ppo_observation_for_instrument": "ppo_observation_builder",
}

__all__ = [
    "FEATURE_GROUPS",
    "PPO_OBS_FEATURE_NAMES",
    "PPO_OBS_SIZE",
    "PPO_OBS_VERSION",
    "EnhancedPPONetwork",
    "LiveActionMaskBuilder",
    "LiveMaskConfig",
    "ObservationContractError",
    "PPOCore",
    "PPOCoreConfig",
    "PPOObservationBuilder",
    "PPOObservationConfig",
    "build_ppo_observation",
    "build_ppo_observation_for_instrument",
    "get_ppo_observation_builder",
]

assert set(__all__) == set(_EXPORTS), "__all__ and _EXPORTS have drifted apart"


def __getattr__(name: str) -> Any:
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:
    from modules.meta.live_action_mask import LiveActionMaskBuilder, LiveMaskConfig
    from modules.meta.ppo_core import EnhancedPPONetwork, PPOCore, PPOCoreConfig
    from modules.meta.ppo_observation_builder import (
        FEATURE_GROUPS,
        PPO_OBS_FEATURE_NAMES,
        PPO_OBS_SIZE,
        PPO_OBS_VERSION,
        ObservationContractError,
        PPOObservationBuilder,
        PPOObservationConfig,
        build_ppo_observation,
        build_ppo_observation_for_instrument,
        get_ppo_observation_builder,
    )
