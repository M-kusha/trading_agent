"""
Meta Module - PPO Agent and Related Components
==============================================

This module contains the PPO-based intelligent arbiter system:

Architecture (v3.0):
- PPOCore: Pure RL engine (no SmartInfoBus knowledge)
- ArbiterLogic: Domain logic for per-instrument decisions
- PPOAgentShell: SmartInfoBus gateway

Types:
- InstrumentDecision: Decision for a single instrument
- ArbiterMultiDecision: Multi-instrument decision container
- MemoryGateInfo: Normalized memory gate data
- RiskInfo: Normalized risk data
- GatingResult: Combined gating result

Observation Builder (v4.0):
- PPOObservationBuilder: Builds 64-dim observations
- build_ppo_observation: Global observation builder
- build_ppo_observation_for_instrument: Per-instrument observation builder

Note: Legacy PPOAgent/PPOConfig have been removed. The 3-layer architecture
(PPOCore + ArbiterLogic + PPOAgentShell) is the canonical implementation.
"""

from modules.meta.ppo_types import (
    InstrumentDecision,
    ArbiterMultiDecision,
    MemoryGateInfo,
    RiskInfo,
    GatingResult,
    InstrumentStats,
    InstrumentStatsTracker,
    DEFAULT_INSTRUMENTS,
    PRIMARY_INSTRUMENT,
)

from modules.meta.ppo_core import (
    PPOCore,
    PPOCoreConfig,
    EnhancedPPONetwork,
)

from modules.meta.arbiter_logic import ArbiterLogic

from modules.meta.ppo_agent_shell import (
    PPOAgentShell,
    PPOShellConfig,
)

from modules.meta.ppo_observation_builder import (
    PPOObservationBuilder,
    PPOObservationConfig,
    PPO_OBS_SIZE,
    PPO_OBS_VERSION,
    FEATURE_GROUPS,
    get_ppo_observation_builder,
    build_ppo_observation,
    build_ppo_observation_for_instrument,
)

__all__ = [
    # Types
    "InstrumentDecision",
    "ArbiterMultiDecision",
    "MemoryGateInfo",
    "RiskInfo",
    "GatingResult",
    "InstrumentStats",
    "InstrumentStatsTracker",
    "DEFAULT_INSTRUMENTS",
    "PRIMARY_INSTRUMENT",
    # Core
    "PPOCore",
    "PPOCoreConfig",
    "EnhancedPPONetwork",
    # Arbiter
    "ArbiterLogic",
    # Shell
    "PPOAgentShell",
    "PPOShellConfig",
    # Observation
    "PPOObservationBuilder",
    "PPOObservationConfig",
    "PPO_OBS_SIZE",
    "PPO_OBS_VERSION",
    "FEATURE_GROUPS",
    "get_ppo_observation_builder",
    "build_ppo_observation",
    "build_ppo_observation_for_instrument",
]
