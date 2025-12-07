"""
Legacy Meta Modules
===================

This folder contains deprecated modules that are no longer actively used
in the system. They are preserved for reference and backward compatibility.

DEPRECATED MODULES:
- meta_agent.py: Original MetaAgent system (replaced by PPOAgentShell)
- meta_rl_controller.py: Original MetaRLController (replaced by PPOAgentShell)
- ppo_agent.py: Original monolithic PPOAgent (replaced by 3-layer architecture)

The new architecture uses:
- ppo_types.py: Decision dataclasses
- ppo_core.py: Pure RL engine
- arbiter_logic.py: Domain logic
- ppo_agent_shell.py: SmartInfoBus gateway

For new development, use the modules in modules/meta/ directly.

Date deprecated: 2025-12-06
"""

import warnings

def _warn_legacy():
    warnings.warn(
        "Importing from modules.meta.legacy is deprecated. "
        "Use modules.meta.ppo_agent_shell.PPOAgentShell instead.",
        DeprecationWarning,
        stacklevel=3,
    )

# Lazy imports with deprecation warnings
def __getattr__(name):
    if name == "PPOAgent":
        _warn_legacy()
        from modules.meta.legacy.ppo_agent import PPOAgent
        return PPOAgent
    elif name == "MetaAgent":
        _warn_legacy()
        from modules.meta.legacy.meta_agent import MetaAgent
        return MetaAgent
    elif name == "MetaRLController":
        _warn_legacy()
        from modules.meta.legacy.meta_rl_controller import MetaRLController
        return MetaRLController
    raise AttributeError(f"module 'modules.meta.legacy' has no attribute '{name}'")

__all__ = ["PPOAgent", "MetaAgent", "MetaRLController"]
