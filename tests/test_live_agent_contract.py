"""Contract tests for the live inference adapter.

These exist because the module it replaces (ppo_agent_shell.py) failed silently:
it read bus keys whose producers had been deleted, each with default=None, so
live built a degraded observation and traded on it without raising.
"""

from __future__ import annotations

import asyncio

import pytest


def test_registers_with_the_expected_contract():
    from modules.meta.live_ppo_agent import LivePPOAgent

    meta = LivePPOAgent.__module_metadata__
    assert meta.name == "LivePPOAgent"
    for key in ("ppo_final_decision", "ppo_gate_passed", "ppo_position_size"):
        assert key in meta.provides, f"{key} must be published for the executor chain"
    for key in ("market_data", "account_state", "governor_state"):
        assert key in meta.requires


def test_uses_the_shared_observation_builder():
    """Live and training must not hold two implementations of the observation."""
    from modules.meta.live_ppo_agent import LivePPOAgent
    from modules.meta.ppo_observation_builder import PPO_OBS_SIZE, PPO_OBS_VERSION

    agent = LivePPOAgent()
    assert agent.obs_builder.config.obs_size == PPO_OBS_SIZE
    assert agent.obs_builder.config.version == PPO_OBS_VERSION


def test_refuses_to_act_without_a_policy():
    """An agent with no model must raise, never emit a default decision."""
    from modules.meta.live_ppo_agent import LivePPOAgent

    agent = LivePPOAgent()
    with pytest.raises(RuntimeError, match="no policy loaded"):
        asyncio.run(agent.process())


def test_rejects_a_checkpoint_of_the_wrong_width(tmp_path, monkeypatch):
    """The old shell zero-padded a mismatched checkpoint. This one must not."""
    from modules.meta import live_ppo_agent as mod
    from modules.meta.ppo_observation_builder import PPO_OBS_SIZE

    class StubCore:
        def __init__(self, config):
            self.config = config

        def load(self, path):
            self.config.obs_size = PPO_OBS_SIZE + 7  # deliberately wrong

    class StubConfig:
        obs_size = PPO_OBS_SIZE

    import types
    stub = types.ModuleType("modules.meta.ppo_core")
    stub.PPOCore = StubCore
    stub.PPOCoreConfig = StubConfig
    monkeypatch.setitem(__import__("sys").modules, "modules.meta.ppo_core", stub)

    agent = mod.LivePPOAgent()
    with pytest.raises(ValueError, match="observation dims"):
        agent.load_model(str(tmp_path / "model.zip"))


def test_registry_and_contract_agree():
    """The YAML registry and modules/contracts.py must not drift apart."""
    import io
    from pathlib import Path

    import yaml

    from modules.contracts import module_args

    root = Path(__file__).resolve().parents[1]
    reg = yaml.safe_load(io.open(root / "config" / "module_registry.yaml", encoding="utf-8"))
    entry = reg.get("modules", reg)["LivePPOAgent"]
    contract = module_args("LivePPOAgent")

    assert set(entry["provides"]) == set(contract["provides"])
    assert set(entry["requires"]) == set(contract["requires"])


def test_bus_is_actually_reachable():
    """Regression: BaseModule alone does not provide `smart_bus` - it comes from
    the SmartInfoBus* mixins. The first version of this module inherited only
    BaseModule, so every bus call would have raised AttributeError at the first
    live decision. The existing tests missed it because the no-policy guard
    raises earlier, so nothing ever reached the bus."""
    from modules.meta.live_ppo_agent import LivePPOAgent

    agent = LivePPOAgent()
    assert hasattr(agent, "smart_bus"), "agent has no bus - it cannot publish decisions"

    agent.smart_bus.set("live_agent_probe", {"ok": True}, module="test")
    assert agent.smart_bus.get("live_agent_probe", "LivePPOAgent") == {"ok": True}


def test_missing_bus_inputs_raise_and_name_the_keys():
    """An absent market feed must stop the loop, not produce a default decision."""
    from modules.meta.live_ppo_agent import LivePPOAgent

    agent = LivePPOAgent()
    with pytest.raises(RuntimeError, match="bus keys absent or empty"):
        agent._collect_state()
