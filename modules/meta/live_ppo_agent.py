#!/usr/bin/env python3
"""Live inference adapter.

Replaces ppo_agent_shell.py (2,391 lines), which was deleted because it read
bus keys whose producers no longer exist - TrendExpert_voting_proposal,
memory_gate, danger_zones, world_model info - all with default=None, so live
would have built a silently degraded observation rather than failing.

The design rule here is that live and training must share ONE observation
implementation. This module therefore calls the same PPOObservationBuilder the
environment calls, against the same state producers, so a schema change cannot
drift between the two paths. It holds no feature logic of its own.

Flow:
    bus market/account state
        -> PPOObservationBuilder.build()      (shared, 40 dims, v7.0)
        -> PPOCore.select_action()            (loaded checkpoint)
        -> decode to intent + size            (same mapping as the env)
        -> LiveActionMaskBuilder              (hard constraints)
        -> ppo_final_decision on the bus      (consumed by PositionManager)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.meta.live_action_mask import LiveActionMaskBuilder, LiveMaskConfig
from modules.meta.ppo_observation_builder import (
    PPO_OBS_SIZE,
    PPO_OBS_VERSION,
    ObservationContractError,
    PPOObservationBuilder,
)
from modules.utils import simulation_time as simclock


@module(**module_args("LivePPOAgent"))
class LivePPOAgent(BaseModule):

    def _initialize(self) -> None:
        self.logger = logging.getLogger("LivePPOAgent")
        self.obs_builder = PPOObservationBuilder()
        self.mask_builder = LiveActionMaskBuilder(LiveMaskConfig())
        self.core: Optional[Any] = None
        self._last_decision: Dict[str, Any] = {}
        self._consecutive_failures = 0

        simclock.set_mode(simclock.TimeMode.LIVE)
        self.logger.info(
            "LivePPOAgent ready: observation schema v%s (%d dims)",
            PPO_OBS_VERSION,
            PPO_OBS_SIZE,
        )

    def load_model(self, path: str) -> None:
        from modules.meta.ppo_core import PPOCore, PPOCoreConfig

        core = PPOCore(PPOCoreConfig())
        core.load(path)
        if core.config.obs_size != PPO_OBS_SIZE:
            raise ValueError(
                f"checkpoint expects {core.config.obs_size} observation dims, "
                f"builder produces {PPO_OBS_SIZE} (schema v{PPO_OBS_VERSION}). "
                f"Retrain or load a matching checkpoint."
            )
        self.core = core
        self.logger.info("loaded policy from %s", path)

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        if self.core is None:
            raise RuntimeError(
                "LivePPOAgent has no policy loaded. Call load_model() before "
                "trading - an agent without a policy must not emit decisions."
            )

        state = self._collect_state()
        try:
            obs = self.obs_builder.build(**state)
        except ObservationContractError:
            # A malformed observation is never tradeable. Surfacing it stops the
            # loop rather than letting a padded or partial vector reach the
            # policy, which is exactly how the previous shell failed.
            self._consecutive_failures += 1
            raise

        self._consecutive_failures = 0
        action_id = int(self.core.select_action(obs, deterministic=True)[0])

        mask = self._build_mask(state["account_state"])
        if not bool(mask[action_id]):
            action_id = 0  # HOLD is always legal

        intent, size_mult = self.mask_builder.decode_action(action_id)
        decision = {
            "action_id": action_id,
            "intent": intent,
            "size_mult": float(size_mult),
            "obs_version": PPO_OBS_VERSION,
            "obs_size": int(PPO_OBS_SIZE),
            "timestamp": simclock.now().isoformat(),
        }
        self._last_decision = decision
        self._publish(decision, mask)
        return decision

    def _collect_state(self) -> Dict[str, Dict[str, Any]]:
        """Gather the builder's inputs from the bus.

        Every key here has a live producer. Missing data raises rather than
        defaulting: a decision made on absent market state is worse than no
        decision.
        """
        name = "LivePPOAgent"
        required = {
            "market_data": "market_data",
            "expert_signals": "expert_signals",
            "risk_state": "risk_data",
            "account_state": "account_state",
            "trading_mode_state": "trading_mode_state",
            "governor_state": "governor_state",
        }
        state: Dict[str, Dict[str, Any]] = {}
        missing = []
        for arg, bus_key in required.items():
            value = self.smart_bus.get(bus_key, name)
            if not isinstance(value, dict) or not value:
                missing.append(bus_key)
            else:
                state[arg] = value
        if missing:
            raise RuntimeError(
                f"LivePPOAgent cannot build an observation - bus keys absent or "
                f"empty: {', '.join(missing)}"
            )
        return state

    def _build_mask(self, account_state: Dict[str, Any]) -> np.ndarray:
        return self.mask_builder.get_action_mask(
            has_position=bool(account_state.get("has_position", False)),
            current_dd=float(account_state.get("current_drawdown", 0.0) or 0.0),
            daily_dd=float(account_state.get("daily_drawdown", 0.0) or 0.0),
            daily_trades=int(account_state.get("trades_today", 0) or 0),
            consecutive_losses=int(account_state.get("consecutive_losses", 0) or 0),
        )

    def _publish(self, decision: Dict[str, Any], mask: np.ndarray) -> None:
        name = "LivePPOAgent"
        thesis = (
            f"policy chose {decision['intent']} (size x{decision['size_mult']:.2f}) "
            f"on observation schema v{PPO_OBS_VERSION}"
        )
        self.smart_bus.set("ppo_final_decision", decision, module=name, thesis=thesis)
        self.smart_bus.set("ppo_gate_passed", decision["intent"] != "hold", module=name, thesis=thesis)
        self.smart_bus.set("ppo_position_size", decision["size_mult"], module=name, thesis=thesis)
        self.smart_bus.set("action_mask", mask.tolist(), module=name, thesis="legal actions this step")

    def reset(self) -> None:
        super().reset()
        self._last_decision = {}
        self._consecutive_failures = 0
