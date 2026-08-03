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
        -> PPOObservationBuilder.build()      (shared, 45 dims, v8.0)
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
    DEFAULT_INSTRUMENT,
    PPO_OBS_SIZE,
    PPO_OBS_VERSION,
    ObservationContractError,
    PPOObservationBuilder,
)
from modules.utils import simulation_time as simclock
from modules.utils.info_bus import InfoBusManager, SmartInfoBus
from trading.state import LiveStateHost


@module(**module_args("LivePPOAgent"))
class LivePPOAgent(BaseModule):

    # BaseModule alone does not provide `smart_bus` - it is installed by the
    # SmartInfoBus* mixins. Declaring and acquiring it explicitly means a
    # missing bus fails at construction rather than on the first decision.
    smart_bus: SmartInfoBus

    def _initialize(self) -> None:
        self.logger = logging.getLogger("LivePPOAgent")
        self.smart_bus = getattr(self, "smart_bus", None) or InfoBusManager.get_instance()
        self.obs_builder = PPOObservationBuilder()
        # Live observation state is produced by the SAME environment class that
        # training uses, fed a rolling window of broker bars. There is therefore
        # no second implementation to drift - see trading/state/live_state_host.
        self.state_host = LiveStateHost(instrument=DEFAULT_INSTRUMENT)
        self.mask_builder = LiveActionMaskBuilder(LiveMaskConfig())
        self.core: Optional[Any] = None
        self._last_decision: Dict[str, Any] = {}
        self._last_account_snapshot: Dict[str, Any] = {}
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
        mask = self._build_mask(self._last_account_snapshot)
        if not bool(getattr(self.core, "is_discrete_action_space", False)):
            raise RuntimeError("live policy must use the discrete MaskablePPO action contract")
        self.core.select_action(obs, deterministic=True, action_mask=mask)
        decoded = getattr(self.core, "last_discrete_action", None)
        if decoded is None:
            raise RuntimeError("discrete policy returned no decodable action")
        action_id = int(decoded.action_id)
        if action_id < 0 or action_id >= len(mask):
            raise RuntimeError(f"policy returned out-of-range action {action_id}")
        if not bool(mask[action_id]):
            raise RuntimeError(f"masked policy returned illegal action {action_id}")

        intent, size_mult = self.mask_builder.decode_action(action_id)
        direction = intent if intent in ("long", "short") else "flat"
        decision = {
            "action_id": action_id,
            "intent": intent,
            "direction": direction,
            "confidence": 1.0 if intent in ("long", "short") else 0.0,
            "gate_passed": intent != "hold",
            "instrument": DEFAULT_INSTRUMENT,
            "size_mult": float(size_mult),
            "obs_version": PPO_OBS_VERSION,
            "obs_size": int(PPO_OBS_SIZE),
            "timestamp": simclock.now().isoformat(),
        }
        self._last_decision = decision
        self._publish(decision, mask)
        return decision

    def _collect_state(self) -> Dict[str, Any]:
        """Build the observation inputs from live bars.

        Previously this read six pre-computed dicts off the bus, which meant a
        second implementation of every state producer and the drift that came
        with it. Now only raw bars and account state come from the bus, and the
        training environment derives everything else - so a schema change lands
        on both paths at once or neither.
        """
        name = "LivePPOAgent"

        frames = self.smart_bus.get("ohlcv_frames", name)
        if not isinstance(frames, dict) or not frames:
            raise RuntimeError(
                "LivePPOAgent has no bar window. MarketDataProvider must publish "
                "'ohlcv_frames' as {timeframe: DataFrame} before a decision can "
                "be made - an observation built from absent market data is not "
                "tradeable."
            )
        self.state_host.update(frames)

        account = self.smart_bus.get("account_state", name)
        if isinstance(account, dict) and account:
            required = (
                "balance",
                "equity",
                "initial_balance",
                "day_start_balance",
                "peak_balance",
                "trades_today",
                "consecutive_losses",
                "has_position",
                "position_count",
                "position",
                "risk_day",
                "risk_anchor_authoritative",
                "timestamp",
            )
            missing = [key for key in required if key not in account]
            if missing:
                raise RuntimeError(
                    "LivePPOAgent account state lacks authoritative risk anchors: "
                    + ", ".join(missing)
                )
            try:
                position_count = int(account["position_count"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError("LivePPOAgent account position_count is invalid") from exc
            has_position = bool(account["has_position"])
            if position_count < 0 or has_position != (position_count > 0):
                raise RuntimeError(
                    f"LivePPOAgent inconsistent position state: has_position={has_position}, "
                    f"position_count={position_count}"
                )
            if position_count > 1:
                raise RuntimeError(
                    "LivePPOAgent uses a single-position observation contract; "
                    f"broker reports {position_count} positions"
                )
            if has_position and not isinstance(account["position"], dict):
                raise RuntimeError("LivePPOAgent open position lacks a normalized position payload")
            if not has_position and account["position"] is not None:
                raise RuntimeError("LivePPOAgent flat account carries a non-null position payload")

            self.state_host.sync_account(
                balance=float(account.get("balance", 0.0) or 0.0),
                equity=float(account.get("equity", 0.0) or 0.0),
                position=account["position"],
                daily_trades=int(account["trades_today"]),
                consecutive_losses=int(account["consecutive_losses"]),
                initial_balance=float(account["initial_balance"]),
                day_start_balance=float(account["day_start_balance"]),
                peak_balance=float(account["peak_balance"]),
            )
            self._last_account_snapshot = dict(account)
        else:
            raise RuntimeError(
                "LivePPOAgent has no account state. Sizing and risk dimensions "
                "would describe a simulated account rather than the live one."
            )

        return self.state_host.observation_inputs()

    def _build_mask(self, account_state: Dict[str, Any]) -> np.ndarray:
        return self.mask_builder.get_action_mask(
            has_position=bool(account_state.get("has_position", False)),
            trade_open_allowed=bool(account_state.get("risk_anchor_authoritative", False)),
            current_dd=account_state.get("current_drawdown"),
            daily_dd=account_state.get("daily_drawdown"),
            daily_trades=int(account_state.get("trades_today", 0) or 0),
            consecutive_losses=int(account_state.get("consecutive_losses", 0) or 0),
            risk_timestamp=account_state.get("timestamp"),
            current_time=simclock.now(),
        )

    def _publish(self, decision: Dict[str, Any], mask: np.ndarray) -> None:
        name = "LivePPOAgent"
        thesis = (
            f"policy chose {decision['intent']} (size x{decision['size_mult']:.2f}) "
            f"on observation schema v{PPO_OBS_VERSION}"
        )
        self.smart_bus.set("ppo_final_decision", decision, module=name, thesis=thesis)
        self.smart_bus.set("ppo_gate_passed", decision["gate_passed"], module=name, thesis=thesis)
        self.smart_bus.set("ppo_position_size", decision["size_mult"], module=name, thesis=thesis)
        self.smart_bus.set("action_mask", mask.tolist(), module=name, thesis="legal actions this step")

    def reset(self) -> None:
        super().reset()
        self._last_decision = {}
        self._last_account_snapshot = {}
        self._consecutive_failures = 0
