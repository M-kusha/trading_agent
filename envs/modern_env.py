# envs/modern_env.py
"""
Modern SmartInfoBus Trading Environment (Unified, Bus-First)

Key ideas:
- **Bus-first**: the env consumes market data, features, rewards & limits from SmartInfoBus
  when present; it falls back to local logic only when needed.
- **No duplication**: if MarketDataProvider (or any other module) is publishing market_data,
  the env avoids republishing those keys.
- **Execution is external**: another module owns positions/trades/fills/portfolio metrics.
  This env does NOT publish execution or portfolio keys.
- **Soft coupling**: everything bus-related is optional and guarded; the env works offline.

Gymnasium v0.26+ API (step returns obs, reward, terminated, truncated, info).
"""

from __future__ import annotations

import copy
import warnings
import asyncio
import threading
import time
from typing import Any, Dict, Optional, Tuple, List, Set, cast

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from concurrent.futures import Future

from .config import TradingConfig, MarketState, EpisodeMetrics

# SmartInfoBus infrastructure
try:
    from modules.utils.info_bus import InfoBusManager
    from modules.utils.audit_utils import RotatingLogger
    SMARTINFOBUS_AVAILABLE = True
except Exception:
    InfoBusManager = None  # type: ignore
    RotatingLogger = None  # type: ignore
    SMARTINFOBUS_AVAILABLE = False

# Core module system
try:
    from modules.core.module_system import ModuleOrchestrator
    MODULE_SYSTEM_AVAILABLE = True
except Exception:
    ModuleOrchestrator = None  # type: ignore
    MODULE_SYSTEM_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


class ModernTradingEnv(gym.Env):
    """
    SmartInfoBus-integrated trading env (bus-first) without embedded execution.
    Gymnasium v0.26+ API.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # ──────────────────────────────────────────────────────────────
    # Init
    # ──────────────────────────────────────────────────────────────
    def __init__(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        config: Optional[TradingConfig] = None,
    ):
        super().__init__()

        # Defaults to avoid init failures
        self.config = config or TradingConfig()
        self._default_obs_size = int(getattr(self.config, "environment_observation_size", 256) or 256)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self._default_obs_size,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_dim = int(np.prod(self.action_space.shape)) if self.action_space.shape else 0

        # Execution config (fallbacks — modules may override via bus)
        self.primary_timeframe = getattr(self.config, "primary_timeframe", "H1") or "H1"
        self.default_spread = float(getattr(self.config, "default_spread", 0.0) or 0.0)
        self.slippage_pts = float(getattr(self.config, "slippage_pts", 0.0) or 0.0)
        self.commission_per_million = float(getattr(self.config, "commission_per_million", 0.0) or 0.0)

        # Bus-first policy toggles (all optional; sane defaults)
        self.bus_first = bool(getattr(self.config, "bus_first", True))
        self.prefer_bus_data = bool(getattr(self.config, "prefer_bus_data", True))
        self.prefer_bus_features = bool(getattr(self.config, "prefer_bus_features", True))
        self.prefer_bus_rewards = bool(getattr(self.config, "prefer_bus_rewards", True))
        self.prefer_bus_limits = bool(getattr(self.config, "prefer_bus_limits", True))
        self.halt_on_emergency = bool(getattr(self.config, "halt_on_emergency", True))

        # Intent filters (kept for API stability; no longer used for execution here)
        self.min_confidence = float(getattr(self.config, "min_confidence", 0.0) or 0.0)
        self.min_intensity = float(getattr(self.config, "min_intensity", 0.0) or 0.0)
        self.ignore_hold = bool(getattr(self.config, "ignore_hold", True))

        # Logger
        self.logger = self._create_logger()

        # SmartInfoBus / Orchestrator
        self.smart_bus = None  # type: ignore[assignment]
        self.smart_bus_enabled = False
        self.orchestrator = None  # type: ignore[assignment]
        self.orchestrator_enabled = False

        # Async loop for orchestrator
        self._aio_loop: Optional[asyncio.AbstractEventLoop] = None
        self._aio_thread: Optional[threading.Thread] = None
        self._aio_ready = threading.Event()
        self._pending_futures: Set[Future] = set()
        self._pend_lock = threading.Lock()
        self._bus_ready = threading.Event()
        self._orch_ready = threading.Event()

        # Bring systems up
        self._initialize_systems()
        self._start_event_loop_thread()

        # Data (fallback local data)
        self.orig_data = data_dict
        self.data = copy.deepcopy(data_dict)
        self.instruments: List[str] = list(self.data.keys())
        self._validate_data()

        if not self.instruments:
            raise ValueError("No instruments provided to ModernTradingEnv")
        if self.primary_timeframe not in self.data[self.instruments[0]]:
            self.primary_timeframe = list(self.data[self.instruments[0]].keys())[0]

        # Market/account anchors (env-internal; env no longer publishes execution or portfolio keys)
        initial_balance = float(self.config.initial_balance)
        self.market_state = MarketState(balance=initial_balance, peak_balance=initial_balance, current_step=0, current_drawdown=0.0)
        self.balance: float = float(initial_balance)
        self.equity: float = float(initial_balance)
        self._last_equity: float = self.equity

        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        self.episode_metrics = EpisodeMetrics()

        # Finalize spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2 * len(self.instruments),), dtype=np.float32)
        self.action_dim = int(np.prod(self.action_space.shape)) if self.action_space.shape else 0
        self.observation_space = self._get_observation_space()

        # Detect whether provider is active on the bus
        self._bus_data_active = self._detect_bus_data_active()

        # Setup env context
        self._setup_environment()

        # Only publish local market windows if provider is NOT active
        if not self._bus_data_active:
            self._store_market_data_local()

        modules = len(self.orchestrator.modules) if (self.orchestrator and hasattr(self.orchestrator, "modules")) else 0
        self.logger.info(f"🚀 MODERN_ENV_INITIALIZED: {len(self.instruments)} instruments, {modules} modules - Bus-first")

    # ──────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────
    def _create_logger(self):
        try:
            if SMARTINFOBUS_AVAILABLE and RotatingLogger:
                max_lines = int(getattr(self.config, "log_rotation_lines", 2000) or 2000)
                return RotatingLogger(
                    name="ModernTradingEnv",
                    log_path="logs/modern_env.log",
                    max_lines=max_lines,
                    operator_mode=True,
                )
            import logging
            lg = logging.getLogger("ModernTradingEnv")
            if not lg.handlers:
                h = logging.StreamHandler()
                import logging as _L
                fmt = _L.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                h.setFormatter(fmt)
                lg.addHandler(h)
                lg.setLevel(_L.INFO)
            return lg
        except Exception as e:
            print(f"[WARN] Failed to create logger: {e}")

            class Fallback:
                def info(self, m): print(f"[INFO] {m}")
                def warning(self, m): print(f"[WARN] {m}")
                def error(self, m): print(f"[ERROR] {m}")
                def debug(self, m): print(f"[DBG] {m}")
            return Fallback()

    # ──────────────────────────────────────────────────────────────
    # SmartInfoBus / Orchestrator bring-up
    # ──────────────────────────────────────────────────────────────
    def _initialize_systems(self):
        # SmartInfoBus
        if SMARTINFOBUS_AVAILABLE and InfoBusManager:
            try:
                IB = cast(Any, InfoBusManager)

                def init_bus():
                    try:
                        self.smart_bus = IB.get_instance()  # type: ignore[attr-defined]
                        self.smart_bus_enabled = True
                    except Exception as e:
                        self.logger.warning(f"SmartInfoBus init failed: {e}")
                        self.smart_bus = None
                        self.smart_bus_enabled = False
                    finally:
                        self._bus_ready.set()

                t = threading.Thread(target=init_bus, daemon=True)
                t.start()
                t.join(timeout=max(0.5, float(getattr(self.config, "info_bus_init_timeout", 2.0))))
            except Exception as e:
                self.logger.warning(f"SmartInfoBus bring-up error: {e}")

        if not self.smart_bus_enabled or self.smart_bus is None:
            # Minimal in-process fallback bus
            self.smart_bus = self._create_fallback_smart_bus()
            self.smart_bus_enabled = True
            self._bus_ready.set()

        # Orchestrator
        if MODULE_SYSTEM_AVAILABLE and ModuleOrchestrator:
            try:
                MO = cast(Any, ModuleOrchestrator)

                def init_orch():
                    try:
                        # Reuse singleton orchestrator to avoid repeated boot sequences
                        orch = MO.get_instance()  # type: ignore[attr-defined]
                        if hasattr(orch, "initialize"):
                            orch.initialize()
                        self.orchestrator = orch
                        self.orchestrator_enabled = True
                    except Exception as e:
                        self.logger.warning(f"Orchestrator init failed: {e}")
                        self.orchestrator = None
                        self.orchestrator_enabled = False
                    finally:
                        self._orch_ready.set()

                t = threading.Thread(target=init_orch, daemon=True)
                t.start()
                t.join(timeout=max(1.0, float(getattr(self.config, "orchestrator_init_timeout", 2.0))))
            except Exception as e:
                self.logger.warning(f"Orchestrator bring-up error: {e}")

        self._start_post_init_monitor()

    def _start_event_loop_thread(self):
        def runner():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._aio_loop = loop
                self._aio_ready.set()
                loop.run_forever()
            except Exception as e:
                self.logger.warning(f"Async loop thread error: {e}")

        if self._aio_thread and self._aio_thread.is_alive():
            return
        self._aio_thread = threading.Thread(target=runner, daemon=True)
        self._aio_thread.start()
        self._aio_ready.wait(timeout=1.5)

    # ──────────────────────────────────────────────────────────────
    # Bus data detection
    # ──────────────────────────────────────────────────────────────
    def _detect_bus_data_active(self) -> bool:
        if not (self.smart_bus and self.prefer_bus_data):
            return False
        try:
            mtd = self.smart_bus.get("multi_timeframe_data", "Environment")
            if isinstance(mtd, dict) and mtd:
                return True
        except Exception:
            pass
        try:
            mi = self.smart_bus.get("module_insights", "Environment")
            if isinstance(mi, dict) and mi.get("provider") == "MarketDataProvider":
                return True
        except Exception:
            pass
        try:
            md = self.smart_bus.get("market_data", "Environment")
            if isinstance(md, dict) and md:
                k = next(iter(md))
                if isinstance(md[k], dict) and "close" in md[k]:
                    return True
        except Exception:
            pass
        return False

    # ──────────────────────────────────────────────────────────────
    # Data / spaces / setup
    # ──────────────────────────────────────────────────────────────
    def _validate_data(self):
        required = {"open", "high", "low", "close"}
        for inst, tfs in self.data.items():
            if not isinstance(tfs, dict) or not tfs:
                raise ValueError(f"Empty timeframes for instrument {inst}")
            for tf, df in tfs.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    raise ValueError(f"Empty DataFrame for {inst}/{tf}")
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"Missing columns for {inst}/{tf}: {missing}")
                if "volume" not in df.columns:
                    self.data[inst][tf]["volume"] = 1.0

    def _get_observation_space(self) -> spaces.Box:
        default_size = int(getattr(self, "_default_obs_size", 256))
        try:
            if self.smart_bus and self.prefer_bus_features:
                sz = self.smart_bus.get("environment_observation_size", "Environment")
                if isinstance(sz, int) and sz > 0:
                    default_size = sz
        except Exception:
            pass
        return spaces.Box(low=-np.inf, high=np.inf, shape=(default_size,), dtype=np.float32)

    def _setup_environment(self):
        try:
            if self.smart_bus:
                # 1) Always publish environment_config (now includes 'mode')
                env_cfg = {
                    "instruments": self.instruments,
                    "initial_balance": float(self.config.initial_balance),
                    "action_dim": int(self.action_dim),
                    "max_steps": int(self.config.max_steps),
                    "bus_data_active": bool(self._bus_data_active),
                    "mode": "live" if getattr(self.config, "live_mode", False) else "sim",
                }
                self.smart_bus.set("environment_config", env_cfg, module="Environment",
                                thesis="Environment configuration")

                # 2) Provide execution_mode only if nobody else has
                if self.smart_bus.get("execution_mode", "Environment") is None:
                    self.smart_bus.set(
                        "execution_mode",
                        env_cfg["mode"],
                        module="Environment",
                        thesis="Default execution mode (env fallback)"
                    )

        except Exception:
            pass

    def _store_market_data_local(self):
        """
        Publish local market windows ONLY when MarketDataProvider is not active.
        This prevents duplicate market_data feeds on the bus.
        """
        if not self.smart_bus or self._bus_data_active:
            return

        step = int(self.current_step)
        aggregated = {}
        for instrument in self.instruments:
            for timeframe in ["H1", "H4", "D1"]:
                try:
                    if timeframe not in self.data[instrument]:
                        continue
                    df = self.data[instrument][timeframe]
                    if step >= len(df):
                        continue
                    # rolling window
                    w = min(100, step + 1)
                    s = max(0, step - w + 1)
                    ohlcv = {
                        "open": df["open"].iloc[s:step + 1].values,
                        "high": df["high"].iloc[s:step + 1].values,
                        "low": df["low"].iloc[s:step + 1].values,
                        "close": df["close"].iloc[s:step + 1].values,
                        "volume": df["volume"].iloc[s:step + 1].values,
                        "step": step,
                        "instrument": instrument,
                        "timeframe": timeframe,
                    }
                    self.smart_bus.set(
                        f"market_data_{instrument}_{timeframe}",
                        ohlcv,
                        module="Environment",
                        thesis=f"Local market data window for {instrument} {timeframe} (fallback)",
                    )
                    aggregated.setdefault(instrument, {})[timeframe] = ohlcv
                except Exception:
                    pass

        try:
            self.smart_bus.set("market_data", aggregated, module="Environment", thesis="Aggregated market data (fallback)")
            self.smart_bus.set("step_idx", int(step), module="Environment", thesis="Current step index (fallback)")
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.logger.info(f"🔄 ENVIRONMENT_RESET: Episode {self.episode_count + 1}")
        self.episode_count += 1
        self.episode_metrics = EpisodeMetrics()
        self.data = copy.deepcopy(self.orig_data)

        initial_balance = float(self.config.initial_balance)
        self.market_state = MarketState(balance=initial_balance, peak_balance=initial_balance, current_step=self._select_starting_step(), current_drawdown=0.0)
        self.current_step = int(self.market_state.current_step)

        self.balance = float(initial_balance)
        self.equity = float(initial_balance)
        self._last_equity = float(initial_balance)

        try:
            if self.smart_bus:
                # Keep environment_config and execution_mode refreshed at reset
                env_cfg = {
                    "instruments": self.instruments,
                    "initial_balance": float(self.config.initial_balance),
                    "action_dim": int(self.action_dim),
                    "max_steps": int(self.config.max_steps),
                    "bus_data_active": bool(self._bus_data_active),
                    "mode": "live" if getattr(self.config, "live_mode", False) else "sim",
                }
                self.smart_bus.set(
                    "environment_config",
                    env_cfg,
                    module="Environment",
                    thesis="Environment configuration (reset)"
                )
                self.smart_bus.set(
                    "execution_mode",
                    env_cfg["mode"],
                    module="Environment",
                    thesis="Execution mode (reset)"
                )
                # Legacy alias for consumers expecting env_mode
                try:
                    self.smart_bus.set(
                        "env_mode",
                        env_cfg["mode"],
                        module="Environment",
                        thesis="Alias: env_mode (reset)"
                    )
                except Exception:
                    pass

                # Episode info (safe to publish)
                self.smart_bus.set(
                    "episode_info",
                    {"episode": self.episode_count, "step": self.current_step, "balance": self.market_state.balance, "reset": True},
                    module="Environment",
                    thesis=f"Episode {self.episode_count} reset information",
                )
        except Exception:
            pass

        # Detect provider each reset (hot-reload)
        self._bus_data_active = self._detect_bus_data_active()

        # Publish local market windows only if provider is NOT active
        if not self._bus_data_active:
            self._store_market_data_local()

        try:
            if self.smart_bus and self.prefer_bus_features:
                self.smart_bus.set(
                    "environment_observation_size",
                    int(self.observation_space.shape[0]) if self.observation_space.shape else self._default_obs_size,
                    module="Environment",
                    thesis="Declared observation vector size",
                )
        except Exception:
            pass

        obs = self._get_observation()

        # Publish observation or fallback
        try:
            if self.smart_bus:
                existing = self.smart_bus.get("environment_observation", "Environment") if self.prefer_bus_features else None
                if self.prefer_bus_features and existing is not None:
                    self.smart_bus.set(
                        "environment_observation_fallback",
                        obs,
                        module="Environment",
                        thesis="Fallback environment observation (provider present)",
                    )
                else:
                    self.smart_bus.set(
                        "environment_observation",
                        obs,
                        module="Environment",
                        thesis="Initial environment observation",
                    )
        except Exception:
            pass

        modules = len(self.orchestrator.modules) if (self.orchestrator and hasattr(self.orchestrator, "modules")) else 0
        info = {
            "episode": self.episode_count,
            "step": self.current_step,
            "balance": self.market_state.balance,
            "modules_active": modules,
            "reset": True,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # normalize action
        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float32)
        action = action.astype(np.float32).reshape(self.action_dim,) if self.action_dim > 0 else action.astype(np.float32)

        self.current_step += 1
        self.market_state.current_step = int(self.current_step)

        # publish action & legacy alias
        try:
            if self.smart_bus:
                self.smart_bus.set("agent_action", action, module="Environment", thesis=f"Agent action at step {self.current_step}")
                self.smart_bus.set("final_trading_action", action, module="Environment", thesis="Environment echo of action")
        except Exception:
            pass

        # Refresh environment_config + execution_mode every step (owner refresh to avoid TTL)
        try:
            if self.smart_bus:
                env_cfg = {
                    "instruments": self.instruments,
                    "initial_balance": float(self.config.initial_balance),
                    "action_dim": int(self.action_dim),
                    "max_steps": int(self.config.max_steps),
                    "bus_data_active": bool(self._bus_data_active),
                    "mode": "live" if getattr(self.config, "live_mode", False) else "sim",
                }
                self.smart_bus.set(
                    "environment_config",
                    env_cfg,
                    module="Environment",
                    thesis=f"Environment configuration (step {self.current_step})"
                )
                self.smart_bus.set(
                    "execution_mode",
                    env_cfg["mode"],
                    module="Environment",
                    thesis=f"Execution mode (step {self.current_step})"
                )
                # Legacy alias for consumers expecting env_mode
                try:
                    self.smart_bus.set(
                        "env_mode",
                        env_cfg["mode"],
                        module="Environment",
                        thesis=f"Alias: env_mode (step {self.current_step})"
                    )
                except Exception:
                    pass
        except Exception:
            pass

        # Update market snapshots only if provider isn't active
        if not self._bus_data_active:
            self._store_market_data_local()

        # publish market_state anchors (guarded to avoid duplication)
        try:
            if self.smart_bus:
                if self.smart_bus.get("market_state", "Environment") is None:
                    self.smart_bus.set(
                        "market_state",
                        {
                            "balance": float(self.market_state.balance),
                            "step": self.current_step,
                            "drawdown": float(self.market_state.current_drawdown),
                            "peak_balance": float(self.market_state.peak_balance),
                        },
                        module="Environment",
                        thesis="Current market state (env anchor; not an execution feed)",
                    )
                if self.smart_bus.get("risk_metrics", "Environment") is None:
                    self.smart_bus.set(
                        "risk_metrics",
                        {
                            "balance": float(self.market_state.balance),
                            "equity": float(self.market_state.balance),
                            "current_drawdown": float(self.market_state.current_drawdown),
                        },
                        module="Environment",
                        thesis="Runtime risk metrics (env anchor)",
                    )
                if self.smart_bus.get("performance_data", "Environment") is None:
                    self.smart_bus.set(
                        "performance_data",
                        {
                            "balance": float(self.market_state.balance),
                            "initial_balance": float(self.config.initial_balance),
                            "starting_balance": float(self.config.initial_balance),
                        },
                        module="Environment",
                        thesis="Performance anchors (env anchor)",
                    )
        except Exception:
            pass

        # market_context only if provider isn't active (avoid double publish)
        if not self._bus_data_active:
            try:
                if self.smart_bus and self.instruments:
                    inst = self.instruments[0]
                    tf = "H1" if "H1" in self.data[inst] else list(self.data[inst].keys())[0]
                    df = self.data[inst][tf]
                    s = max(0, self.current_step - 50)
                    e = min(self.current_step, len(df) - 1)
                    window = df["close"].iloc[s:e+1].to_numpy(dtype=np.float64)
                    if window.size >= 2:
                        ret = np.diff(window) / np.maximum(window[:-1], 1e-12)
                        vol = float(np.std(ret))
                        slope = float(np.polyfit(np.arange(window.size), window, 1)[0]) if window.size >= 5 else 0.0
                    else:
                        vol, slope = 0.0, 0.0

                    if vol < 0.003:
                        vol_level = "low"
                    elif vol < 0.01:
                        vol_level = "medium"
                    elif vol < 0.02:
                        vol_level = "high"
                    else:
                        vol_level = "extreme"

                    regime = "trending" if abs(slope) > 0 and vol_level != "low" else "ranging"
                    if vol_level in ("high", "extreme") and abs(slope) < 1e-12:
                        regime = "volatile"

                    self.smart_bus.set(
                        "market_context",
                        {"regime": regime, "volatility_level": vol_level, "consensus": 0.5},
                        module="Environment",
                        thesis=f"Basic regime/vol estimate ({inst}/{tf}) - fallback",
                    )
            except Exception:
                pass

        # Non-blocking orchestrator execution
        if self.orchestrator_enabled and self.orchestrator and hasattr(self.orchestrator, "execute_step"):
            self._run_orchestrator_step({})
            try:
                wait_ms = float(getattr(self.config, "orchestrator_sync_wait_ms", 0.0) or 0.0)
                if wait_ms > 0:
                    time.sleep(min(wait_ms, 200.0) / 1000.0)
            except Exception:
                pass

        # Reward shaping (bus-first)
        reward: Optional[float] = None
        try:
            if self.smart_bus and self.prefer_bus_rewards:
                sr = self.smart_bus.get("shaped_reward", "Environment")
                if isinstance(sr, dict) and "reward" in sr:
                    reward = float(sr["reward"])
                elif isinstance(sr, (int, float, np.floating)):
                    reward = float(sr)
        except Exception:
            pass
        if reward is None:
            reward = 0.0  # no embedded PnL delta here

        # update drawdown anchors locally (balance unchanged here)
        if self.market_state.balance > self.market_state.peak_balance:
            self.market_state.peak_balance = self.market_state.balance
            self.market_state.current_drawdown = 0.0
        else:
            denom = max(self.market_state.peak_balance, 1e-12)
            self.market_state.current_drawdown = (self.market_state.peak_balance - self.market_state.balance) / denom

        # Observation (bus-first consumption; avoid overriding provider output)
        obs = self._get_observation()
        try:
            if self.smart_bus:
                existing = self.smart_bus.get("environment_observation", "Environment") if self.prefer_bus_features else None
                if self.prefer_bus_features and existing is not None:
                    self.smart_bus.set(
                        "environment_observation_fallback",
                        obs,
                        module="Environment",
                        thesis=f"Fallback observation at step {self.current_step}",
                    )
                else:
                    self.smart_bus.set(
                        "environment_observation",
                        obs,
                        module="Environment",
                        thesis=f"Observation at step {self.current_step}",
                    )
        except Exception:
            pass

        terminated, truncated = self._check_termination()
        modules = len(self.orchestrator.modules) if (self.orchestrator and hasattr(self.orchestrator, "modules")) else 0
        info = {
            "step": self.current_step,
            "balance": float(self.market_state.balance),
            "drawdown": float(self.market_state.current_drawdown),
            "reward": float(reward),
            "modules_executed": modules,
            "terminated": terminated,
            "truncated": truncated,
        }
        return obs, float(reward), terminated, truncated, info

    # ──────────────────────────────────────────────────────────────
    # Orchestrator scheduling
    # ──────────────────────────────────────────────────────────────
    def _run_orchestrator_step(self, inputs: Dict[str, Any]) -> None:
        try:
            coro = self.orchestrator.execute_step(inputs)  # type: ignore[attr-defined]
        except Exception as e:
            self.logger.warning(f"Cannot build orchestrator coroutine: {e}")
            return

        loop = self._aio_loop
        if loop and loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                with self._pend_lock:
                    self._pending_futures.add(fut)

                def _on_done(t: Future):
                    try:
                        exc = t.exception()
                        if exc:
                            self.logger.error(f"Orchestrator step failed: {exc}")
                    except Exception:
                        pass
                    finally:
                        with self._pend_lock:
                            self._pending_futures.discard(t)

                fut.add_done_callback(_on_done)
            except Exception as e:
                self.logger.warning(f"Failed to schedule orchestrator step: {e}")
        else:
            threading.Thread(target=lambda: asyncio.run(coro), daemon=True).start()

    def _select_starting_step(self) -> int:
        if not self.instruments:
            return 0
        min_len = min((len(df) for inst in self.instruments for df in self.data[inst].values()), default=0)
        if min_len < 100:
            return 0
        max_start = max(50, int(min_len) - int(self.config.max_steps) - 50)
        if max_start <= 50:
            return 0
        return int(np.random.randint(50, max_start))

    # ──────────────────────────────────────────────────────────────
    # Observation creation
    # ──────────────────────────────────────────────────────────────
    def _get_observation(self) -> np.ndarray:
        expected = self.observation_space.shape[0] if self.observation_space.shape else self._default_obs_size

        # Prefer bus-provided observation if policy says so
        if self.smart_bus and self.prefer_bus_features:
            try:
                obs = self.smart_bus.get("environment_observation", "Environment")
            except Exception:
                obs = None
            if obs is None:
                try:
                    obs = self.smart_bus.get("environment_observation_fallback", "Environment")
                except Exception:
                    obs = None
            if obs is not None:
                if isinstance(obs, np.ndarray):
                    flat = obs.astype(np.float32).flatten()
                elif isinstance(obs, (list, tuple)):
                    flat = np.asarray(obs, dtype=np.float32).flatten()
                else:
                    flat = None

                if flat is not None:
                    if flat.size < expected:
                        out = np.zeros(expected, dtype=np.float32)
                        out[:flat.size] = flat
                        return out
                    return flat[:expected]

        # fallback engineered obs (scale-free features)
        return self._create_fallback_observation(expected)

    def _create_fallback_observation(self, expected_size: int) -> np.ndarray:
        feats: List[float] = []
        feats.extend([
            float(self.market_state.balance) / max(float(self.config.initial_balance), 1e-9),
            float(self.market_state.current_drawdown),
            float(self.current_step) / max(1.0, float(self.config.max_steps)),
        ])

        for instrument in self.instruments:
            for timeframe in ["H1", "H4", "D1"]:
                if timeframe in self.data[instrument]:
                    df = self.data[instrument][timeframe]
                    if self.current_step < len(df):
                        close_ = float(df["close"].iloc[self.current_step])
                        open_ = float(df["open"].iloc[self.current_step])
                        high_ = float(df["high"].iloc[self.current_step])
                        low_ = float(df["low"].iloc[self.current_step])
                        vol_ = float(df["volume"].iloc[self.current_step])

                        s = max(0, self.current_step - 50)
                        m_close = float(np.mean(df["close"].iloc[s:self.current_step+1])) if self.current_step >= s else close_
                        m_vol = float(np.mean(df["volume"].iloc[s:self.current_step+1])) if self.current_step >= s else max(vol_, 1.0)

                        close_rel = (close_ / max(m_close, 1e-12)) - 1.0
                        range_rel = (high_ - low_) / max(abs(close_), 1e-12)
                        change_rel = (close_ - open_) / max(abs(open_), 1e-12)
                        vol_norm = vol_ / max(m_vol, 1e-12)

                        if self.current_step >= 5:
                            prev = float(df["close"].iloc[self.current_step - 5])
                            mom5 = (close_ - prev) / max(abs(prev), 1e-12)
                        else:
                            mom5 = 0.0

                        if self.current_step >= 20:
                            recent = df["close"].iloc[self.current_step - 19:self.current_step + 1].to_numpy(dtype=np.float64)
                            v = float(np.std(recent, dtype=np.float64) / max(abs(float(np.mean(recent, dtype=np.float64))), 1e-12))
                        else:
                            v = 0.01

                        feats.extend([close_rel, range_rel, change_rel, vol_norm, mom5, v])
                    else:
                        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for instrument in self.instruments:
            h1 = h4 = d1 = 0.0
            if "H1" in self.data[instrument] and 5 <= self.current_step < len(self.data[instrument]["H1"]):
                df = self.data[instrument]["H1"]
                cur, past = float(df["close"].iloc[self.current_step]), float(df["close"].iloc[self.current_step - 5])
                h1 = (cur - past) / max(abs(past), 1e-12)
            if "H4" in self.data[instrument] and 5 <= self.current_step < len(self.data[instrument]["H4"]):
                df = self.data[instrument]["H4"]
                cur, past = float(df["close"].iloc[self.current_step]), float(df["close"].iloc[self.current_step - 5])
                h4 = (cur - past) / max(abs(past), 1e-12)
            if "D1" in self.data[instrument] and 5 <= self.current_step < len(self.data[instrument]["D1"]):
                df = self.data[instrument]["D1"]
                cur, past = float(df["close"].iloc[self.current_step]), float(df["close"].iloc[self.current_step - 5])
                d1 = (cur - past) / max(abs(past), 1e-12)
            feats.extend([h1, h4, d1, 1.0 if (h1 > 0 and h4 > 0 and d1 > 0) else 0.0, 1.0 if (h1 < 0 and h4 < 0 and d1 < 0) else 0.0])

        arr = np.asarray(feats, dtype=np.float32)
        if arr.size < expected_size:
            out = np.zeros(expected_size, dtype=np.float32)
            out[:arr.size] = arr
            return out
        return arr[:expected_size]

    # ──────────────────────────────────────────────────────────────
    # Limits & misc
    # ──────────────────────────────────────────────────────────────
    def _check_termination(self) -> Tuple[bool, bool]:
        """Bus-first termination: emergency_mode / bus limits override config where available."""
        # 1) Emergency mode (if any module raised it)
        if self.smart_bus and self.halt_on_emergency:
            try:
                em = self.smart_bus.get("emergency_mode", "Environment")
                if isinstance(em, dict):
                    if bool(em.get("halt", False)) or bool(em.get("active", False)):
                        return True, False
                elif isinstance(em, (int, float)) and em:  # non-zero interpreted as active
                    return True, False
                elif isinstance(em, bool) and em:
                    return True, False
            except Exception:
                pass

        # 2) Bus-provided limits (e.g., ComplianceModule / PortfolioRiskSystem)
        bus_max_dd = None
        if self.smart_bus and self.prefer_bus_limits:
            try:
                rl = self.smart_bus.get("risk_limits", "Environment")
                if isinstance(rl, dict):
                    if "max_drawdown" in rl:
                        bus_max_dd = float(rl["max_drawdown"])
                    elif "max_drawdown_pct" in rl:
                        bus_max_dd = float(rl["max_drawdown_pct"])
            except Exception:
                pass
            if bus_max_dd is None:
                try:
                    comp = self.smart_bus.get("compliance", "Environment")
                    if isinstance(comp, dict):
                        md = comp.get("limits") or comp
                        if isinstance(md, dict):
                            if "max_drawdown" in md:
                                bus_max_dd = float(md["max_drawdown"])
                            elif "max_drawdown_pct" in md:
                                bus_max_dd = float(md["max_drawdown_pct"])
                except Exception:
                    pass

        dd_limit = float(bus_max_dd) if bus_max_dd is not None else float(self.config.max_drawdown)

        # 3) Apply limits and step bounds
        if self.market_state.current_drawdown > dd_limit:
            return True, False
        if int(self.current_step) >= int(self.config.max_steps):
            return False, True
        if self.market_state.balance <= 0:
            return True, False
        return False, False

    # ──────────────────────────────────────────────────────────────
    # Fallback SmartBus + post-init monitor
    # ──────────────────────────────────────────────────────────────
    def _create_fallback_smart_bus(self):
        class FallbackSmartBus:
            def __init__(self):
                self._store = {}
                self._lock = threading.Lock()
                self._module_disabled = set()
                self._data_store = self._store  # for external status probes
                self._is_fallback = True

            def set(self, key, value, module=None, thesis=None):
                with self._lock:
                    self._store[key] = value

            def get(self, key, module=None, default=None):
                with self._lock:
                    return self._store.get(key, default)

            def register_provider(self, module, keys): return True
            def register_consumer(self, module, keys): return True

            def get_performance_metrics(self):
                with self._lock:
                    return {
                        "active": True,
                        "data_keys": len(self._store),
                        "disabled_modules": list(self._module_disabled),
                    }
        return FallbackSmartBus()

    def _start_post_init_monitor(self):
        def monitor():
            try:
                if not self._bus_ready.is_set():
                    self._bus_ready.wait(timeout=max(1.0, float(getattr(self.config, "info_bus_init_timeout", 2.0)) * 5))
                if self._bus_ready.is_set() and getattr(self.smart_bus, "_is_fallback", False):
                    try:
                        real_bus = InfoBusManager.get_instance() if (SMARTINFOBUS_AVAILABLE and InfoBusManager) else None
                        if real_bus is not None:
                            self.smart_bus = real_bus
                            self.logger.info("SmartInfoBus is ready - switched from fallback to real bus")
                    except Exception as e:
                        self.logger.warning(f"Failed switching to real SmartInfoBus: {e}")

                if not self._orch_ready.is_set():
                    wait_time = float(getattr(self.config, "orchestrator_init_timeout", 2.0)) * (3 if bool(getattr(self.config, "orchestrator_async_init", True)) else 1)
                    self._orch_ready.wait(timeout=max(2.0, wait_time))

                if self._orch_ready.is_set() and self.orchestrator and not self.orchestrator_enabled:
                    self.orchestrator_enabled = True
                    self.logger.info("ModuleOrchestrator is ready - enabling orchestrator execution")
            except Exception as e:
                self.logger.warning(f"Post-init monitor error: {e}")

        threading.Thread(target=monitor, daemon=True).start()

    # ──────────────────────────────────────────────────────────────
    # Diagnostics & rendering
    # ──────────────────────────────────────────────────────────────
    def get_smartinfobus_status(self) -> Dict[str, Any]:
        modules_active = len(self.orchestrator.modules) if self.orchestrator and hasattr(self.orchestrator, "modules") else 0
        bus = self.smart_bus
        data_keys = len(getattr(bus, "_data_store", {})) if bus else 0
        disabled = list(getattr(bus, "_module_disabled", [])) if bus else []
        try:
            metrics = bus.get_performance_metrics() if bus else {}
        except Exception:
            metrics = {}
        return {
            "performance_metrics": metrics,
            "modules_active": modules_active,
            "modules_disabled": disabled,
            "data_keys": data_keys,
            "current_step": self.current_step,
            "episode": self.episode_count,
        }

    def render(self, mode: str = "human"):
        if mode == "human":
            print(
                f"Step: {self.current_step}, "
                f"Balance: €{self.market_state.balance:.2f}, "
                f"Drawdown: {self.market_state.current_drawdown:.1%}"
            )

    def close(self):
        self.logger.info("Environment closed")
        try:
            self.orchestrator_enabled = False
        except Exception:
            pass

        # IMPORTANT: Do not shutdown the global orchestrator singleton here.
        # Multiple environments may share the same ModuleOrchestrator via get_instance().
        # Shutting it down here clears module registries and causes KeyError on next step
        # (e.g., 'SessionManager' missing). If a full process shutdown is required, call
        # ModuleOrchestrator.get_instance().shutdown() explicitly from the top-level runner.
        # Keeping the orchestrator alive avoids repeated boot sequences and preserves state.

        loop = self._aio_loop
        try:
            if loop and loop.is_running():
                with self._pend_lock:
                    to_cancel = list(self._pending_futures)
                for fut in to_cancel:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                try:
                    tick = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop)
                    tick.result(timeout=0.5)
                except Exception:
                    pass
                loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass

        try:
            if self._aio_thread and self._aio_thread.is_alive():
                self._aio_thread.join(timeout=1.5)
        except Exception:
            pass

        try:
            with self._pend_lock:
                self._pending_futures.clear()
        except Exception:
            pass
        self._aio_loop = None
        self._aio_thread = None

        # Do NOT touch order_queue or any execution feeds on close
        self.smart_bus = None

    # ──────────────────────────────────────────────────────────────
    # Legacy helpers (SB3 compatibility)
    # ──────────────────────────────────────────────────────────────
    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    @property
    def unwrapped(self):
        return self


__all__ = ["ModernTradingEnv"]
