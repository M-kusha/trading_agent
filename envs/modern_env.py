# envs/modern_env.py
"""
Modern SmartInfoBus Trading Environment (Unified, Bus-First)

Key ideas:
- Bus-first: the env consumes market data, features, rewards & limits from SmartInfoBus
  when present; it falls back to local logic only when needed.
- No duplication: if MarketDataProvider (or any other module) is publishing market_data,
  the env avoids republishing those keys.
- Execution is external: another module owns positions/trades/fills/portfolio metrics.
  This env does NOT publish execution or portfolio keys.
- Soft coupling: everything bus-related is optional and guarded; the env works offline.

Gymnasium v0.26+ API (step returns obs, reward, terminated, truncated, info).
"""

from __future__ import annotations

import copy
import warnings
import asyncio
import threading
import time
from typing import Any, Dict, Optional, Tuple, List, Set, cast, TYPE_CHECKING

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

# Unified PPO observation builder (v4.0)
# Ensures training (SB3 PPO) and live (PPOAgentShell) use identical observation schemas
# The 64-dim observation includes market, account, risk, consensus, world model, and trading mode signals
try:
    from modules.meta.ppo_observation_builder import (
        PPOObservationBuilder,
        PPO_OBS_SIZE,
        PPO_OBS_VERSION,
        get_ppo_observation_builder,
    )
    PPO_OBS_BUILDER_AVAILABLE = True
except ImportError:
    PPOObservationBuilder = None  # type: ignore
    PPO_OBS_SIZE = 64  # Must match modules.meta.ppo_observation_builder v4.0
    PPO_OBS_VERSION = "4.0"
    get_ppo_observation_builder = None  # type: ignore
    PPO_OBS_BUILDER_AVAILABLE = False

if TYPE_CHECKING:
    # Expose the builder type to static type checkers without importing at runtime
    from modules.meta.ppo_observation_builder import PPOObservationBuilder  # type: ignore

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

        # Observation size: use PPO_OBS_SIZE (64) for unified training/live schema (v4.0)
        # config.environment_observation_size is legacy; PPO_OBS_SIZE takes precedence
        self._default_obs_size = PPO_OBS_SIZE  # 64 dims (unified PPO schema v4.0)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._default_obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_dim = int(np.prod(self.action_space.shape)) if self.action_space.shape else 0

        # Unified PPO observation builder (v4.0)
        # Observation builder instance (may be None if builder unavailable)
        self.obs_builder = None
        if PPO_OBS_BUILDER_AVAILABLE and get_ppo_observation_builder is not None:
            self.obs_builder = get_ppo_observation_builder()

        # Execution config (fallbacks — modules may override via bus)
        # M15 is the primary trading/decision timeframe; uses config.primary_timeframe which defaults to M15
        self.primary_timeframe = getattr(self.config, "primary_timeframe", "M15") or "M15"
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
        # Backpressure controls (defaults guarded via config where available)
        try:
            self._orch_inflight_limit = int(getattr(self.config, "orchestrator_max_inflight", 1) or 1)
        except Exception:
            self._orch_inflight_limit = 1
        try:
            self._orch_step_interval = int(getattr(self.config, "orchestrator_step_interval", 1) or 1)
        except Exception:
            self._orch_step_interval = 1

        # Bring systems up
        self._initialize_systems()
        self._start_event_loop_thread()

        # Data (fallback local data)
        self.orig_data = data_dict
        self.data = copy.deepcopy(data_dict)
        self.instruments: List[str] = list(self.data.keys())
        self._validate_data()

        # Track minimum available data length across all instruments/timeframes
        try:
            data_lengths: Dict[str, int] = {}
            for inst in self.instruments:
                for tf, df in self.data[inst].items():
                    data_lengths[f"{inst}/{tf}"] = len(df)

            self._min_data_len = min(data_lengths.values()) if data_lengths else 0

            # Log data inventory for diagnostics
            if data_lengths:
                self.logger.info(f"📊 DATA_INVENTORY: {len(self.instruments)} instruments")
                for key, length in sorted(data_lengths.items()):
                    self.logger.info(f"  - {key}: {length:,} bars")
                self.logger.info(f"  ➜ Minimum length: {self._min_data_len:,} bars")

                # Warn if data is critically short
                if self._min_data_len < 50:
                    self.logger.warning(
                        f"⚠️ CRITICAL: Minimum data length is only {self._min_data_len} bars! "
                        f"Episodes will end after ~{self._min_data_len - 1} steps. "
                        f"This severely limits training effectiveness."
                    )
        except Exception as e:
            self.logger.error(f"Failed to compute data lengths: {e}")
            self._min_data_len = 0

        if not self.instruments:
            raise ValueError("No instruments provided to ModernTradingEnv")
        if self.primary_timeframe not in self.data[self.instruments[0]]:
            self.primary_timeframe = list(self.data[self.instruments[0]].keys())[0]

        # Market/account anchors (env-internal; env no longer publishes execution or portfolio keys)
        initial_balance = float(self.config.initial_balance)
        self.market_state = MarketState(
            balance=initial_balance,
            peak_balance=initial_balance,
            current_step=0,
            current_drawdown=0.0,
        )
        self.balance: float = float(initial_balance)
        self.equity: float = float(initial_balance)
        self._last_equity: float = self.equity

        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        self.episode_metrics = EpisodeMetrics()

        # Finalize spaces
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2 * len(self.instruments),),
            dtype=np.float32,
        )
        self.action_dim = int(np.prod(self.action_space.shape)) if self.action_space.shape else 0
        self.observation_space = self._get_observation_space()

        # Detect whether provider is active on the bus
        self._bus_data_active = self._detect_bus_data_active()

        # Setup env context
        self._setup_environment()

        # Only capture local market windows if provider is NOT active
        if not self._bus_data_active:
            self._store_market_data_local()

        # CRITICAL FIX: If data is too short for meaningful training, raise an error
        min_required_bars = int(getattr(self.config, "min_required_data_bars", 50))
        if self._min_data_len < min_required_bars:
            raise ValueError(
                f"⚠️ INSUFFICIENT DATA: Minimum data length is {self._min_data_len} bars, "
                f"but at least {min_required_bars} bars are required for meaningful training. "
                f"\n\nPossible causes:"
                f"\n  1. CSV files in data/processed/ have very few rows"
                f"\n  2. MarketDataProvider is publishing small windows instead of full history"
                f"\n  3. Data files are corrupted or improperly formatted"
                f"\n\nSolutions:"
                f"\n  1. Check your data files and ensure they have sufficient historical data"
                f"\n  2. Set config.min_required_data_bars to a lower value (not recommended)"
                f"\n  3. Disable bus data: set prefer_bus_data=False in config"
            )

        modules = len(self.orchestrator.modules) if (self.orchestrator and hasattr(self.orchestrator, "modules")) else 0
        self.logger.info(
            f"🚀 MODERN_ENV_INITIALIZED: {len(self.instruments)} instruments, {modules} modules - Bus-first"
        )

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
                env_cfg = {
                    "instruments": self.instruments,
                    "initial_balance": float(self.config.initial_balance),
                    "action_dim": int(self.action_dim),
                    "max_steps": int(self.config.max_steps),
                    "bus_data_active": bool(self._bus_data_active),
                    "mode": "live" if getattr(self.config, "live_mode", False) else "sim",
                }
                try:
                    self._env_environment_config = env_cfg
                except Exception:
                    pass
        except Exception:
            pass

    def _store_market_data_local(self):
        """
        Capture local market windows ONLY when MarketDataProvider is not active.

        This is strictly for diagnostics / fallback; we do NOT publish provider-owned
        keys like 'market_data' to the bus from the Environment.
        """
        if not self.smart_bus or self._bus_data_active:
            return

        step = int(self.current_step)
        aggregated: Dict[str, Dict[str, Any]] = {}
        for instrument in self.instruments:
            for timeframe in ["M15", "H1", "H4", "D1"]:
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
                        "open": df["open"].iloc[s : step + 1].values,
                        "high": df["high"].iloc[s : step + 1].values,
                        "low": df["low"].iloc[s : step + 1].values,
                        "close": df["close"].iloc[s : step + 1].values,
                        "volume": df["volume"].iloc[s : step + 1].values,
                        "step": step,
                        "instrument": instrument,
                        "timeframe": timeframe,
                    }
                    aggregated.setdefault(instrument, {})[timeframe] = ohlcv
                except Exception:
                    pass

        # Store fallback locally for diagnostics; do not publish provider-owned keys to the bus
        try:
            self._env_fallback_market_data = aggregated
            self._env_fallback_step = step
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
        # PERF: Don't deep copy data - it's read-only during training
        # Deep copy of 60k+ rows per instrument was causing multi-minute stalls on reset
        # self.data = copy.deepcopy(self.orig_data)  # OLD: very slow!
        self.data = self.orig_data  # NEW: reference only (data is read-only)

        # Recompute minimum data length on each reset (in case of hot-reload)
        try:
            self._min_data_len = min(
                (len(df) for inst in self.instruments for df in self.data[inst].values()),
                default=0,
            )
        except Exception:
            self._min_data_len = 0

        initial_balance = float(self.config.initial_balance)
        self.market_state = MarketState(
            balance=initial_balance,
            peak_balance=initial_balance,
            current_step=self._select_starting_step(),
            current_drawdown=0.0,
        )
        self.current_step = int(self.market_state.current_step)

        self.balance = float(initial_balance)
        self.equity = float(initial_balance)
        self._last_equity = float(initial_balance)

        # Detect provider each reset (hot-reload)
        self._bus_data_active = self._detect_bus_data_active()

        # Capture local market windows only if provider is NOT active
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
                existing = (
                    self.smart_bus.get("environment_observation", "Environment")
                    if self.prefer_bus_features
                    else None
                )
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

    def _cleanup_pending_futures(self):
        """Remove completed futures from tracking to prevent memory accumulation"""
        try:
            with self._pend_lock:
                completed = {f for f in self._pending_futures if f.done()}
                self._pending_futures -= completed

                pending_count = len(self._pending_futures)
                if pending_count > 10:
                    self.logger.warning(
                        f"[PERF] Pending futures accumulating: {pending_count} futures still pending"
                    )
        except Exception as e:
            self.logger.error(f"Failed to cleanup pending futures: {e}")

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # normalize action
        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float32)
        action = (
            action.astype(np.float32).reshape(self.action_dim,)
            if self.action_dim > 0
            else action.astype(np.float32)
        )

        self.current_step += 1
        self.market_state.current_step = int(self.current_step)

        # Cleanup pending futures every 5 steps to prevent accumulation
        if self.current_step % 5 == 0:
            self._cleanup_pending_futures()

        # ═══════════════════════════════════════════════════════════════
        # AUTONOMOUS PPO ACTION INTERPRETATION (v4.1)
        # ═══════════════════════════════════════════════════════════════
        # Interpret PPO action semantics:
        #   action[0] = direction_score ∈ [-1, 1]
        #   action[1] = size_score ∈ [-1, 1]
        #
        # For autonomous training, PPO's direction is derived directly from
        # direction_score without expert blending.
        # ═══════════════════════════════════════════════════════════════
        
        # Get per-instrument actions (action array has 2 dims per instrument)
        ppo_direction = "flat"
        ppo_direction_score = 0.0
        ppo_size_score = 0.0
        ppo_confidence = 0.0
        
        if action is not None and len(action) >= 2:
            # For first instrument (or global action)
            ppo_direction_score = float(action[0])
            ppo_size_score = float(action[1])
            ppo_confidence = float(np.clip(abs(ppo_direction_score), 0.0, 1.0))
            
            # Interpret direction using configurable thresholds
            long_th = float(getattr(self.config, "direction_long_threshold", 0.3))
            short_th = float(getattr(self.config, "direction_short_threshold", -0.3))
            
            if ppo_direction_score > long_th:
                ppo_direction = "long"
            elif ppo_direction_score < short_th:
                ppo_direction = "short"
            else:
                ppo_direction = "flat"
        
        # Compute position size from size_score: [-1,1] → [0,1]
        raw_position_size = (ppo_size_score + 1.0) / 2.0

        # publish action & legacy alias (throttled for training speed)
        # Only publish every 10 steps to reduce bus overhead
        should_publish = (self.current_step % 10 == 0)
        try:
            if self.smart_bus and should_publish:
                self.smart_bus.set(
                    "agent_action",
                    action,
                    module="Environment",
                    thesis=f"Agent action at step {self.current_step}",
                )
                self.smart_bus.set(
                    "final_trading_action",
                    action,
                    module="Environment",
                    thesis="Environment echo of action",
                )
                
                # Publish step_idx for cooldown tracking across modules
                self.smart_bus.set(
                    "step_idx",
                    int(self.current_step),
                    module="Environment",
                    thesis=f"Current simulation step: {self.current_step}",
                )
                
                # Publish interpreted PPO decision for autonomous training
                # This is the PPO's direct intent, before any expert blending
                self.smart_bus.set(
                    "ppo_autonomous_decision",
                    {
                        "direction": ppo_direction,
                        "direction_score": ppo_direction_score,
                        "size_score": ppo_size_score,
                        "raw_position_size": raw_position_size,
                        "confidence": ppo_confidence,
                        "step": self.current_step,
                        "autonomous_training": bool(getattr(self.config, "ppo_autonomous_training", True)),
                    },
                    module="Environment",
                    thesis=f"PPO autonomous direction: {ppo_direction} (score={ppo_direction_score:.2f})",
                )
        except Exception:
            pass

        # Update local market snapshots only if provider isn't active (throttled)
        if not self._bus_data_active and should_publish:
            self._store_market_data_local()

        # publish market_state anchors (guarded to avoid duplication, only on publish steps)
        try:
            if self.smart_bus and should_publish:
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
        # Throttled: only compute expensive regime detection every 50 steps
        if not self._bus_data_active and (self.current_step % 50 == 0):
            try:
                if self.smart_bus and self.instruments:
                    inst = self.instruments[0]
                    tf = "H1" if "H1" in self.data[inst] else list(self.data[inst].keys())[0]
                    df = self.data[inst][tf]
                    s = max(0, self.current_step - 50)
                    e = min(self.current_step, len(df) - 1)
                    window = df["close"].iloc[s : e + 1].to_numpy(dtype=np.float64)
                    if window.size >= 2:
                        ret = np.diff(window) / np.maximum(window[:-1], 1e-12)
                        vol = float(np.std(ret))
                        slope = (
                            float(np.polyfit(np.arange(window.size), window, 1)[0])
                            if window.size >= 5
                            else 0.0
                        )
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

                    try:
                        self._env_market_context = {
                            "regime": regime,
                            "volatility_level": vol_level,
                            "consensus": 0.5,
                        }
                    except Exception:
                        pass
            except Exception:
                pass

        # Non-blocking orchestrator execution (with simple backpressure)
        if self.orchestrator_enabled and self.orchestrator and hasattr(self.orchestrator, "execute_step"):
            try:
                interval = int(
                    getattr(self.config, "orchestrator_step_interval", self._orch_step_interval)
                    or self._orch_step_interval
                )
            except Exception:
                interval = self._orch_step_interval

            should_fire_this_step = (interval <= 1) or (self.current_step % max(1, interval) == 0)

            can_schedule = False
            if should_fire_this_step:
                try:
                    with self._pend_lock:
                        inflight = len(self._pending_futures)
                    limit = max(
                        1,
                        int(
                            getattr(self.config, "orchestrator_max_inflight", self._orch_inflight_limit)
                            or self._orch_inflight_limit
                        ),
                    )
                    can_schedule = inflight < limit
                except Exception:
                    can_schedule = True  # be permissive if check fails

            if can_schedule:
                self._run_orchestrator_step({})
                try:
                    wait_ms = float(getattr(self.config, "orchestrator_sync_wait_ms", 0.0) or 0.0)
                    if wait_ms > 0:
                        time.sleep(min(wait_ms, 200.0) / 1000.0)
                except Exception:
                    pass

        else:
            # Orchestrator not enabled; optionally throttle step speed for stability
            try:
                ss_ms = float(getattr(self.config, "step_sleep_ms", 0.0) or 0.0)
                if ss_ms > 0:
                    time.sleep(min(200.0, ss_ms) / 1000.0)
            except Exception:
                pass

        # Sync balance/equity from Executor's account_state after orchestrator runs
        try:
            if self.smart_bus:
                account_state = self.smart_bus.get("account_state", "Environment", default=None)
                if isinstance(account_state, dict):
                    new_balance = account_state.get("balance")
                    new_equity = account_state.get("equity")
                    if new_balance is not None:
                        self.market_state.balance = float(new_balance)
                        self.balance = float(new_balance)
                    if new_equity is not None:
                        self.equity = float(new_equity)
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

        # Fallback: simple PnL-based reward (always provide some learning signal)
        if reward is None:
            try:
                current_balance = float(self.market_state.balance)
                prev_equity = float(getattr(self, "_last_equity", current_balance))
                pnl_delta = current_balance - prev_equity
                initial = float(getattr(self.config, "initial_balance", 3000.0) or 3000.0)
                reward = pnl_delta / max(initial, 1.0) * 10.0
                reward = float(np.clip(reward, -1.0, 1.0))
                self._last_equity = current_balance
            except Exception:
                reward = 0.0

        # update drawdown anchors locally (balance unchanged here)
        if self.market_state.balance > self.market_state.peak_balance:
            self.market_state.peak_balance = self.market_state.balance
            self.market_state.current_drawdown = 0.0
        else:
            denom = max(self.market_state.peak_balance, 1e-12)
            self.market_state.current_drawdown = (
                self.market_state.peak_balance - self.market_state.balance
            ) / denom

        # Observation (bus-first consumption; avoid overriding provider output)
        obs = self._get_observation()
        try:
            if self.smart_bus:
                existing = (
                    self.smart_bus.get("environment_observation", "Environment")
                    if self.prefer_bus_features
                    else None
                )
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
        if terminated or truncated:
            try:
                reason = "unknown"
                if int(self.current_step) >= int(self.config.max_steps):
                    reason = "max_steps"
                elif self._min_data_len and int(self.current_step) >= int(self._min_data_len) - 1:
                    reason = "data_end"
                elif float(self.market_state.balance) <= 0.0:
                    reason = "bankrupt"
                else:
                    reason = "drawdown_or_limit"
                self.logger.info(
                    f"[EPISODE_END] episode={self.episode_count} steps={self.current_step} reason={reason}"
                )
            except Exception:
                pass
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
        min_len = min(
            (len(df) for inst in self.instruments for df in self.data[inst].values()),
            default=0,
        )
        if min_len < 100:
            return 0
        max_start = max(50, int(min_len) - int(self.config.max_steps) - 50)
        if max_start <= 50:
            return 0
        return int(np.random.randint(50, max_start))

    # ──────────────────────────────────────────────────────────────
    # Observation creation (Unified PPO schema, bus-first)
    # ──────────────────────────────────────────────────────────────
    def _get_observation(self) -> np.ndarray:
        """
        Build observation using the unified PPO observation builder (v3.0).

        Bus-first semantics:
        - If MarketDataProvider is active on SmartInfoBus AND prefer_bus_data=True,
          let PPOObservationBuilder pull OHLC/multi-timeframe data directly from
          the bus (exactly like PPOAgentShell in live trading).
        - Otherwise, fall back to local CSV data via _prepare_market_data_for_obs().

        The same 48-dim observation schema is used in both:
        - Training (SB3 PPO via ModernTradingEnv)
        - Live trading (PPOAgentShell via SmartInfoBus)
        """
        expected = (
            self.observation_space.shape[0]
            if self.observation_space.shape
            else self._default_obs_size
        )

        # If unified observation builder is available, use it
        if self.obs_builder is not None:
            try:
                use_bus_market_data = (
                    self.smart_bus_enabled
                    and self.smart_bus is not None
                    and bool(getattr(self, "_bus_data_active", False))
                    and bool(getattr(self, "prefer_bus_data", True))
                )

                if use_bus_market_data:
                    # MarketDataProvider owns OHLC/multi-TF data; builder will read from bus
                    market_data = None
                else:
                    # Fallback: build market_data from local CSVs
                    market_data = self._prepare_market_data_for_obs()

                # Prepare account state (env anchor; Executor may override via bus)
                account_state = {
                    "balance": float(self.market_state.balance),
                    "initial_balance": float(self.config.initial_balance),
                    "current_drawdown": float(self.market_state.current_drawdown),
                    "current_step": int(self.current_step),
                    "max_steps": int(self.config.max_steps),
                    # The following are placeholders because env does not own execution.
                    # PPOObservationBuilder is designed to tolerate these defaults and will
                    # prefer richer account_state from SmartInfoBus if available.
                    "episode_return": 0.0,
                    "position_direction": 0.0,
                    "position_size": 0.0,
                    "unrealized_pnl": 0.0,
                    "time_in_position": 0,
                    "trades_today": 0,
                    "last_action": 0.0,
                    "win_rate": 0.5,
                    "pnl_trend": 0.0,
                }

                obs = self.obs_builder.build(
                    market_data=market_data,
                    account_state=account_state,
                    smart_bus=self.smart_bus if self.smart_bus_enabled else None,
                    module_name="Environment",
                )

                # Shape normalization: always end up with a flat float32 vector
                if isinstance(obs, np.ndarray):
                    flat = obs.astype(np.float32).flatten()
                else:
                    flat = np.asarray(obs, dtype=np.float32).flatten()

                if flat.size < expected:
                    out = np.zeros(expected, dtype=np.float32)
                    out[: flat.size] = flat
                    return out
                return flat[:expected]

            except Exception as e:
                self.logger.warning(
                    f"[ENV] Unified obs builder failed: {e}, using fallback observation"
                )

        # Fallback: try bus-provided observation
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
                        out[: flat.size] = flat
                        return out
                    return flat[:expected]

        # Legacy numeric fallback (only if builder + bus are unavailable)
        return self._create_fallback_observation(expected)

    def _prepare_market_data_for_obs(self) -> Dict[str, Any]:
        """
        Prepare market data from local DataFrame for the observation builder.

        Returns data structured as: {symbol: {timeframe: {open, high, low, close, volume}}}
        """
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for instrument in self.instruments:
            result[instrument] = {}
            for timeframe in ["M15", "H1", "H4", "D1"]:  # M15 first (primary)
                if timeframe not in self.data[instrument]:
                    continue
                df = self.data[instrument][timeframe]
                if self.current_step >= len(df):
                    continue

                # Get rolling window (up to 100 bars)
                lookback = min(100, self.current_step + 1)
                start_idx = max(0, self.current_step - lookback + 1)
                end_idx = self.current_step + 1

                result[instrument][timeframe] = {
                    "open": df["open"].iloc[start_idx:end_idx].values,
                    "high": df["high"].iloc[start_idx:end_idx].values,
                    "low": df["low"].iloc[start_idx:end_idx].values,
                    "close": df["close"].iloc[start_idx:end_idx].values,
                    "volume": df["volume"].iloc[start_idx:end_idx].values,
                }

        return result

    def _create_fallback_observation(self, expected_size: int) -> np.ndarray:
        feats: List[float] = []
        feats.extend(
            [
                float(self.market_state.balance) / max(float(self.config.initial_balance), 1e-9),
                float(self.market_state.current_drawdown),
                float(self.current_step) / max(1.0, float(self.config.max_steps)),
            ]
        )

        for instrument in self.instruments:
            for timeframe in ["M15", "H1", "H4", "D1"]:
                if timeframe in self.data[instrument]:
                    df = self.data[instrument][timeframe]
                    if self.current_step < len(df):
                        close_ = float(df["close"].iloc[self.current_step])
                        open_ = float(df["open"].iloc[self.current_step])
                        high_ = float(df["high"].iloc[self.current_step])
                        low_ = float(df["low"].iloc[self.current_step])
                        vol_ = float(df["volume"].iloc[self.current_step])

                        s = max(0, self.current_step - 50)
                        m_close = (
                            float(np.mean(df["close"].iloc[s : self.current_step + 1]))
                            if self.current_step >= s
                            else close_
                        )
                        m_vol = (
                            float(np.mean(df["volume"].iloc[s : self.current_step + 1]))
                            if self.current_step >= s
                            else max(vol_, 1.0)
                        )

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
                            recent = df["close"].iloc[self.current_step - 19 : self.current_step + 1].to_numpy(
                                dtype=np.float64
                            )
                            v = float(
                                np.std(recent, dtype=np.float64)
                                / max(abs(float(np.mean(recent, dtype=np.float64))), 1e-12)
                            )
                        else:
                            v = 0.01

                        feats.extend([close_rel, range_rel, change_rel, vol_norm, mom5, v])
                    else:
                        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for instrument in self.instruments:
            m15 = h1 = h4 = d1 = 0.0
            if "M15" in self.data[instrument] and 5 <= self.current_step < len(self.data[instrument]["M15"]):
                df = self.data[instrument]["M15"]
                cur, past = float(df["close"].iloc[self.current_step]), float(
                    df["close"].iloc[self.current_step - 5]
                )
                m15 = (cur - past) / max(abs(past), 1e-12)
            if "H1" in self.data[instrument] and 5 <= self.current_step < len(self.data[instrument]["H1"]):
                df = self.data[instrument]["H1"]
                cur, past = float(df["close"].iloc[self.current_step]), float(
                    df["close"].iloc[self.current_step - 5]
                )
                h1 = (cur - past) / max(abs(past), 1e-12)
            if "H4" in self.data[instrument] and 5 <= self.current_step < len(self.data[instrument]["H4"]):
                df = self.data[instrument]["H4"]
                cur, past = float(df["close"].iloc[self.current_step]), float(
                    df["close"].iloc[self.current_step - 5]
                )
                h4 = (cur - past) / max(abs(past), 1e-12)
            if "D1" in self.data[instrument] and 5 <= self.current_step < len(self.data[instrument]["D1"]):
                df = self.data[instrument]["D1"]
                cur, past = float(df["close"].iloc[self.current_step]), float(
                    df["close"].iloc[self.current_step - 5]
                )
                d1 = (cur - past) / max(abs(past), 1e-12)
            feats.extend(
                [
                    m15,
                    h1,
                    h4,
                    d1,
                    1.0 if (m15 > 0 and h1 > 0 and h4 > 0 and d1 > 0) else 0.0,
                    1.0 if (m15 < 0 and h1 < 0 and h4 < 0 and d1 < 0) else 0.0,
                ]
            )

        # Add prop firm features (AI needs to see how close to limits)
        prop_firm_feats = self._get_prop_firm_observation_features()
        feats.extend(prop_firm_feats)

        # Add memory features (AI learns from memory signals)
        memory_feats = self._get_memory_observation_features()
        feats.extend(memory_feats)

        arr = np.asarray(feats, dtype=np.float32)
        if arr.size < expected_size:
            out = np.zeros(expected_size, dtype=np.float32)
            out[: arr.size] = arr
            return out
        return arr[:expected_size]

    def _get_prop_firm_observation_features(self) -> List[float]:
        """
        Get prop firm-related features for observation.

        These help the AI understand:
        1. How close to daily DD limit (0-1, 1=at limit)
        2. How close to max DD limit (0-1, 1=at limit)
        3. Profit progress toward target (0-1, 1=target reached)
        4. Can trade flag (0 or 1)
        """
        daily_dd_ratio = 0.0
        max_dd_ratio = 0.0
        profit_progress = 0.0
        can_trade = 1.0

        try:
            if self.smart_bus:
                prop_status = self.smart_bus.get("prop_firm_status", "Environment")
                if isinstance(prop_status, dict):
                    daily_limit = prop_status.get("daily_dd_remaining", 0.05) + prop_status.get(
                        "daily_dd_used", 0.0
                    )
                    if daily_limit > 0:
                        daily_dd_ratio = prop_status.get("daily_dd_used", 0.0) / daily_limit

                    max_limit = prop_status.get("max_dd_remaining", 0.10) + prop_status.get(
                        "max_dd_used", 0.0
                    )
                    if max_limit > 0:
                        max_dd_ratio = prop_status.get("max_dd_used", 0.0) / max_limit

                    can_trade = 1.0 if prop_status.get("can_trade", True) else 0.0

            # Calculate profit progress from balance
            initial = float(self.config.initial_balance)
            current = float(self.market_state.balance)
            profit_target = float(getattr(self.config, "profit_target", 0.10))

            if initial > 0 and profit_target > 0:
                current_profit = (current - initial) / initial
                profit_progress = max(0.0, current_profit / profit_target)
        except Exception:
            pass

        return [
            float(np.clip(daily_dd_ratio, 0.0, 1.5)),  # Daily DD ratio (allow >1 to show breach)
            float(np.clip(max_dd_ratio, 0.0, 1.5)),  # Max DD ratio
            float(np.clip(profit_progress, 0.0, 2.0)),  # Profit progress (allow >1 for over-target)
            can_trade,  # Can trade flag
        ]

    def _get_memory_observation_features(self) -> List[float]:
        """
        Get memory-related features for observation.

        These help the AI learn from memory signals:
        1. risk_multiplier: 0-1, lower = memory thinks setup is riskier
        2. danger_similarity: 0-1, how similar to past losing trades
        3. loss_prob: 0-1, neural network's P(loss) prediction
        4. veto_active: 0 or 1, whether memory wants to block
        5. signed_bias: -1 to 1, playbook's directional recommendation
        6. playbook_confidence: 0-1, confidence in playbook recall
        7. consecutive_losses_norm: 0-1, normalized loss streak (0=none, 1=5+)
        8. neural_risk_hint: 0-1, attention-based risk estimate

        NOTE: The hard veto is still applied in PPO arbiter - these features
        let the AI LEARN from memory, but memory can still override if certain.
        """
        risk_multiplier = 1.0
        danger_similarity = 0.0
        loss_prob = 0.0
        veto_active = 0.0
        signed_bias = 0.0
        playbook_confidence = 0.5
        consecutive_losses_norm = 0.0
        neural_risk_hint = 0.5

        try:
            if self.smart_bus:
                memory_gate = self.smart_bus.get("memory_gate", "Environment")
                if isinstance(memory_gate, dict):
                    risk_multiplier = float(memory_gate.get("risk_multiplier", 1.0))
                    danger_similarity = float(memory_gate.get("danger_similarity", 0.0))
                    loss_prob = float(memory_gate.get("loss_prob", 0.0))
                    veto_active = 1.0 if memory_gate.get("veto", False) else 0.0

                    vetoed = memory_gate.get("vetoed_instruments", [])
                    if isinstance(vetoed, list) and len(vetoed) > 0:
                        veto_active = max(veto_active, len(vetoed) / 2.0)

                    consec = memory_gate.get("consecutive_losses_by_instrument", {})
                    if isinstance(consec, dict) and consec:
                        max_streak = max(consec.values()) if consec.values() else 0
                        consecutive_losses_norm = min(1.0, max_streak / 5.0)

                memory_vote = self.smart_bus.get("memory_vote", "Environment")
                if isinstance(memory_vote, dict):
                    signed_bias = float(memory_vote.get("signed_bias", 0.0))
                    playbook_confidence = float(memory_vote.get("confidence", 0.5))
                    neural_risk_hint = float(memory_vote.get("neural_risk_hint", 0.5))

                if neural_risk_hint == 0.5:
                    direct_hint = self.smart_bus.get("neural_risk_hint", "Environment")
                    if direct_hint is not None:
                        try:
                            neural_risk_hint = float(direct_hint)
                        except (TypeError, ValueError):
                            pass

        except Exception:
            pass

        return [
            float(np.clip(risk_multiplier, 0.0, 1.0)),  # 1: Risk multiplier (inverted: 0=risky, 1=safe)
            float(np.clip(danger_similarity, 0.0, 1.0)),  # 2: Danger zone similarity
            float(np.clip(loss_prob, 0.0, 1.0)),  # 3: Neural P(loss)
            float(np.clip(veto_active, 0.0, 1.0)),  # 4: Memory veto active
            float(np.clip(signed_bias, -1.0, 1.0)),  # 5: Playbook directional bias
            float(np.clip(playbook_confidence, 0.0, 1.0)),  # 6: Playbook confidence
            float(np.clip(consecutive_losses_norm, 0.0, 1.0)),  # 7: Loss streak (normalized)
            float(np.clip(neural_risk_hint, 0.0, 1.0)),  # 8: Neural attention risk
        ]

    # ──────────────────────────────────────────────────────────────
    # Limits & misc
    # ──────────────────────────────────────────────────────────────
    def _check_termination(self) -> Tuple[bool, bool]:
        """Bus-first termination: emergency_mode / bus limits / prop firm rules override config."""
        # 1) Emergency mode (if any module raised it)
        if self.smart_bus and self.halt_on_emergency:
            try:
                em = self.smart_bus.get("emergency_mode", "Environment")
                if isinstance(em, dict):
                    if bool(em.get("halt", False)) or bool(em.get("active", False)):
                        return True, False
                elif isinstance(em, (int, float)) and em:
                    return True, False
                elif isinstance(em, bool) and em:
                    return True, False
            except Exception:
                pass

        # 2) Prop firm limit check (from UnifiedLotCalculator via bus)
        if self.smart_bus and self.prefer_bus_limits:
            try:
                prop_firm_status = self.smart_bus.get("prop_firm_status", "Environment")
                if isinstance(prop_firm_status, dict):
                    if bool(prop_firm_status.get("must_close_all", False)):
                        return True, False
            except Exception:
                pass

        # 3) Bus-provided limits (e.g., ComplianceModule / PortfolioRiskSystem)
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

        # 4) Apply limits and step bounds
        if self.market_state.current_drawdown > dd_limit:
            return True, False
        try:
            if self._min_data_len and int(self.current_step) >= int(self._min_data_len) - 1:
                return False, True
        except Exception:
            pass
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
                self._data_store = self._store
                self._is_fallback = True
                self._owners = {}

            def set(self, key, value, module=None, thesis=None):
                with self._lock:
                    self._store[key] = value

            def get(self, key, module=None, default=None):
                with self._lock:
                    return self._store.get(key, default)

            def register_provider(self, module, keys): return True
            def register_consumer(self, module, keys): return True

            def declare_owner(self, key: str, module: str):
                """Declare an owning module for a key (fallback no-op)."""
                with self._lock:
                    if key not in self._store:
                        self._store[key] = None
                    self._owners[key] = module
                return True

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
                    self._bus_ready.wait(
                        timeout=max(
                            1.0,
                            float(getattr(self.config, "info_bus_init_timeout", 2.0)) * 5,
                        )
                    )
                if self._bus_ready.is_set() and getattr(self.smart_bus, "_is_fallback", False):
                    try:
                        real_bus = (
                            InfoBusManager.get_instance()
                            if (SMARTINFOBUS_AVAILABLE and InfoBusManager)
                            else None
                        )
                        if real_bus is not None:
                            self.smart_bus = real_bus
                            self.logger.info(
                                "SmartInfoBus is ready - switched from fallback to real bus"
                            )
                    except Exception as e:
                        self.logger.warning(f"Failed switching to real SmartInfoBus: {e}")

                if not self._orch_ready.is_set():
                    wait_time = float(
                        getattr(self.config, "orchestrator_init_timeout", 2.0)
                    ) * (3 if bool(getattr(self.config, "orchestrator_async_init", True)) else 1)
                    self._orch_ready.wait(timeout=max(2.0, wait_time))

                if self._orch_ready.is_set() and self.orchestrator and not self.orchestrator_enabled:
                    self.orchestrator_enabled = True
                    self.logger.info(
                        "ModuleOrchestrator is ready - enabling orchestrator execution"
                    )
            except Exception as e:
                self.logger.warning(f"Post-init monitor error: {e}")

        threading.Thread(target=monitor, daemon=True).start()

    # ──────────────────────────────────────────────────────────────
    # Diagnostics & rendering
    # ──────────────────────────────────────────────────────────────
    def get_smartinfobus_status(self) -> Dict[str, Any]:
        modules_active = (
            len(self.orchestrator.modules)
            if self.orchestrator and hasattr(self.orchestrator, "modules")
            else 0
        )
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

        # Do not shutdown the global orchestrator singleton here.

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
