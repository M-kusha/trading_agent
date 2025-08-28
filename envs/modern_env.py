# envs/modern_env.py
"""
Modern SmartInfoBus Trading Environment
Zero-wiring architecture with automatic module discovery
Gymnasium-compatible (hardened) + non-blocking orchestrator execution
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

# Configuration
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
    SmartInfoBus-integrated trading env (zero-wiring).
    Gymnasium v0.26+ API: reset()->(obs, info), step()->(obs, reward, terminated, truncated, info).
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

        # Config & state
        self.config = config or TradingConfig()
        self.current_step = 0

        # Logger first (avoid circular deps)
        self.logger = self._create_logger()

        # SmartInfoBus / Orchestrator handles
        self.smart_bus = None  # type: ignore[assignment]
        self.smart_bus_enabled = False
        self.orchestrator = None  # type: ignore[assignment]
        self.orchestrator_enabled = False

        # Background asyncio loop (to avoid blocking .step() / .reset())
        self._aio_loop = None  # type: Optional[asyncio.AbstractEventLoop]
        self._aio_thread = None  # type: Optional[threading.Thread]
        self._aio_ready = threading.Event()
        # Track scheduled orchestrator futures to cancel/drain on close
        self._pending_futures = set()  # type: Set[Future]
        self._pend_lock = threading.Lock()

        # readiness events for late enabling
        self._bus_ready = threading.Event()
        self._orch_ready = threading.Event()

        # Bring systems up (with timeouts + fallbacks)
        self._initialize_systems()

        # Data
        self.orig_data = data_dict  # type: Dict[str, Dict[str, pd.DataFrame]]
        self.data = copy.deepcopy(data_dict)  # type: Dict[str, Dict[str, pd.DataFrame]]
        self.instruments = list(self.data.keys())  # type: List[str]
        self._validate_data()

        # Market state
        initial_balance = float(self.config.initial_balance)
        self.market_state = MarketState(
            balance=initial_balance,
            peak_balance=initial_balance,
            current_step=0,
            current_drawdown=0.0,
        )

        # Episode tracking
        self.episode_count = 0
        self.episode_metrics = EpisodeMetrics()

        # Spaces
        self.action_dim = 2 * len(self.instruments)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.observation_space = self._get_observation_space()

        # Setup env & publish initial context
        self._setup_environment()

        # Start dedicated event loop thread for the orchestrator
        self._start_event_loop_thread()

        modules = len(self.orchestrator.modules) if (self.orchestrator and hasattr(self.orchestrator, "modules")) else 0
        self.logger.info(
            f"🚀 MODERN_ENV_INITIALIZED: {len(self.instruments)} instruments, {modules} modules - Zero-wiring architecture active"
        )

    # ──────────────────────────────────────────────────────────────
    # Logging
    # ──────────────────────────────────────────────────────────────
    def _create_logger(self):
        try:
            if SMARTINFOBUS_AVAILABLE and RotatingLogger:
                return RotatingLogger(
                    name="ModernTradingEnv",
                    log_path="logs/modern_env.log",
                    max_lines=2000,
                    operator_mode=True,
                )
            # Fallback std logger
            import logging
            lg = logging.getLogger("ModernTradingEnv")
            if not lg.handlers:
                h = logging.StreamHandler()
                fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                h.setFormatter(fmt)
                lg.addHandler(h)
                lg.setLevel(logging.INFO)
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
                        # Pylance: InfoBusManager can be None at import-time; we guard and cast.
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
            # Fallback in-process bus (thread-safe)
            self.smart_bus = self._create_fallback_smart_bus()
            self.smart_bus_enabled = True
            self._bus_ready.set()

        # Orchestrator
        if MODULE_SYSTEM_AVAILABLE and ModuleOrchestrator:
            try:
                MO = cast(Any, ModuleOrchestrator)
                def init_orch():
                    try:
                        # Pylance: ModuleOrchestrator may be None; we guard and cast.
                        orch = MO()  # type: ignore[call-arg]
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

        # Post-init monitor to flip from fallback -> real when ready
        self._start_post_init_monitor()

    def _start_event_loop_thread(self):
        """Run a dedicated asyncio loop in a background thread for orchestrator coroutines."""
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
                try:
                    if not df.index.is_monotonic_increasing:
                        self.logger.warning(f"Index not monotonic for {inst}/{tf}")
                except Exception:
                    pass

    def _get_observation_space(self) -> spaces.Box:
        default_size = 256
        try:
            if self.smart_bus:
                sz = self.smart_bus.get("environment_observation_size", "Environment")
                if isinstance(sz, int) and sz > 0:
                    default_size = sz
        except Exception:
            pass
        return spaces.Box(low=-np.inf, high=np.inf, shape=(default_size,), dtype=np.float32)

    def _setup_environment(self):
        try:
            if self.smart_bus:
                self.smart_bus.set(
                    "environment_config",
                    {
                        "instruments": self.instruments,
                        "initial_balance": float(self.config.initial_balance),
                        "action_dim": self.action_dim,
                        "max_steps": int(self.config.max_steps),
                    },
                    module="Environment",
                    thesis="Environment configuration for module access",
                )
        except Exception as e:
            self.logger.warning(f"Failed to publish environment_config: {e}")

        self._store_market_data()

    def _store_market_data(self):
        step = int(self.market_state.current_step)
        if not self.smart_bus:
            return
        for instrument in self.instruments:
            for timeframe in ["H1", "H4", "D1"]:
                try:
                    if timeframe not in self.data[instrument]:
                        continue
                    df = self.data[instrument][timeframe]
                    if step >= len(df):
                        continue
                    price = float(df["close"].iloc[step])
                    self.smart_bus.set(
                        f"price_{instrument}_{timeframe}",
                        price,
                        module="Environment",
                        thesis=f"Current {instrument} price at step {step}",
                    )
                    # publish small rolling window (best-effort)
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
                        thesis=f"Market data window for {instrument} {timeframe}",
                    )
                except Exception:
                    # keep quiet to avoid spam during tight loops
                    pass

    # ──────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.logger.info(f"🔄 ENVIRONMENT_RESET: Episode {self.episode_count + 1}")

        # episode tracking
        self.episode_count += 1
        self.episode_metrics = EpisodeMetrics()

        # reset data snapshot
        self.data = copy.deepcopy(self.orig_data)

        # market state
        initial_balance = float(self.config.initial_balance)
        self.market_state = MarketState(
            balance=initial_balance,
            peak_balance=initial_balance,
            current_step=self._select_starting_step(),
            current_drawdown=0.0,
        )
        self.current_step = int(self.market_state.current_step)

        # publish reset info
        try:
            if self.smart_bus:
                self.smart_bus.set(
                    "episode_info",
                    {
                        "episode": self.episode_count,
                        "step": self.current_step,
                        "balance": self.market_state.balance,
                        "reset": True,
                    },
                    module="Environment",
                    thesis=f"Episode {self.episode_count} reset information",
                )
        except Exception:
            pass

        self._store_market_data()
        # Publish environment observation sizing for downstream modules
        try:
            if self.smart_bus:
                self.smart_bus.set(
                    "environment_observation_size",
                    int(self.observation_space.shape[0]) if self.observation_space.shape else 256,
                    module="Environment",
                    thesis="Declared observation vector size"
                )
        except Exception:
            pass
        obs = self._get_observation()
        # Publish initial observation for consumers needing it at reset
        try:
            if self.smart_bus is not None:
                self.smart_bus.set(
                    "environment_observation",
                    obs,
                    module="Environment",
                    thesis="Initial environment observation"
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
        action = action.astype(np.float32).reshape(self.action_dim,)

        self.current_step += 1
        self.market_state.current_step = int(self.current_step)

        # publish action
        try:
            if self.smart_bus:
                self.smart_bus.set(
                    "agent_action",
                    action,
                    module="Environment",
                    thesis=f"Agent action at step {self.current_step}",
                )
                # Surface the final_trading_action alias for compatibility
                self.smart_bus.set(
                    "final_trading_action",
                    action,
                    module="Environment",
                    thesis="Environment echo of action as final_trading_action"
                )
        except Exception:
            pass

        # update market data + state snapshot
        self._store_market_data()
        try:
            if self.smart_bus:
                self.smart_bus.set(
                    "market_state",
                    {
                        "balance": self.market_state.balance,
                        "step": self.current_step,
                        "drawdown": self.market_state.current_drawdown,
                        "peak_balance": self.market_state.peak_balance,
                    },
                    module="Environment",
                    thesis="Current market state for modules",
                )
        except Exception:
            pass

        # Non-blocking orchestrator execution (fire-and-forget)
        if self.orchestrator_enabled and self.orchestrator and hasattr(self.orchestrator, "execute_step"):
            self._run_orchestrator_step({})  # inputs can be extended later

        # allow modules to override final action
        try:
            final_action = self.smart_bus.get("final_trading_action", "Environment") if self.smart_bus else None
            if final_action is None:
                final_action = action
        except Exception:
            final_action = action

        reward = float(self._execute_step(final_action))
        obs = self._get_observation()
        # Publish observation each step for downstream consumers
        try:
            if self.smart_bus:
                self.smart_bus.set(
                    "environment_observation",
                    obs,
                    module="Environment",
                    thesis=f"Environment observation at step {self.current_step}"
                )
        except Exception:
            pass
        terminated, truncated = self._check_termination()

        modules = len(self.orchestrator.modules) if (self.orchestrator and hasattr(self.orchestrator, "modules")) else 0
        info = {
            "step": self.current_step,
            "balance": self.market_state.balance,
            "drawdown": self.market_state.current_drawdown,
            "reward": reward,
            "modules_executed": modules,
            "terminated": terminated,
            "truncated": truncated,
        }
        return obs, reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────
    def _run_orchestrator_step(self, inputs: Dict[str, Any]) -> None:
        """
        Schedule orchestrator.execute_step(inputs) on the dedicated loop thread.
        Never blocks the env (prevents hangs in quick tests / evals).
        """
        try:
            coro = self.orchestrator.execute_step(inputs)  # type: ignore[attr-defined]
        except Exception as e:
            self.logger.warning(f"Cannot build orchestrator coroutine: {e}")
            return

        loop = self._aio_loop
        if loop and loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                # track for graceful shutdown
                with self._pend_lock:
                    self._pending_futures.add(fut)

                def _on_done(t: Future):
                    # log if failed and remove from tracking
                    try:
                        exc = t.exception()
                        if exc:
                            self.logger.error(f"Orchestrator step failed: {exc}")
                    except Exception:
                        # accessing exception() can itself raise if cancelled; ignore
                        pass
                    finally:
                        with self._pend_lock:
                            self._pending_futures.discard(t)

                fut.add_done_callback(_on_done)
            except Exception as e:
                self.logger.warning(f"Failed to schedule orchestrator step: {e}")
        else:
            # Last-resort: run in a throwaway thread (still non-blocking)
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

    def _get_observation(self) -> np.ndarray:
        # try module-provided observation
        expected = self.observation_space.shape[0] if self.observation_space.shape else 256
        try:
            obs = self.smart_bus.get("environment_observation", "Environment") if self.smart_bus else None
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

        # fallback engineered obs
        return self._create_fallback_observation(expected)

    def _create_fallback_observation(self, expected_size: int) -> np.ndarray:
        feats: List[float] = []
        feats.extend([
            self.market_state.balance / 10000.0,
            float(self.market_state.current_drawdown),
            float(self.current_step) / 1000.0,
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

                        base = max(close_, 1e-12)
                        feats.extend([
                            close_ / 10000.0,
                            (high_ - low_) / base,
                            (close_ - open_) / base,
                            vol_ / 1000.0,
                        ])

                        # momentum (5)
                        if self.current_step >= 5:
                            prev = float(df["close"].iloc[self.current_step - 5])
                            feats.append((close_ - prev) / max(prev, 1e-12))
                        else:
                            feats.append(0.0)

                        # volatility (20)
                        if self.current_step >= 20:
                            recent = df["close"].iloc[self.current_step - 19:self.current_step + 1].to_numpy(dtype=np.float64)
                            m = float(np.mean(recent, dtype=np.float64))
                            v = float(np.std(recent, dtype=np.float64) / m) if m > 0 else 0.0
                            feats.append(v)
                        else:
                            feats.append(0.01)
                    else:
                        feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                else:
                    feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # cross-timeframe trend alignments
        for instrument in self.instruments:
            h1 = h4 = d1 = 0.0
            if "H1" in self.data[instrument] and 5 <= self.current_step < len(self.data[instrument]["H1"]):
                df = self.data[instrument]["H1"]
                cur, past = float(df["close"].iloc[self.current_step]), float(df["close"].iloc[self.current_step - 5])
                h1 = (cur - past) / max(past, 1e-12)
            if "H4" in self.data[instrument] and 5 <= self.current_step < len(self.data[instrument]["H4"]):
                df = self.data[instrument]["H4"]
                cur, past = float(df["close"].iloc[self.current_step]), float(df["close"].iloc[self.current_step - 5])
                h4 = (cur - past) / max(past, 1e-12)
            if "D1" in self.data[instrument] and 5 <= self.current_step < len(self.data[instrument]["D1"]):
                df = self.data[instrument]["D1"]
                cur, past = float(df["close"].iloc[self.current_step]), float(df["close"].iloc[self.current_step - 5])
                d1 = (cur - past) / max(past, 1e-12)
            feats.extend([h1, h4, d1, 1.0 if (h1 > 0 and h4 > 0 and d1 > 0) else 0.0, 1.0 if (h1 < 0 and h4 < 0 and d1 < 0) else 0.0])

        arr = np.asarray(feats, dtype=np.float32)
        if arr.size < expected_size:
            out = np.zeros(expected_size, dtype=np.float32)
            out[:arr.size] = arr
            return out
        return arr[:expected_size]

    def _execute_step(self, action: np.ndarray) -> float:
        # Prefer module-provided trading result if any
        try:
            tr = self.smart_bus.get("trading_result", "Environment") if self.smart_bus else None
        except Exception:
            tr = None

        if tr:
            pnl = float(tr.get("pnl", 0.0))
            self.market_state.balance += pnl
            if self.market_state.balance > self.market_state.peak_balance:
                self.market_state.peak_balance = self.market_state.balance
                self.market_state.current_drawdown = 0.0
            else:
                denom = max(self.market_state.peak_balance, 1e-12)
                self.market_state.current_drawdown = (self.market_state.peak_balance - self.market_state.balance) / denom
            return pnl / 100.0

        # No result: mild penalty to encourage producing signals
        # Also publish a minimal trading_result placeholder for compatibility
        try:
            if self.smart_bus:
                self.smart_bus.set(
                    "trading_result",
                    {"pnl": 0.0, "source": "Environment", "step": int(self.current_step)},
                    module="Environment",
                    thesis="Placeholder trading result (no external provider)"
                )
        except Exception:
            pass
        return -0.01

    def _check_termination(self) -> Tuple[bool, bool]:
        if self.market_state.current_drawdown > float(self.config.max_drawdown):
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

            def get(self, key, module=None):
                with self._lock:
                    return self._store.get(key)

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
                f"Balance: ${self.market_state.balance:.2f}, "
                f"Drawdown: {self.market_state.current_drawdown:.1%}"
            )

    def close(self):
        self.logger.info("Environment closed")
        # prevent new schedules
        try:
            self.orchestrator_enabled = False
        except Exception:
            pass

        # allow orchestrator to perform its own shutdown steps
        try:
            if self.orchestrator and hasattr(self.orchestrator, "shutdown"):
                # synchronous shutdown; cancels internal monitors, saves state
                self.orchestrator.shutdown()
        except Exception as e:
            try:
                self.logger.warning(f"Orchestrator shutdown warning: {e}")
            except Exception:
                pass

        # Cancel any pending orchestrator futures and drain the loop once
        loop = self._aio_loop
        try:
            if loop and loop.is_running():
                # snapshot and cancel
                with self._pend_lock:
                    to_cancel = list(self._pending_futures)
                for fut in to_cancel:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                # tick the loop to process cancellations
                try:
                    tick = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop)
                    tick.result(timeout=0.5)
                except Exception:
                    pass
                # finally stop the loop
                loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass

        # join background thread
        try:
            if self._aio_thread and self._aio_thread.is_alive():
                self._aio_thread.join(timeout=1.5)
        except Exception:
            pass

        # clear references
        try:
            with self._pend_lock:
                self._pending_futures.clear()
        except Exception:
            pass
        self._aio_loop = None
        self._aio_thread = None

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
