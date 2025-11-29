"""
Enhanced Training Callback - Separate Module
File: train/enhanced_training_callback.py
Production-ready with proper error handling and SmartInfoBus integration
Refactored to be Pylance-friendly (no type identity collisions)
"""

from __future__ import annotations

import os
import json
import time
import threading
import asyncio
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Any, Optional, List, Deque
from collections import deque, defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# WebSocket client for sending metrics to backend
WEBSOCKET_AVAILABLE = False
try:
    import websockets
    from websockets.sync.client import connect as ws_connect
    WEBSOCKET_AVAILABLE = True
except ImportError:
    websockets = None  # type: ignore
    ws_connect = None  # type: ignore

# Guard against partially available import where ws_connect is None
if ws_connect is None:
    WEBSOCKET_AVAILABLE = False

try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except Exception:
    requests = None  # type: ignore
    REQUESTS_AVAILABLE = False


# ───────────────────────────────────────────────────────────────────
# WebSocket Metrics Broadcaster - Sends training metrics to backend
# ───────────────────────────────────────────────────────────────────
class WebSocketMetricsBroadcaster:
    """Broadcasts training metrics to backend via WebSocket on port 8001 with HTTP fallback."""
    
    def __init__(self, host: str = "localhost", port: int = 8001, http_fallback_url: Optional[str] = None):
        self.uri = f"ws://{host}:{port}"
        self.http_url = http_fallback_url or "http://localhost:8000/api/training/metrics"
        self.ws: Any = None
        self.connected = False
        self._lock = threading.Lock()
        self._connect_attempts = 0
        # Try websocket up to 3 times with delays before falling back to HTTP
        disable_ws = os.getenv("METRICS_WS_DISABLE", "0") == "1" or os.getenv("TRAINING_METRICS_WS_DISABLE", "0") == "1"
        self._max_connect_attempts = 0 if disable_ws else 3
        self._http_warned = False
        self._deferred_connect = True  # Defer connection until first metrics send
        self._first_connect_done = False
        
    def connect(self) -> bool:
        """Attempt to connect to the backend WebSocket server with retries"""
        if self._max_connect_attempts == 0:
            return False
        if not WEBSOCKET_AVAILABLE or ws_connect is None:
            if not self._first_connect_done:
                print("[WARN] websockets package not available - using HTTP fallback for metrics")
                self._first_connect_done = True
            return False
            
        if self.connected and self.ws:
            return True
            
        with self._lock:
            # Already exceeded max attempts
            if self._connect_attempts >= self._max_connect_attempts:
                return False
            
            # Retry loop with exponential backoff
            while self._connect_attempts < self._max_connect_attempts:
                self._connect_attempts += 1
                try:
                    connector = ws_connect
                    if connector is None:
                        # Should not happen because of guards, but keep safe for type checkers
                        return False
                    # Use longer timeout and disable compression for reliability
                    self.ws = connector(
                        self.uri, 
                        open_timeout=10,
                        close_timeout=5,
                        max_size=2**20,  # 1MB max message size
                    )
                    self.connected = True
                    self._first_connect_done = True
                    print(f"[OK] Connected to training metrics server at {self.uri}")
                    return True
                except Exception as e:
                    if self._connect_attempts < self._max_connect_attempts:
                        # Wait before retry (exponential backoff: 1s, 2s, 4s)
                        wait_time = 2 ** (self._connect_attempts - 1)
                        time.sleep(wait_time)
                    else:
                        # Final attempt failed - switch to HTTP fallback silently
                        if not self._first_connect_done:
                            print(f"[INFO] WebSocket unavailable, using HTTP fallback for training metrics")
                            self._first_connect_done = True
            
            self.connected = False
            return False

    def _send_http_fallback(self, metrics: Dict[str, Any]) -> bool:
        """
        Fallback path to push metrics via HTTP if websocket handshake fails.
        Uses requests when available, otherwise urllib from stdlib.
        """
        payload = {
            "type": "training_metrics",
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        }

        try:
            if REQUESTS_AVAILABLE and requests is not None:
                resp = requests.post(self.http_url, json=payload, timeout=2)
                return 200 <= resp.status_code < 300
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.http_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2) as resp:  # nosec B310
                return 200 <= getattr(resp, "status", 0) < 300
        except Exception as e:
            if not self._http_warned:
                # Avoid log spam if backend endpoint is unavailable
                print(f"[WARN] HTTP metrics fallback failed: {e}")
                self._http_warned = True
            return False
            
    def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics to the backend WebSocket server"""
        if not self.connected:
            if not self.connect():
                return self._send_http_fallback(metrics)
                
        try:
            with self._lock:
                if self.ws is None:
                    return self._send_http_fallback(metrics)
                    
                message = {
                    "type": "training_metrics",
                    "data": metrics,
                    "timestamp": datetime.now().isoformat()
                }
                self.ws.send(json.dumps(message))
                return True
        except Exception:
            self.connected = False
            self.ws = None
            # Try to reconnect on next send
            return self._send_http_fallback(metrics)
            
    def close(self):
        """Close the WebSocket connection"""
        try:
            with self._lock:
                if self.ws:
                    self.ws.close()
                self.ws = None
                self.connected = False
        except Exception:
            pass


# Import beautiful visualizer
try:
    from train.training_visualizer import BeautifulTrainingVisualizer as _BeautifulTrainingVisualizer
    VISUALIZER_AVAILABLE = True
    BeautifulTrainingVisualizer: Any = _BeautifulTrainingVisualizer
except Exception:
    VISUALIZER_AVAILABLE = False
    # Fallback class to avoid None type errors
    class _FallbackVisualizer:
        def __init__(self, **kwargs): pass
        def render_complete_display(self, *args, **kwargs): pass
    BeautifulTrainingVisualizer = _FallbackVisualizer

# ───────────────────────────────────────────────────────────────────
# Resolve dependencies without type-identity collisions
# Use "*_Cls" variables typed as Any to hold class refs (real or fallback)
# ───────────────────────────────────────────────────────────────────
from typing import Any
from modules.utils.metrics_utils import sanitize_metrics

SMARTINFOBUS_AVAILABLE = False
MONITORING_AVAILABLE = False

# Defaults (will be overwritten if real modules import successfully)
InfoBusManager: Any = None
RotatingLogger_Cls: Any = None
format_operator_message_func: Any = None
SystemUtilities_Cls: Any = None
EnglishExplainer_Cls: Any = None

HealthMonitor_Cls: Any = None
PerformanceTracker_Cls: Any = None
IntegrationValidator_Cls: Any = None
ErrorPinpointer_Cls: Any = None
create_error_handler_func: Any = None

# Try SmartInfoBus utils
try:
    from modules.utils.info_bus import InfoBusManager as _InfoBusManager  # type: ignore
    from modules.utils.audit_utils import RotatingLogger as _RotatingLogger, format_operator_message as _format_operator_message  # type: ignore
    from modules.utils.system_utilities import SystemUtilities as _SystemUtilities, EnglishExplainer as _EnglishExplainer  # type: ignore
    SMARTINFOBUS_AVAILABLE = True
    InfoBusManager = _InfoBusManager
    RotatingLogger_Cls = _RotatingLogger
    format_operator_message_func = _format_operator_message
    SystemUtilities_Cls = _SystemUtilities
    EnglishExplainer_Cls = _EnglishExplainer
except Exception:
    # Fallbacks that won’t collide with external types
    SMARTINFOBUS_AVAILABLE = False

    class _FallbackSmartBus:
        def __init__(self):
            self._d: Dict[str, Any] = {}

        def set(self, k, v, module=None, thesis=None):
            self._d[k] = v

        # Align with env/training script API by supporting `default`
        def get(self, k, module=None, default=None):
            return self._d.get(k, default)

        def get_performance_metrics(self):
            return {"active": True, "disabled_modules": [], "active_data_keys": len(self._d)}

        # Optional for scripts that inspect internal storage
        @property
        def _data_store(self):
            return dict(self._d)

        def export_session(self, path: str):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self._d, f, indent=2)

    class _FallbackInfoBusManager:
        @staticmethod
        def get_instance():
            return _FallbackSmartBus()

    class _FallbackRotatingLogger:
        def __init__(self, name=None, **kwargs):
            import logging
            self._l = logging.getLogger(name or "EnhancedCallback")
            if not self._l.handlers:
                h = logging.StreamHandler()
                self._l.addHandler(h)
                self._l.setLevel(logging.INFO)
        def info(self, msg): self._l.info(msg)
        def warning(self, msg): self._l.warning(msg)
        def error(self, msg): self._l.error(msg)
        def critical(self, msg): self._l.critical(msg)

    def _fallback_format_operator_message(message: str = "", icon: str = "ℹ️", **kwargs):
        det = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        return f"{icon} {message}" + (f" ({det})" if det else "")

    class _FallbackSystemUtilities: ...
    class _FallbackEnglishExplainer: ...

    InfoBusManager = _FallbackInfoBusManager
    RotatingLogger_Cls = _FallbackRotatingLogger
    format_operator_message_func = _fallback_format_operator_message
    SystemUtilities_Cls = _FallbackSystemUtilities
    EnglishExplainer_Cls = _FallbackEnglishExplainer

# Try monitoring/error modules (typed as Any to avoid signature drift issues)
try:
    from modules.monitoring.health_monitor import HealthMonitor as _HealthMonitor  # type: ignore
    from modules.monitoring.performance_tracker import PerformanceTracker as _PerformanceTracker  # type: ignore
    from modules.monitoring.integration_validator import IntegrationValidator as _IntegrationValidator  # type: ignore
    from modules.core.error_pinpointer import ErrorPinpointer as _ErrorPinpointer, create_error_handler as _create_error_handler  # type: ignore
    MONITORING_AVAILABLE = True
    HealthMonitor_Cls = _HealthMonitor
    PerformanceTracker_Cls = _PerformanceTracker
    IntegrationValidator_Cls = _IntegrationValidator
    ErrorPinpointer_Cls = _ErrorPinpointer
    create_error_handler_func = _create_error_handler
except Exception:
    MONITORING_AVAILABLE = False

    class _LocalHealthMonitor:
        def __init__(self, **kwargs): self._running = False
        def start(self): self._running = True
        def stop(self): self._running = False
        def check_system_health(self):
            return {
                "overall_status": "ok",
                "overall_score": 95,
                "system": {"cpu_percent": 30.0, "memory_percent": 50.0},
                "modules": {"healthy_modules": 5, "total_modules": 5, "module_details": {}},
            }
        def get_status(self): return {"running": self._running}

    class _LocalPerformanceTracker:
        def __init__(self, **kwargs): pass
        def record_metric(self, *a, **k): pass
        def generate_performance_report(self):
            return type("R", (), {"module_metrics": {}})()

    class _LocalIntegrationValidator:
        def __init__(self, **kwargs): pass
        def validate_system(self):
            return type("R", (), {"integration_score": 95, "issues": []})()

    class _LocalErrorPinpointer:
        def analyze_error(self, e, ctx): return f"{ctx}: {e}"

    def _local_create_error_handler(*_a, **_k):
        class _H:
            def handle_error(self, e, context): pass
        return _H()

    HealthMonitor_Cls = _LocalHealthMonitor
    PerformanceTracker_Cls = _LocalPerformanceTracker
    IntegrationValidator_Cls = _LocalIntegrationValidator
    ErrorPinpointer_Cls = _LocalErrorPinpointer
    create_error_handler_func = _local_create_error_handler


# ───────────────────────────────────────────────────────────────────
# Callback
# ───────────────────────────────────────────────────────────────────
class ModernEnhancedTrainingCallback(BaseCallback):
    """
    Enhanced training callback with comprehensive monitoring, TensorBoard logging,
    SmartInfoBus integration, and SB3 compatibility (no logger-name conflicts).
    """

    def __init__(self, total_timesteps: int, config: Any,
                 metrics_broadcaster: Any = None, verbose: int = 1,
                 use_beautiful_display: bool = True,
                 enable_ws_broadcast: bool = False):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.config = config
        
        # Auto-create WebSocket broadcaster if not provided and enabled
        if metrics_broadcaster is None and enable_ws_broadcast:
            try:
                ws_broadcaster = WebSocketMetricsBroadcaster(host="localhost", port=8001)
                # Don't connect immediately - defer to first metrics send for better reliability
                # This avoids race conditions with backend startup
                self.metrics_broadcaster = ws_broadcaster
                print("[INFO] WebSocket metrics broadcaster initialized (will connect on first send)")
            except Exception as e:
                self.metrics_broadcaster = None
                print(f"[WARN] Failed to create WebSocket broadcaster: {e}")
        else:
            self.metrics_broadcaster = metrics_broadcaster
            
        self.use_beautiful_display = use_beautiful_display and VISUALIZER_AVAILABLE

        # Initialize beautiful visualizer
        self.visualizer = None
        if self.use_beautiful_display:
            try:
                self.visualizer = BeautifulTrainingVisualizer(config=config, terminal_width=120)
            except Exception as e:
                print(f"[WARN] Failed to initialize visualizer: {e}")
                self.use_beautiful_display = False

        # Runtime state
        self.start_time: datetime = datetime.now()
        self.last_print_time: datetime = self.start_time
        self.best_reward: float = -float("inf")
        self.current_episode_reward: float = 0.0
        self.episode_count: int = 0
        self.consecutive_failures: int = 0
        self.circuit_breaker_state = {"active": False, "failures": 0}
        self.display_update_counter: int = 0
        # Global step offset when resuming from checkpoints
        self.initial_num_timesteps: int = 0

        # Rolling windows
        self.episode_rewards: Deque[float] = deque(maxlen=2000)
        self.episode_lengths: Deque[int] = deque(maxlen=2000)
        self.step_durations_ms: Deque[float] = deque(maxlen=500)
        self.performance_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.health_alerts: Deque[Dict[str, Any]] = deque(maxlen=100)
        self.info_bus_quality_history: Deque[Dict[str, Any]] = deque(maxlen=500)
        self.module_performance_history = defaultdict(lambda: deque(maxlen=100))

        # Health cadence
        self.health_check_interval = max(50, int(getattr(config, "risk_check_frequency", 1)) * 50)
        self.last_health_check = 0

        # Cache for training metrics (SB3's logger.name_to_value resets between updates)
        # We retain the last non-zero values so metrics aren't always 0 between PPO updates
        self._cached_training_metrics: Dict[str, float] = {
            "clip_fraction": 0.0,
            "explained_variance": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "policy_gradient_loss": 0.0,
            "approx_kl": 0.0,
        }
        self._last_ppo_update_step: int = 0  # Track when PPO last updated

        # Safe logger
        self.training_log = RotatingLogger_Cls(
            name="EnhancedTrainingCallback",
            log_path=f"logs/training/enhanced_callback_{datetime.now():%Y%m%d}.log",
            max_lines=int(getattr(config, "log_rotation_lines", 2000)),
            operator_mode=True,
            plain_english=True,
        )

        # Smart bus instance
        self.smart_bus = InfoBusManager.get_instance()

        # Monitoring suite
        self.error_pinpointer = ErrorPinpointer_Cls()
        self.error_handler = create_error_handler_func("TrainingCallback", self.error_pinpointer)  # type: ignore[call-arg]
        self.health_monitor = HealthMonitor_Cls(check_interval=30)  # auto_start handled manually
        self.performance_tracker = PerformanceTracker_Cls(orchestrator=None)
        self.system_utilities = SystemUtilities_Cls()
        self.english_explainer = EnglishExplainer_Cls()
        self.integration_validator = IntegrationValidator_Cls()

        try:
            self.health_monitor.start()
        except Exception as e:
            self.training_log.warning(f"Health monitor start failed: {e}")

        self.training_log.info(format_operator_message_func(
            message="Enhanced training callback initialized",
            icon="[ROCKET]",
            total_timesteps=total_timesteps,
            smartinfobus_v4=SMARTINFOBUS_AVAILABLE,
            monitoring=MONITORING_AVAILABLE,
        ))

        mode_str = "LIVE" if getattr(config, "live_mode", False) else "OFFLINE"
        print(f"\n[ROCKET] ENHANCED TRAINING CALLBACK READY")
        print(f"[STATS] Total timesteps: {total_timesteps:,}")
        print(f"🔗 SmartInfoBus v4.0: {'ENABLED' if SMARTINFOBUS_AVAILABLE else 'FALLBACK'}")
        print(f"[CHART] Monitoring: {'ENHANCED' if MONITORING_AVAILABLE else 'BASIC'}")
        print(f"🎨 Beautiful Display: {'ENABLED' if self.use_beautiful_display else 'DISABLED'}")
        print(f"[CHART] Mode: {mode_str}")
        print("─" * 60)

        self._tb_ready = False  # set at _on_training_start

    # ── SB3 hooks ───────────────────────────────────────────────────
    def _bus_get_multi(self, key: str, modules: List[str], default: Any) -> Any:
        if not SMARTINFOBUS_AVAILABLE:
            return default
        for module_name in modules:
            try:
                value = self.smart_bus.get(key, module=module_name, default=None)
                if value is not None:
                    return value
            except Exception:
                continue
        return default

    def _on_training_start(self) -> None:
        self.start_time = datetime.now()
        self._tb_ready = bool(getattr(self.model, "logger", None))
        # Capture starting global step so visualizer and metrics
        # continue counting from checkpoints instead of resetting
        try:
            self.initial_num_timesteps = int(getattr(self.model, "num_timesteps", 0) or 0)
        except Exception:
            self.initial_num_timesteps = 0

        # Integration validation (non-fatal)
        try:
            vr = self.integration_validator.validate_system()
            ok = float(getattr(vr, "integration_score", 100)) >= 80.0
            self.training_log.info(format_operator_message_func(
                message="System integration validated" if ok else "System integration warnings",
                icon="[OK]" if ok else "[WARN]",
                score=f"{getattr(vr, 'integration_score', 100):.1f}",
                issues=len(getattr(vr, "issues", [])),
            ))
        except Exception as e:
            self.training_log.warning(f"Integration validation failed: {e}")

        # Bus marker
        try:
            self.smart_bus.set(
                "enhanced_training_start",
                {
                    "timestamp": datetime.now().isoformat(),
                    "total_timesteps": self.total_timesteps,
                    "config_mode": "live" if getattr(self.config, "live_mode", False) else "offline",
                    "enhanced_monitoring": True,
                },
                module="EnhancedTrainingCallback",
                thesis="Training session started",
            )
        except Exception:
            pass

    def _on_step(self) -> bool:
        t0 = time.perf_counter()
        ok = True
        try:
            # Throttled progress print
            now = datetime.now()
            # Update display every 2 seconds for beautiful display, 10 seconds for basic
            update_interval = 2 if self.use_beautiful_display else 10
            if (now - self.last_print_time).total_seconds() >= update_interval:
                self._print_enhanced_progress()
                self.last_print_time = now

            # Collect metrics every 10 steps
            if self.n_calls % 10 == 0:
                metrics = self._collect_enhanced_metrics()
                self._update_performance_tracking(metrics)
                self._tb_log(metrics)
                if self.metrics_broadcaster:
                    try:
                        self.metrics_broadcaster.send_metrics(metrics)
                    except Exception as e:
                        self.training_log.warning(f"Metrics broadcast failed: {e}")

            # Health check cadence
            if self.n_calls - self.last_health_check >= self.health_check_interval:
                self._perform_enhanced_health_check()
                self.last_health_check = self.n_calls

            # Episode tracking
            self._track_episode_progress()

            # Circuit breaker
            if self._check_circuit_breaker():
                self.training_log.critical(format_operator_message_func(
                    message="Circuit breaker activated",
                    icon="[ALERT]",
                    failures=self.circuit_breaker_state["failures"],
                ))
                ok = False
        except Exception as e:
            self.consecutive_failures += 1
            self.training_log.error(format_operator_message_func(
                message="Enhanced callback step error",
                icon="[CRASH]",
                error=str(e),
                consecutive_failures=self.consecutive_failures,
            ))
            ok = self.consecutive_failures <= 10
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.step_durations_ms.append(dt_ms)
            try:
                self.performance_tracker.record_metric("EnhancedCallback", "step_ms", dt_ms, ok)
            except Exception:
                pass
        return ok

    def _on_training_end(self) -> None:
        end = datetime.now()
        dur = end - self.start_time

        if hasattr(self.health_monitor, "stop"):
            try:
                self.health_monitor.stop()
            except Exception:
                pass

        final_report = {
            "training_duration": str(dur),
            "total_episodes": self.episode_count,
            "total_steps": self.n_calls,
            "best_reward": self.best_reward,
            "avg_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "health_alerts_total": len(self.health_alerts),
            "circuit_breaker": self.circuit_breaker_state,
        }

        print(f"\n\n[OK] ENHANCED TRAINING COMPLETED!")
        print("=" * 70)
        print(f"[TIME]  Duration: {dur}")
        print(f"[STATS] Total steps: {self.n_calls:,}")
        print(f"🎬 Total episodes: {self.episode_count}")
        print(f"[TROPHY] Best reward: {self.best_reward:.2f}")
        if self.episode_rewards:
            print(f"[CHART] Final avg reward: {np.mean(self.episode_rewards):.2f}")
        print("=" * 70)

        self.training_log.info(format_operator_message_func(
            message="Enhanced training completed",
            icon="[OK]",
            duration=str(dur),
            episodes=self.episode_count,
            best_reward=self.best_reward,
            health_alerts=len(self.health_alerts),
        ))

        try:
            self.smart_bus.set(
                "enhanced_training_completed",
                final_report,
                module="EnhancedTrainingCallback",
                thesis="Training completed",
            )
        except Exception:
            pass

        # Save final report (best-effort)
        try:
            os.makedirs("logs/training", exist_ok=True)
            path = f"logs/training/enhanced_final_report_{datetime.now():%Y%m%d_%H%M%S}.json"
            with open(path, "w") as f:
                json.dump(final_report, f, indent=2)
            print(f"📁 Final report saved: {path}")
        except Exception:
            pass
        
        # Close WebSocket broadcaster if we own it
        if self.metrics_broadcaster and isinstance(self.metrics_broadcaster, WebSocketMetricsBroadcaster):
            try:
                self.metrics_broadcaster.close()
                print("[OK] WebSocket metrics broadcaster closed")
            except Exception as e:
                print(f"[WARN] Error closing WebSocket broadcaster: {e}")

    # ── Telemetry helpers ───────────────────────────────────────────
    def _print_enhanced_progress(self):
        """Display beautiful training progress (or fallback to basic display)"""
        # Use beautiful visualizer if available
        if self.use_beautiful_display and self.visualizer:
            try:
                metrics = self._collect_enhanced_metrics()
                self.visualizer.render_complete_display(self.smart_bus, metrics)
            except Exception as e:
                # Fallback to basic display on error
                print(f"\n[WARN] Visualizer error: {e}")
                import traceback
                traceback.print_exc()
                self.use_beautiful_display = False
                self._print_basic_progress()
        else:
            # Fallback to basic progress display
            self._print_basic_progress()

    def _print_basic_progress(self):
        """Basic text-based progress display (fallback)"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        # Use global step (checkpoint-aware) for progress
        global_step = self.initial_num_timesteps + self.n_calls
        progress = (global_step / max(self.total_timesteps, 1)) * 100.0

        eta_str = "calculating..."
        if progress > 0:
            total_est = elapsed / (progress / 100.0)
            rem = total_est - elapsed
            eta_str = f"{rem/60:.1f}min" if rem > 60 else f"{rem:.0f}s"

        # Latencies
        p50 = float(np.percentile(self.step_durations_ms, 50)) if self.step_durations_ms else 0.0
        p95 = float(np.percentile(self.step_durations_ms, 95)) if self.step_durations_ms else 0.0

        last = f"{self.episode_rewards[-1]:.2f}" if self.episode_rewards else "N/A"
        best = f"{self.best_reward:.2f}" if self.best_reward != -float("inf") else "N/A"

        print(
            f"\r[RELOAD] Step: {global_step:,}/{self.total_timesteps:,} "
            f"({progress:.1f}%) | [TIME] {elapsed/60:.1f}min | "
            f"[MONEY] Last: {last} | [TROPHY] Best: {best} | "
            f"[LAT] P50:{p50:.1f}ms P95:{p95:.1f}ms | [WAIT] ETA: {eta_str}",
            end="",
            flush=True,
        )

    def _collect_enhanced_metrics(self) -> Dict[str, Any]:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        # Global, checkpoint-aware step for visualizer / dashboards
        global_step = self.initial_num_timesteps + self.n_calls
        progress = global_step / max(self.total_timesteps, 1)
        # Keep SPS based on steps taken in this run
        sps = (self.n_calls / elapsed) if elapsed > 0 else 0.0

        m: Dict[str, Any] = {
            "timestep": global_step,
            "total_timesteps": self.total_timesteps,
            "progress_pct": progress * 100.0,
            "episodes": self.episode_count,
            "elapsed_time_s": elapsed,
            "steps_per_second": sps,
            "step_ms_p50": float(np.percentile(self.step_durations_ms, 50)) if self.step_durations_ms else 0.0,
            "step_ms_p95": float(np.percentile(self.step_durations_ms, 95)) if self.step_durations_ms else 0.0,
            "episode_reward_mean": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "episode_reward_std": float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0,
            "episode_reward_recent": float(np.mean(list(self.episode_rewards)[-10:])) if len(self.episode_rewards) >= 10 else 0.0,
            # Sanitize best reward to avoid -Infinity in JSON/logs
            "best_episode_reward": (None if self.best_reward == -float("inf") else float(self.best_reward)),
            "current_episode_reward": self.current_episode_reward,
            "consecutive_failures": self.consecutive_failures,
            "circuit_breaker_active": self.circuit_breaker_state["active"],
        }

        # Env metrics (best-effort)
        m.update(self._extract_environment_metrics())

        # Health/perf (best-effort)
        m.update(self._get_health_metrics())

        # Model metrics (best-effort)
        m.update(self._extract_model_metrics())

        # Get risk and confidence from InfoBus
        try:
            session_risk = self._bus_get_multi(
                'session_risk',
                ['MarketModule', 'DynamicRiskController', 'TradingModeManager'],
                None,
            )
            if isinstance(session_risk, dict):
                risk_level = session_risk.get('risk_level', session_risk.get('state'))
                if risk_level:
                    m['risk_level'] = risk_level
                risk_score = session_risk.get('risk_score', session_risk.get('composite_score'))
                if risk_score is not None:
                    try:
                        m['risk_score'] = float(risk_score)
                    except (TypeError, ValueError):
                        pass

            dynamic_risk = self._bus_get_multi('risk_scaling', ['DynamicRiskController'], None)
            if 'risk_level' not in m and isinstance(dynamic_risk, dict):
                m['risk_level'] = dynamic_risk.get('current_mode')
            if 'risk_score' not in m and isinstance(dynamic_risk, dict):
                risk_scale = dynamic_risk.get('current_risk_scale')
                if risk_scale is not None:
                    try:
                        m['risk_score'] = float(risk_scale)
                    except (TypeError, ValueError):
                        pass

            committee_confidence = self._bus_get_multi(
                'committee_confidence',
                ['EnhancedVotingCommitteeCoordinator', 'VotingKernel'],
                None,
            )
            if committee_confidence is None:
                trade_vote_snapshot = self._bus_get_multi('trade_vote_v2', ['VotingKernel', 'StrategyArbiter'], None)
                if isinstance(trade_vote_snapshot, dict):
                    committee_confidence = trade_vote_snapshot.get('confidence')
            if committee_confidence is None:
                consensus_snapshot = self._bus_get_multi('committee_consensus', ['EnhancedVotingCommitteeCoordinator', 'VotingKernel'], None)
                if isinstance(consensus_snapshot, dict):
                    committee_confidence = consensus_snapshot.get('consensus_strength', consensus_snapshot.get('score'))
            if committee_confidence is not None:
                try:
                    m['decision_confidence'] = float(committee_confidence)
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass

        return sanitize_metrics(m)


    def _extract_environment_metrics(self) -> Dict[str, Any]:
        try:
            env = None
            if hasattr(self.training_env, "get_attr"):
                envs = self.training_env.get_attr("unwrapped", indices=[0])
                env = envs[0] if envs else None

            if env is None:
                return {"env_smartinfobus_status": "not_available"}

            if getattr(env, "smart_bus", None):
                ms = getattr(env, "market_state", None)
                
                # FIX: Read balance from SmartInfoBus (synced from Executor) instead of stale market_state
                balance = 0.0
                equity = 0.0
                instruments = []
                try:
                    account_state = self.smart_bus.get("account_state", "EnhancedCallback", default=None)
                    if isinstance(account_state, dict):
                        balance = float(account_state.get("balance", 0.0) or 0.0)
                        equity = float(account_state.get("equity", 0.0) or 0.0)
                    else:
                        # Fallback to env.market_state if bus data not available
                        balance = float(getattr(ms, "balance", 0.0)) if ms else 0.0
                        equity = balance
                    
                    # Get trading instruments from environment config
                    env_config = self.smart_bus.get("environment_config", "EnhancedCallback", default=None)
                    if isinstance(env_config, dict):
                        instruments = env_config.get("instruments", [])
                    if not instruments:
                        # Fallback to env attribute
                        instruments = getattr(env, "instruments", ["EUR/USD", "XAU/USD"])
                except Exception:
                    balance = float(getattr(ms, "balance", 0.0)) if ms else 0.0
                    equity = balance
                    instruments = getattr(env, "instruments", ["EUR/USD", "XAU/USD"])
                
                return {
                    "env_smartinfobus_status": "active",
                    "env_current_step": int(getattr(env, "current_step", 0)),
                    "env_drawdown": float(getattr(ms, "current_drawdown", 0.0)) if ms else 0.0,
                    "env_balance": balance,
                    "env_equity": equity,
                    "env_modules": int(len(getattr(getattr(env, "orchestrator", None), "modules", []))) if getattr(env, "orchestrator", None) else 0,
                    "instruments": instruments,
                }
            return {"env_smartinfobus_status": "none"}
        except Exception as e:
            return {"env_error": str(e)}

    def _get_health_metrics(self) -> Dict[str, Any]:
        try:
            hs: Dict[str, Any] = {}
            if self.health_monitor:
                st = self.health_monitor.check_system_health()
                hs.update({
                    "system_health_score": st.get("overall_score", 100),
                    "system_health_status": st.get("overall_status", "unknown"),
                    "system_cpu_percent": st.get("system", {}).get("cpu_percent", 0),
                    "system_memory_percent": st.get("system", {}).get("memory_percent", 0),
                    "healthy_modules": st.get("modules", {}).get("healthy_modules", 0),
                    "total_modules": st.get("modules", {}).get("total_modules", 0),
                    "health_monitoring_active": self.health_monitor.get_status().get("running", False),
                })
            pr = self.performance_tracker.generate_performance_report()
            mm = getattr(pr, "module_metrics", {})
            if mm:
                avg_ms = float(np.mean([v.get("avg_time_ms", 0.0) for v in mm.values()]))
                err_rate = float(np.mean([v.get("error_rate", 0.0) for v in mm.values()]))
                hs.update({
                    "performance_avg_ms": avg_ms,
                    "performance_success_rate": (1.0 - err_rate) * 100.0,
                })
            else:
                hs.update({"performance_avg_ms": 0.0, "performance_success_rate": 100.0})
            return hs
        except Exception as e:
            return {"health_error": str(e)}

    def _extract_model_metrics(self) -> Dict[str, Any]:
        try:
            out: Dict[str, Any] = {
                "model_device": str(getattr(self.model, "device", "unknown")),
            }

            # Robust learning rate extraction: prefer lr_schedule if present
            lr_float: float = 0.0
            try:
                lr_sched = getattr(self.model, "lr_schedule", None)
                if callable(lr_sched):
                    # SB3 uses progress_remaining in [1..0]; ensure we pass a float
                    pr_raw: Any = getattr(self.model, "_current_progress_remaining", None)
                    if isinstance(pr_raw, (int, float)):
                        progress_remaining: float = float(pr_raw)
                    elif getattr(pr_raw, "item", None):
                        # torch/np scalar
                        try:
                            progress_remaining = float(pr_raw.item())  # type: ignore[call-arg]
                        except Exception:
                            progress_remaining = 0.5
                    elif pr_raw is None:
                        # fallback: assume mid-training
                        progress_remaining = 0.5
                    else:
                        # last-resort conversion
                        try:
                            progress_remaining = float(pr_raw)  # type: ignore[arg-type]
                        except Exception:
                            progress_remaining = 0.5

                    lr_val: Any = lr_sched(progress_remaining)
                    if isinstance(lr_val, (int, float)):
                        lr_float = float(lr_val)
                    elif getattr(lr_val, "item", None):
                        lr_float = float(lr_val.item())  # type: ignore[call-arg]
                    else:
                        lr_float = float(lr_val)  # type: ignore[arg-type]
                else:
                    val: Any = getattr(self.model, "learning_rate", 0.0)
                    if isinstance(val, (int, float)):
                        lr_float = float(val)
                    elif getattr(val, "item", None):
                        lr_float = float(val.item())  # type: ignore[call-arg]
                    else:
                        lr_float = float(val)  # type: ignore[arg-type]
            except Exception:
                lr_float = 0.0
            out["learning_rate"] = lr_float

            # Optional SB3 metrics (best-effort)
            # SB3's logger.name_to_value resets between PPO updates (every n_steps)
            # We cache the last non-zero values to avoid always showing 0 between updates
            if getattr(self.model, "logger", None) and hasattr(self.model.logger, "name_to_value"):
                nd = self.model.logger.name_to_value
                
                # Check if this is a fresh PPO update (non-zero values present)
                clip_frac = nd.get("train/clip_fraction", 0)
                if clip_frac != 0 or nd.get("train/policy_gradient_loss", 0) != 0:
                    # Fresh update - cache all values
                    self._cached_training_metrics.update({
                        "clip_fraction": nd.get("train/clip_fraction", 0),
                        "explained_variance": nd.get("train/explained_variance", 0),
                        "policy_loss": nd.get("train/policy_gradient_loss", 0),  # Use policy_gradient_loss
                        "value_loss": nd.get("train/value_loss", 0),
                        "entropy_loss": nd.get("train/entropy_loss", 0),
                        "policy_gradient_loss": nd.get("train/policy_gradient_loss", 0),
                        "approx_kl": nd.get("train/approx_kl", 0),
                    })
                    self._last_ppo_update_step = self.n_calls
                    
                # Always return cached values (they persist between updates)
                out.update(self._cached_training_metrics)
                out["last_ppo_update_step"] = self._last_ppo_update_step
            return out
        except Exception:
            return {}

    def _perform_enhanced_health_check(self):
        summary = {"timestamp": datetime.now().isoformat(), "step": self.n_calls, "issues": []}
        try:
            st = self.health_monitor.check_system_health() if self.health_monitor else {}
            score = st.get("overall_score", 100)
            if score < 80:
                summary["issues"].append(f"System health low: {score}")
            sysm = st.get("system", {})
            if float(sysm.get("cpu_percent", 0)) > 85:
                summary["issues"].append(f"High CPU: {sysm['cpu_percent']:.1f}%")
            if float(sysm.get("memory_percent", 0)) > 90:
                summary["issues"].append(f"High RAM: {sysm['memory_percent']:.1f}%")
        except Exception as e:
            summary["issues"].append(f"Health monitor failed: {e}")

        summary["status"] = "OK" if not summary["issues"] else ("WARNING" if len(summary["issues"]) <= 2 else "CRITICAL")

        if summary["status"] != "OK":
            self.training_log.warning(format_operator_message_func(message="Health check issues", icon="[WARN]", issues=len(summary["issues"]), status=summary["status"]))
        else:
            self.training_log.info(format_operator_message_func(message="Health check passed", icon="[OK]"))

        self.health_alerts.append(summary)
        try:
            self.smart_bus.set("enhanced_health_status", summary, module="EnhancedTrainingCallback", thesis=f"Health={summary['status']}")
        except Exception:
            pass

    def _track_episode_progress(self):
        dones = self.locals.get("dones", None)
        rew = self.locals.get("rewards", None)

        # Rewards: handle scalar/array/list robustly
        if isinstance(rew, (list, tuple, np.ndarray)):
            if len(rew) > 0:
                self.current_episode_reward += float(np.asarray(rew).flatten()[0])
        elif isinstance(rew, (int, float, np.floating)):
            self.current_episode_reward += float(rew)

        # Dones: handle scalar/array/list robustly
        done_any = False
        if isinstance(dones, (list, tuple, np.ndarray)):
            done_any = bool(np.asarray(dones).astype(bool).any())
        elif isinstance(dones, (bool, np.bool_)):
            done_any = bool(dones)

        if done_any:
            self.episode_count += 1
            ep_rew = float(self.current_episode_reward)
            self.episode_rewards.append(ep_rew)

            if ep_rew > self.best_reward:
                self.best_reward = ep_rew
                # Disabled console output - using beautiful visualizer
                try:
                    self.smart_bus.set(
                        "training_new_best",
                        {"episode": self.episode_count, "reward": self.best_reward, "timestamp": datetime.now().isoformat()},
                        module="EnhancedTrainingCallback",
                        thesis="New best episode",
                    )
                except Exception:
                    pass

            # Episode stats disabled - using beautiful visualizer

            self.current_episode_reward = 0.0
            self.consecutive_failures = 0

    def _check_circuit_breaker(self) -> bool:
        conditions: List[str] = []
        if self.consecutive_failures > 15:
            conditions.append(f"Excessive failures: {self.consecutive_failures}")
        if len(self.health_alerts) >= 5:
            last5 = list(self.health_alerts)[-5:]
            crit = sum(1 for a in last5 if a.get("status") == "CRITICAL")
            if crit >= 3:
                conditions.append("Multiple critical health alerts")

        if conditions:
            self.circuit_breaker_state.update({"active": True, "failures": len(conditions)})
            try:
                self.smart_bus.set(
                    "training_emergency_stop",
                    {"conditions": conditions, "step": self.n_calls, "timestamp": datetime.now().isoformat()},
                    module="EnhancedTrainingCallback",
                    thesis="Circuit breaker",
                )
            except Exception:
                pass
            return True
        return False

    def _update_performance_tracking(self, metrics: Dict[str, Any]):
        snap = {
            "step": metrics.get("timestep", self.n_calls),
            "timestamp": datetime.now().isoformat(),
            "episode": self.episode_count,  # Add episode count for memory module
            "episodes": self.episode_count,  # Alias for compatibility
            "episode_reward_mean": metrics.get("episode_reward_mean", 0.0),
            "steps_per_second": metrics.get("steps_per_second", 0.0),
            "system_health_score": metrics.get("system_health_score", 100.0),
            "env_balance": metrics.get("env_balance", 0.0),
            "circuit_breaker_active": metrics.get("circuit_breaker_active", False),
        }
        self.performance_history.append(snap)

        try:
            self.performance_tracker.record_metric("EnhancedCallback", "sps", metrics.get("steps_per_second", 0.0), True)
            self.performance_tracker.record_metric("EnhancedCallback", "reward_mean", metrics.get("episode_reward_mean", 0.0), True)
        except Exception:
            pass

        # Throttle disk writes
        if self.n_calls % 1000 == 0:
            try:
                os.makedirs("logs/training", exist_ok=True)
                # Sanitize combined payload before writing to avoid NaN/Inf in JSON
                payload = {**sanitize_metrics(metrics), "snapshot": snap}
                with open(f"logs/training/enhanced_metrics_{datetime.now():%Y%m%d}.jsonl", "a") as f:
                    f.write(json.dumps(payload, default=str) + "\n")
            except Exception:
                pass

        try:
            self.smart_bus.set(
                "enhanced_performance",
                snap,
                module="EnhancedTrainingCallback",
                thesis=f"Perf sps={snap['steps_per_second']:.2f}, health={snap['system_health_score']}",
            )
        except Exception:
            pass

    def _tb_log(self, metrics: Dict[str, Any]):
        if not self._tb_ready:
            return
        try:
            # Use checkpoint-aware global step for TensorBoard so
            # resumed runs continue from the previous step count
            step = int(metrics.get("timestep", self.initial_num_timesteps + self.n_calls))
            # minimal TB logging; add records carefully to avoid noisy logs
            lr = metrics.get("learning_rate", None)
            if lr is not None:
                self.model.logger.record("custom/learning_rate", lr)
            self.model.logger.record("custom/steps_per_second", metrics.get("steps_per_second", 0.0))
            self.model.logger.record("custom/episode_reward_mean", metrics.get("episode_reward_mean", 0.0))
            self.model.logger.record("custom/step_ms_p50", metrics.get("step_ms_p50", 0.0))
            self.model.logger.record("custom/step_ms_p95", metrics.get("step_ms_p95", 0.0))
            self.model.logger.dump(step)
        except Exception:
            pass


# Kept for compatibility with older imports
class ModuleHealthTracker:
    def __init__(self, health_monitor: Optional[Any]):
        self.health_monitor = health_monitor
        self.module_health_history = defaultdict(lambda: deque(maxlen=50))

    def get_health_summary(self) -> Dict[str, Any]:
        try:
            if self.health_monitor:
                st = self.health_monitor.check_system_health()
                m = st.get("modules", {})
                tot = int(m.get("total_modules", 0))
                ok = int(m.get("healthy_modules", 0))
                return {
                    "total_modules": tot,
                    "healthy_modules": ok,
                    "degraded_modules": max(tot - ok, 0),
                    "health_percentage": (ok / max(tot, 1)) * 100.0,
                    "monitoring_active": True,
                }
        except Exception:
            pass
        return {"total_modules": 0, "healthy_modules": 0, "degraded_modules": 0, "health_percentage": 100.0, "monitoring_active": False}


__all__ = ["ModernEnhancedTrainingCallback", "ModuleHealthTracker"]
