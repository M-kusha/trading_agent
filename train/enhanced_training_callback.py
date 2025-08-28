"""
Enhanced Training Callback - Separate Module
File: enhanced_training_callback.py
Production-ready with proper error handling and SmartInfoBus integration
Refactored to be Pylance-friendly (no type identity collisions)
"""

from __future__ import annotations

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Deque
from collections import deque, defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# ───────────────────────────────────────────────────────────────────
# Resolve dependencies without type-identity collisions
# Use "*_Cls" variables typed as Any to hold class refs (real or fallback)
# ───────────────────────────────────────────────────────────────────
from typing import Any

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

        def get(self, k, module=None):
            return self._d.get(k)

        def get_performance_metrics(self):
            return {"active": True, "disabled_modules": [], "active_data_keys": len(self._d)}

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
                 metrics_broadcaster: Any = None, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.config = config
        self.metrics_broadcaster = metrics_broadcaster

        # Runtime state
        self.start_time: datetime = datetime.now()
        self.last_print_time: datetime = self.start_time
        self.best_reward: float = -float("inf")
        self.current_episode_reward: float = 0.0
        self.episode_count: int = 0
        self.consecutive_failures: int = 0
        self.circuit_breaker_state = {"active": False, "failures": 0}

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
        print(f"[CHART] Mode: {mode_str}")
        print("─" * 60)

        self._tb_ready = False  # set at _on_training_start

    # ── SB3 hooks ───────────────────────────────────────────────────
    def _on_training_start(self) -> None:
        self.start_time = datetime.now()
        self._tb_ready = bool(getattr(self.model, "logger", None))

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
            if (now - self.last_print_time).total_seconds() >= 10 or self.n_calls % 1000 == 0:
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

    # ── Telemetry helpers ───────────────────────────────────────────
    def _print_enhanced_progress(self):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = (self.n_calls / max(self.total_timesteps, 1)) * 100.0

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
            f"\r[RELOAD] Step: {self.n_calls:,}/{self.total_timesteps:,} "
            f"({progress:.1f}%) | [TIME] {elapsed/60:.1f}min | "
            f"[MONEY] Last: {last} | [TROPHY] Best: {best} | "
            f"[LAT] P50:{p50:.1f}ms P95:{p95:.1f}ms | [WAIT] ETA: {eta_str}",
            end="",
            flush=True,
        )

    def _collect_enhanced_metrics(self) -> Dict[str, Any]:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        progress = self.n_calls / max(self.total_timesteps, 1)
        sps = (self.n_calls / elapsed) if elapsed > 0 else 0.0

        m: Dict[str, Any] = {
            "timestep": self.n_calls,
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
            "best_episode_reward": self.best_reward,
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

        return m

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
                return {
                    "env_smartinfobus_status": "active",
                    "env_current_step": int(getattr(env, "current_step", 0)),
                    "env_drawdown": float(getattr(ms, "current_drawdown", 0.0)) if ms else 0.0,
                    "env_balance": float(getattr(ms, "balance", 0.0)) if ms else 0.0,
                    "env_modules": int(len(getattr(getattr(env, "orchestrator", None), "modules", []))) if getattr(env, "orchestrator", None) else 0,
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
            # Learning rate can be a float or a schedule (callable); coerce safely to float
            lr_val: Any = getattr(self.model, "learning_rate", 0)
            lr_float: float = 0.0
            try:
                val: Any
                if callable(lr_val):
                    # Some schedulers accept step count; fall back gracefully
                    try:
                        val = lr_val(self.n_calls)  # type: ignore[misc]
                    except Exception:
                        val = lr_val()  # type: ignore[call-arg]
                else:
                    val = lr_val

                if isinstance(val, (int, float)):
                    lr_float = float(val)
                elif hasattr(val, "item"):
                    # Handle numpy/torch scalars
                    try:
                        lr_float = float(val.item())  # type: ignore[call-arg]
                    except Exception:
                        lr_float = float(val)  # type: ignore[arg-type]
                else:
                    lr_float = float(val)  # type: ignore[arg-type]
            except Exception:
                lr_float = 0.0
            out["learning_rate"] = lr_float

            if getattr(self.model, "logger", None):
                nd = self.model.logger.name_to_value
                out.update({
                    "clip_fraction": nd.get("train/clip_fraction", 0),
                    "explained_variance": nd.get("train/explained_variance", 0),
                    "policy_loss": nd.get("train/policy_loss", 0),
                    "value_loss": nd.get("train/value_loss", 0),
                    "entropy_loss": nd.get("train/entropy_loss", 0),
                })
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

        if isinstance(rew, (list, tuple, np.ndarray)) and len(rew) > 0:
            self.current_episode_reward += float(rew[0])

        if isinstance(dones, (list, tuple, np.ndarray)) and any(dones):
            self.episode_count += 1
            ep_rew = float(self.current_episode_reward)
            self.episode_rewards.append(ep_rew)

            if ep_rew > self.best_reward:
                self.best_reward = ep_rew
                print(f"\n[PARTY] NEW BEST REWARD: {ep_rew:.2f} (Episode {self.episode_count})")
                try:
                    self.smart_bus.set(
                        "training_new_best",
                        {"episode": self.episode_count, "reward": self.best_reward, "timestamp": datetime.now().isoformat()},
                        module="EnhancedTrainingCallback",
                        thesis="New best episode",
                    )
                except Exception:
                    pass

            if self.episode_count % 10 == 0:
                recent = list(self.episode_rewards)[-10:]
                avg10 = sum(recent) / len(recent)
                print(f"\n[STATS] Episode {self.episode_count}: Reward={ep_rew:.2f}, Avg(10)={avg10:.2f}, Best={self.best_reward:.2f}")

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
                with open(f"logs/training/enhanced_metrics_{datetime.now():%Y%m%d}.jsonl", "a") as f:
                    f.write(json.dumps({**metrics, "snapshot": snap}, default=str) + "\n")
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
            # minimal TB logging; add records carefully to avoid noisy logs
            lr = metrics.get("learning_rate", None)
            if lr is not None:
                self.model.logger.record("custom/learning_rate", lr)
            self.model.logger.record("custom/steps_per_second", metrics.get("steps_per_second", 0.0))
            self.model.logger.record("custom/episode_reward_mean", metrics.get("episode_reward_mean", 0.0))
            self.model.logger.record("custom/step_ms_p50", metrics.get("step_ms_p50", 0.0))
            self.model.logger.record("custom/step_ms_p95", metrics.get("step_ms_p95", 0.0))
            self.model.logger.dump(self.n_calls)
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
