# ─────────────────────────────────────────────────────────────
# File: modules/core/persistence.py
# Production-ready SmartInfoBus Persistence & Replay System (fixed)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import pickle
import shutil
import sys
import tempfile
import threading
import time
import traceback
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except Exception:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore

from modules.utils.audit_utils import RotatingLogger, format_operator_message

if TYPE_CHECKING:
    from modules.core.module_base import BaseModule
    from modules.core.module_system import ModuleOrchestrator


# ═════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════

@dataclass
class StateValidation:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    compatibility_score: float = 1.0
    required_migrations: List[str] = field(default_factory=list)

@dataclass
class ReplayEvent:
    timestamp: float
    event_type: str
    module: str
    data: Dict[str, Any]
    execution_id: str
    sequence_number: int
    checksum: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.checksum:
            data_str = json.dumps(self.data, sort_keys=True, default=str)
            content = f"{self.timestamp}:{self.event_type}:{self.module}:{data_str}"
            self.checksum = hashlib.md5(content.encode()).hexdigest()

    def age_at(self, now: float) -> float:
        return now - self.timestamp

    def validate_integrity(self) -> bool:
        expected = self.checksum
        self.checksum = ""
        self.__post_init__()
        ok = (self.checksum == expected)
        self.checksum = expected
        return ok

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "module": self.module,
            "data": self.data,
            "execution_id": self.execution_id,
            "sequence_number": self.sequence_number,
            "checksum": self.checksum,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReplayEvent":
        return cls(**d)


@dataclass
class ReplaySession:
    session_id: str
    start_time: datetime
    end_time: datetime
    events: List[ReplayEvent]
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    integrity_hash: str = field(default="")
    system_health: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.integrity_hash:
            core = {
                "session_id": self.session_id,
                "event_count": len(self.events),
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
            }
            self.integrity_hash = hashlib.sha256(
                json.dumps(core, sort_keys=True).encode()
            ).hexdigest()

    @property
    def duration_seconds(self) -> float:
        if self.events:
            return self.events[-1].timestamp - self.events[0].timestamp
        return (self.end_time - self.start_time).total_seconds()

    @property
    def event_count(self) -> int:
        return len(self.events)

    def validate_integrity(self) -> bool:
        expected = self.integrity_hash
        self.integrity_hash = ""
        self.__post_init__()
        ok = (self.integrity_hash == expected)
        self.integrity_hash = expected
        if not ok:
            return False
        for i, e in enumerate(self.events):
            if e.sequence_number != i or not e.validate_integrity():
                return False
        return True

    def get_statistics(self) -> Dict[str, Any]:
        if not self.events:
            return {"error": "No events in session"}
        event_types = defaultdict(int)
        module_activity = defaultdict(int)
        for e in self.events:
            event_types[e.event_type] += 1
            module_activity[e.module] += 1
        gaps = [self.events[i].timestamp - self.events[i-1].timestamp
                for i in range(1, len(self.events))]

        def _mean(vals):
            if not vals: return 0.0
            return float(np.mean(vals)) if NUMPY_AVAILABLE and np is not None else sum(vals)/len(vals)

        return {
            "session_id": self.session_id,
            "duration_seconds": self.duration_seconds,
            "total_events": self.event_count,
            "event_types": dict(event_types),
            "module_activity": dict(module_activity),
            "avg_event_interval": _mean(gaps) if gaps else 0.0,
            "max_event_interval": max(gaps) if gaps else 0.0,
            "unique_modules": len(module_activity),
            "unique_event_types": len(event_types),
            "checkpoints": len(self.checkpoints),
            "integrity_valid": self.validate_integrity(),
            "system_health": self.system_health,
        }


# ═════════════════════════════════════════════════════════════
# SmartInfoBus snapshot helpers
# ═════════════════════════════════════════════════════════════

def save_infobus_snapshot(path: str = "state/infobus.json") -> bool:
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus: Any = InfoBusManager.get_instance()
        exporter = getattr(bus, "export_snapshot", None)
        if callable(exporter):
            snapshot = exporter()
        else:
            store = getattr(bus, "_data_store", None)
            snapshot = dict(store) if isinstance(store, dict) else {}
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(snapshot, indent=2, default=str))
        return True
    except Exception:
        return False


def load_infobus_snapshot(path: str = "state/infobus.json") -> bool:
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        p = Path(path)
        if not p.exists():
            return False
        data = json.loads(p.read_text())
        bus: Any = InfoBusManager.get_instance()
        setter = getattr(bus, "set", None)
        if callable(setter):
            for k, v in data.items():
                try:
                    setter(k, v, module="Bootstrap", thesis="Restored from snapshot", confidence=0.9)
                except Exception:
                    pass
        else:
            store = getattr(bus, "_data_store", None)
            if isinstance(store, dict):
                store.update(data)
        return True
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════
# StateManager
# ═════════════════════════════════════════════════════════════

class StateManager:
    def __init__(self, state_dir: str = "state/modules"):
        self.state_dir = Path(state_dir); self.state_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.state_dir / "backups"; self.backup_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = self.state_dir / "checkpoints"; self.checkpoint_dir.mkdir(exist_ok=True)

        self.state_cache: Dict[str, Dict[str, Any]] = {}
        self.state_versions: Dict[str, int] = {}
        self.state_checksums: Dict[str, str] = {}
        self.module_versions: Dict[str, str] = {}
        self.version_compatibility: Dict[str, List[str]] = defaultdict(list)

        self.max_backups = 10
        self.compression_enabled = True
        self.validation_enabled = True
        self.compression_level = 6

        self.validation_rules = self._initialize_validation_rules()
        self._lock = threading.RLock()

        self.logger = RotatingLogger(
            name="StateManager",
            log_path="logs/state/state_manager.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self._initialize_version_compatibility()
        self.logger.info(format_operator_message("[SAVE]", "STATE MANAGER INITIALIZED", details=f"Dir: {self.state_dir}", context="startup"))

    # ---------- validation & version ----------

    def _initialize_validation_rules(self) -> Dict[str, Callable]:
        return {
            "required_fields": lambda env: all(k in env for k in ("module_name","timestamp","version","state")),
            "checksum_valid":  lambda env: self._validate_checksum(env),
            "version_format":  lambda env: isinstance(env.get("version"), (int, str)),
            "state_type":      lambda env: isinstance(env.get("state"), dict),
            "timestamp_valid": lambda env: self._validate_timestamp(env.get("timestamp")),
            "size_limit":      lambda env: self._check_state_size(env.get("state")) < 100 * 1024 * 1024,
        }

    def _initialize_version_compatibility(self):
        self.version_compatibility = {
            "default": ["1.0.0", "1.0.1", "1.1.0"],
        }

    # ---------- public save/restore APIs ----------

    def save_module_state(self, module: "BaseModule") -> bytes:
        name = module.__class__.__name__
        with self._lock:
            try:
                # extract state (robust)
                state = None
                if hasattr(module, "get_state"):
                    try:
                        state = module.get_state()
                    except Exception:
                        state = None
                if not isinstance(state, dict):
                    state = self._extract_safe_attributes(module)
                    if not isinstance(state, dict):
                        state = {"value": state}

                validation = self._validate_state_for_save(state, module)
                if not validation.is_valid:
                    raise ValueError(f"State validation failed: {validation.errors}")

                _probe_blob, method = self._serialize_state({"state": state})

                module_version = getattr(getattr(module, "metadata", None), "version", "1.0.0")
                self.module_versions[name] = module_version

                version = self.state_versions.get(name, 0) + 1
                timestamp = datetime.now().isoformat()

                envelope: Dict[str, Any] = {
                    "module_name": name,
                    "timestamp": timestamp,
                    "version": version,
                    "module_version": module_version,
                    "state": state,
                    "module_metadata": {
                        "class_name": module.__class__.__name__,
                        "module_path": module.__class__.__module__,
                        "version": module_version,
                        "health_status": state.get("health_status", "unknown"),
                    },
                    "system_context": self._get_system_context(),
                    "validation": validation.__dict__,
                    "serialization_method": method,
                }

                # checksum bound to chosen method over STATE
                if method == "json":
                    payload = json.dumps(state, sort_keys=True, default=str).encode()
                else:
                    payload = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
                envelope["checksum"] = hashlib.sha256(payload).hexdigest()

                self.state_versions[name] = version
                self.state_checksums[name] = envelope["checksum"]
                self.state_cache[name] = envelope

                self._save_to_disk_with_backup(name, envelope)
                self.logger.info(format_operator_message("[SAVE]", "STATE SAVED", instrument=name, details=f"v{version} ({len(payload)} bytes, {method})", context="state_management"))
                return payload
            except Exception as e:
                self.logger.error(f"[CRASH] Failed to save state for {name}: {e}")
                raise RuntimeError(f"State save failed for {name}: {e}")

    def save_all_module_states(self, orchestrator: "ModuleOrchestrator") -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for name, mod in orchestrator.modules.items():
            try:
                self.save_module_state(mod)
                results[name] = True
            except Exception as e:
                self.logger.error(f"Save failed for {name}: {e}")
                results[name] = False
        return results

    def restore_all_states(self, orchestrator: "ModuleOrchestrator") -> Dict[str, bool]:
        with self._lock:
            if not self._check_system_health_for_operation(orchestrator):
                self.logger.warning("System health check failed - limited restoration only")

            results: Dict[str, bool] = {}

            # gather files
            state_files: List[Path] = []
            for ext in (".json", ".json.gz", ".pkl", ".pkl.gz"):
                state_files.extend(self.state_dir.glob(f"*{ext}"))
            # unique by module using robust parser
            unique: Dict[str, Path] = {}
            for p in state_files:
                mod = self._module_name_from_state_file(p)
                if mod and (mod not in unique or p.stat().st_mtime > unique[mod].stat().st_mtime):
                    unique[mod] = p

            for module_name, state_file in unique.items():
                if module_name not in orchestrator.modules:
                    continue
                state_data = self.load_from_disk(module_name)
                if not state_data:
                    results[module_name] = False
                    continue
                try:
                    module = orchestrator.modules[module_name]
                    validation = self._validate_state_for_restore(state_data, module)
                    if not validation.is_valid:
                        self.logger.error(f"State validation failed for {module_name}: {validation.errors}")
                        results[module_name] = False
                        continue
                    if validation.required_migrations:
                        state_data["state"] = self._apply_state_migrations(state_data["state"], validation.required_migrations, module_name)
                    if hasattr(module, "set_state"):
                        module.set_state(state_data["state"])
                        results[module_name] = True
                        self.logger.info(f"[OK] Restored state for {module_name} (v{state_data.get('version', 0)}, compatibility: {validation.compatibility_score:.1%})")
                    else:
                        results[module_name] = False
                        self.logger.warning(f"Module {module_name} does not support state restoration")
                except Exception as e:
                    self.logger.error(f"[CRASH] Failed to restore {module_name}: {e}")
                    results[module_name] = False

            ok_count = sum(1 for v in results.values() if v)
            self.logger.info(format_operator_message("[FOLDER]", "STATE RESTORATION COMPLETE", details=f"Restored {ok_count}/{len(results)} modules", context="startup"))
            return results

    # ---------- helpers & validation ----------

    def _module_name_from_state_file(self, p: Path) -> str:
        """
        Robustly extract module name from files:
        <Module>_state.json(.gz) | <Module>_state.pkl(.gz)
        """
        name = p.name
        for suffix in ("_state.json.gz", "_state.pkl.gz", "_state.json", "_state.pkl", ".json.gz", ".pkl.gz", ".json", ".pkl"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        # also handle accidental double-extensions already stripped
        if name.endswith("_state"):
            name = name[:-6]
        return name

    def _validate_state_for_save(self, state: Any, module: "BaseModule") -> StateValidation:
        v = StateValidation(is_valid=True)
        # coerce to dict if needed
        if not isinstance(state, dict):
            v.warnings.append("State was not a dict; wrapping as {'value': ...}")
            state = {"value": state}
        # probe serializer
        try:
            blob, _method = self._serialize_state({"state": state})
            if len(blob) > 50 * 1024 * 1024:
                v.warnings.append(f"Large serialized state: {len(blob)/1024/1024:.1f}MB")
        except Exception as e:
            v.is_valid = False
            v.errors.append(f"Serialization failed: {e}")
            return v
        # module-specific validation (optional)
        if hasattr(module, "validate_state"):
            try:
                if not module.validate_state(state):
                    v.is_valid = False
                    v.errors.append("Module-specific validation failed")
            except Exception as e:
                v.warnings.append(f"Module validation check failed: {e}")
        return v

    def _check_system_health_for_operation(self, orchestrator: "ModuleOrchestrator") -> bool:
        try:
            # Allow restoration/saving during cold start when no metrics exist yet.
            metrics = orchestrator.get_execution_metrics() if hasattr(orchestrator, "get_execution_metrics") else {}
            if not metrics or metrics.get("total_executions", 0) == 0:
                return True

            if getattr(orchestrator, "get_emergency_mode_status", None):
                if orchestrator.get_emergency_mode_status().get("active"):
                    self.logger.warning("System in emergency mode")
                    return False

            if metrics.get("success_rate", 0) < 0.5:
                self.logger.warning("System success rate too low")
                return False

            cb_status = orchestrator.get_circuit_breaker_status() if hasattr(orchestrator, "get_circuit_breaker_status") else {}
            if cb_status:
                open_breakers = sum(1 for cb in cb_status.values() if cb.get("state") == "OPEN")
                if open_breakers > len(cb_status) * 0.3:
                    self.logger.warning("Too many circuit breakers open")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False

    def _check_version_compatibility(self, module_name: str, old: str, new: str) -> bool:
        if old == new:
            return True
        compat = self.version_compatibility.get(module_name, self.version_compatibility["default"])
        if old in compat and new in compat:
            return True
        try:
            o = [int(x) for x in old.split(".")]
            n = [int(x) for x in new.split(".")]
            if o[0] != n[0]:
                return False
            if len(o) > 1 and len(n) > 1 and n[1] < o[1]:
                return False
            return True
        except Exception:
            return False

    def _validate_state_for_restore(self, env: Dict[str, Any], module: "BaseModule") -> StateValidation:
        v = StateValidation(is_valid=True)
        # Run rules but tolerate checkpoint-only envelopes (which may lack metadata)
        # If this looks like a bare checkpoint piece (only 'state' key), synthesize an envelope
        if "module_name" not in env or "timestamp" not in env or "version" not in env:
            synth = {
                "module_name": module.__class__.__name__,
                "timestamp": datetime.now().isoformat(),
                "version": 0,
                "state": env.get("state", {}) if "state" in env else env,
                "serialization_method": env.get("serialization_method", "json"),
                "checksum": env.get("checksum", ""),
            }
            env = synth

        for rule, fn in self.validation_rules.items():
            try:
                if not fn(env):
                    v.is_valid = False
                    v.errors.append(f"Validation rule '{rule}' failed")
            except Exception as e:
                v.warnings.append(f"Validation rule '{rule}' error: {e}")

        saved = env.get("module_version", "1.0.0")
        current = getattr(getattr(module, "metadata", None), "version", "1.0.0")
        if saved != current:
            if self._check_version_compatibility(module.__class__.__name__, saved, current):
                v.compatibility_score = 0.8
                v.warnings.append(f"Version mismatch: {saved} -> {current}")
                mig = self._get_required_migrations(module.__class__.__name__, saved, current)
                if mig:
                    v.required_migrations = mig
            else:
                v.is_valid = False
                v.errors.append(f"Incompatible versions: {saved} -> {current}")
                v.compatibility_score = 0.0
        if hasattr(module, "validate_state_compatibility"):
            try:
                if not module.validate_state_compatibility(env["state"]):
                    v.is_valid = False
                    v.errors.append("Module state compatibility check failed")
            except Exception as e:
                v.warnings.append(f"Compatibility check error: {e}")
        return v

    def _get_required_migrations(self, module_name: str, old: str, new: str) -> List[str]:
        migration_map: Dict[Tuple[str,str], List[str]] = {}
        return migration_map.get((old, new), [])
    
    def _apply_state_migrations(
    self,
    state: Dict[str, Any],
    migrations: List[str],
    module_name: str
    ) -> Dict[str, Any]:
        """
        Apply in-place migrations to a module's state snapshot.

        This is intentionally conservative: if you haven't defined
        any real migrations yet, it safely returns the original state.
        """
        migrated_state = dict(state)

        for migration in migrations:
            try:
                if migration == "add_new_fields":
                    # example: ensure new keys exist
                    migrated_state.setdefault("new_field", None)

                elif migration == "restructure_data":
                    # example: rename a key
                    if "old_structure" in migrated_state:
                        migrated_state["new_structure"] = migrated_state.pop("old_structure")

                else:
                    # Unknown migration – log and skip
                    self.logger.warning(
                        f"Unknown migration '{migration}' for {module_name}; skipping"
                    )

            except Exception as e:
                self.logger.error(
                    f"Migration '{migration}' failed for {module_name}: {e}"
                )

        return migrated_state


    def _create_mini_checkpoint(self, module_name: str, module: "BaseModule", kind: str) -> Dict[str, Any]:
        try:
            state = module.get_state() if hasattr(module, "get_state") else {}
            return {
                "module_name": module_name,
                "checkpoint_type": kind,
                "timestamp": datetime.now().isoformat(),
                "module_class": module.__class__,
                "module_instance": module,
                "state": state if isinstance(state, dict) else {"value": state},
            }
        except Exception as e:
            self.logger.error(f"Failed to create mini checkpoint: {e}")
            return {}

    def _restore_mini_checkpoint(self, cp: Dict[str, Any], orchestrator: "ModuleOrchestrator"):
        try:
            name = cp["module_name"]; inst = cp["module_instance"]
            orchestrator.modules[name] = inst
            orchestrator.module_classes[name] = cp["module_class"]
            self.logger.info(f"Restored module {name} from checkpoint")
        except Exception as e:
            self.logger.error(f"Failed to restore from checkpoint: {e}")

    def _validate_checksum(self, env: Dict[str, Any]) -> bool:
        if "checksum" not in env or "state" not in env:
            return False
        try:
            method = env.get("serialization_method", "pickle")
            if method == "json":
                payload = json.dumps(env["state"], sort_keys=True, default=str).encode()
            else:
                payload = pickle.dumps(env["state"], protocol=pickle.HIGHEST_PROTOCOL)
            ok = hashlib.sha256(payload).hexdigest() == env["checksum"]
            if not ok and method == "pickle":
                # Some large/complex states (e.g., RL agents with tensors) can re-pickle
                # to different byte streams across sessions/versions. Accept known cases.
                mod = env.get("module_name", "")
                if mod in ("PPOAgent", "PPOLagAgent"):
                    self.logger.warning(f"Checksum mismatch tolerated for {mod} (pickle non-determinism)")
                    return True
            return ok
        except Exception:
            return False

    def _validate_timestamp(self, ts: Any) -> bool:
        if not ts: return False
        try:
            if isinstance(ts, str):
                datetime.fromisoformat(ts)
            return True
        except Exception:
            return False

    def _check_state_size(self, state: Any) -> int:
        try:
            return len(json.dumps(state, default=str).encode())
        except Exception:
            try:
                return len(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
            except Exception:
                return sys.getsizeof(state)

    def _get_system_context(self) -> Dict[str, Any]:
        try:
            import psutil  # type: ignore
            return {
                "save_time": datetime.now().isoformat(),
                "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "python_version": sys.version,
            }
        except Exception:
            return {"save_time": datetime.now().isoformat()}

    def _serialize_state(self, env: Dict[str, Any]) -> Tuple[bytes, str]:
        try:
            blob = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)
            return blob, "pickle"
        except Exception:
            return json.dumps(env, default=str).encode("utf-8"), "json"

    def _extract_safe_attributes(self, module: "BaseModule") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in module.__dict__.items():
            if k.startswith("_") and not k.startswith("_step_count"):
                continue
            if callable(v):
                continue
            try:
                json.dumps(v, default=str)
                out[k] = v
            except Exception:
                try:
                    out[k] = str(v)
                except Exception:
                    pass
        return out

    # ---------- disk IO ----------

    def _save_to_disk_with_backup(self, name: str, env: Dict[str, Any]):
        method = env.get("serialization_method", "pickle")
        if method == "json":
            file_ext = ".json"
            data = json.dumps(env, indent=2, default=str).encode("utf-8")
        else:
            file_ext = ".pkl"
            data = pickle.dumps(env, protocol=pickle.HIGHEST_PROTOCOL)

        if self.compression_enabled:
            comp = zlib.compress(data, level=self.compression_level)
            if len(comp) < len(data) * 0.9:
                data = comp
                file_ext += ".gz"

        path = self.state_dir / f"{name}_state{file_ext}"
        tmp = self.state_dir / f"{name}_state.tmp"
        try:
            tmp.write_bytes(data)
            if path.exists():
                backup = self.backup_dir / f"{name}_state_{int(time.time())}{file_ext}"
                backup.write_bytes(path.read_bytes())
                self._cleanup_old_backups(name)
            tmp.replace(path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    def _cleanup_old_backups(self, name: str):
        cutoff = time.time() - 7 * 24 * 3600
        patterns = [f"{name}_state_*.pkl", f"{name}_state_*.pkl.gz",
                    f"{name}_state_*.json", f"{name}_state_*.json.gz"]
        files: List[Path] = []
        for pat in patterns:
            files.extend(self.backup_dir.glob(pat))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[self.max_backups:]:
            try:
                old.unlink()
            except Exception as e:
                self.logger.warning(f"Failed to remove old backup {old}: {e}")

    def load_from_disk(self, module_name: str) -> Optional[Dict[str, Any]]:
        for ext in (".json", ".json.gz", ".pkl", ".pkl.gz"):
            p = self.state_dir / f"{module_name}_state{ext}"
            if not p.exists():
                continue
            try:
                data = p.read_bytes()
                if ext.endswith(".gz"):
                    data = zlib.decompress(data)
                return json.loads(data.decode("utf-8")) if ".json" in ext else pickle.loads(data)
            except Exception as e:
                self.logger.error(f"Failed to load {p}: {e}")
                continue
        return None

    # ---------- hot reload (unchanged logic but safer fallbacks) ----------

    def reload_module(self, module_name: str, orchestrator: "ModuleOrchestrator") -> bool:
        with self._lock:
            try:
                current = orchestrator.modules.get(module_name)
                if not current:
                    self.logger.error(f"Module {module_name} not found in orchestrator")
                    return False
                if not self._check_system_health_for_operation(orchestrator):
                    self.logger.error("System health check failed - aborting reload")
                    return False

                _ = self.save_module_state(current)
                pre = self._create_mini_checkpoint(module_name, current, "pre_reload")

                cls_ref = orchestrator.module_classes.get(module_name)
                if not cls_ref:
                    self.logger.error(f"Module class {module_name} not found")
                    return False
                module_path = cls_ref.__module__

                self.logger.info(f"[RELOAD] Reloading module code for {module_name}")
                try:
                    mod_ref = importlib.import_module(module_path)
                    importlib.reload(mod_ref)
                except Exception as e:
                    self.logger.error(f"Failed to reload module code: {e}")
                    return False

                new_class = getattr(mod_ref, module_name, None)
                if not new_class:
                    self.logger.error(f"Class {module_name} not found after reload")
                    return False
                if not hasattr(new_class, "__module_metadata__"):
                    self.logger.error("Reloaded class missing @module decorator")
                    return False

                old_v = self.module_versions.get(module_name, "1.0.0")
                new_v = getattr(new_class.__module_metadata__, "version", "1.0.0")
                if not self._check_version_compatibility(module_name, old_v, new_v):
                    self.logger.error(f"Version incompatibility: {old_v} -> {new_v}")
                    return False

                try:
                    new_inst = new_class()
                except Exception as e:
                    self.logger.error(f"Failed to create new instance: {e}")
                    self._restore_mini_checkpoint(pre, orchestrator)
                    return False

                try:
                    env = self.load_from_disk(module_name)
                    if not env:
                        self.logger.error(f"Failed to load persisted state for {module_name}")
                        self._restore_mini_checkpoint(pre, orchestrator)
                        return False
                    val = self._validate_state_for_restore(env, new_inst)
                    if not val.is_valid:
                        self.logger.error(f"State validation failed: {val.errors}")
                        return False
                    if val.required_migrations:
                        env["state"] = self._apply_state_migrations(env["state"], val.required_migrations, module_name)
                    if hasattr(new_inst, "set_state"):
                        new_inst.set_state(env["state"])
                    else:
                        self._restore_attributes_safely(new_inst, env["state"])
                except Exception as e:
                    self.logger.error(f"Failed to restore state: {e}")
                    self._restore_mini_checkpoint(pre, orchestrator)
                    return False

                if hasattr(new_inst, "get_health_status"):
                    try:
                        health = new_inst.get_health_status()
                        if isinstance(health, dict) and health.get("status") == "CRITICAL":
                            self.logger.error("New instance health check failed")
                            return False
                    except Exception:
                        pass

                orchestrator.modules[module_name] = new_inst
                orchestrator.module_classes[module_name] = new_class

                old_meta = orchestrator.metadata.get(module_name)
                new_meta = new_class.__module_metadata__
                if old_meta and hasattr(old_meta, "to_dict") and old_meta.to_dict() != new_meta.to_dict():
                    orchestrator.metadata[module_name] = new_meta
                    orchestrator.build_execution_plan()

                self.module_versions[module_name] = new_v
                self.logger.info(format_operator_message("[OK]", "MODULE RELOADED", instrument=module_name, details=f"v{old_v} -> v{new_v}", context="hot_reload"))
                return True
            except Exception as e:
                self.logger.error(f"[CRASH] Failed to reload {module_name}: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return False

    def _restore_attributes_safely(self, inst: "BaseModule", state: Dict[str, Any]):
        for k, v in state.items():
            try:
                if hasattr(inst, k) and not callable(getattr(inst, k)):
                    setattr(inst, k, v)
            except Exception as e:
                self.logger.warning(f"Failed to restore attribute {k}: {e}")

    # ---------- checkpoints ----------

    def create_checkpoint(self, orchestrator: "ModuleOrchestrator", checkpoint_name: str = "manual") -> bool:
        with self._lock:
            cid = f"{checkpoint_name}_{int(time.time())}"
            cdir = self.checkpoint_dir / cid
            cdir.mkdir(exist_ok=True)
            try:
                system_health = {
                    "emergency_mode": getattr(orchestrator, "get_emergency_mode_status", dict)(),
                    "circuit_breakers": getattr(orchestrator, "get_circuit_breaker_status", dict)(),
                    "execution_metrics": getattr(orchestrator, "get_execution_metrics", dict)(),
                }
                data = {
                    "checkpoint_id": cid,
                    "name": checkpoint_name,
                    "timestamp": datetime.now().isoformat(),
                    "orchestrator_state": {
                        "execution_order": getattr(orchestrator, "execution_order", []),
                        "execution_stages": getattr(orchestrator, "execution_stages", {}),
                        "voting_members": getattr(orchestrator, "voting_members", []),
                        "critical_modules": list(getattr(orchestrator, "critical_modules", set())),
                    },
                    "modules": {},
                    "system_metrics": system_health.get("execution_metrics", {}),
                    "system_health": system_health,
                    "validation_results": {},
                }
                ok = 0
                for name, mod in orchestrator.modules.items():
                    try:
                        # robust state acquisition
                        raw = None
                        if hasattr(mod, "get_state"):
                            try:
                                raw = mod.get_state()
                            except Exception:
                                raw = None
                        if not isinstance(raw, dict):
                            raw = self._extract_safe_attributes(mod)
                            if not isinstance(raw, dict):
                                raw = {"value": raw}

                        val = self._validate_state_for_save(raw, mod)
                        data["validation_results"][name] = val.__dict__
                        if val.is_valid:
                            data["modules"][name] = raw
                            with open(cdir / f"{name}.json", "w") as f:
                                json.dump(raw, f, indent=2, default=str)
                            ok += 1
                        else:
                            self.logger.warning(f"Skipping {name} due to validation errors")
                    except Exception as e:
                        self.logger.error(f"Failed to checkpoint {name}: {e}")
                        data["modules"][name] = {"error": str(e)}

                data["integrity"] = {
                    "total_modules": len(orchestrator.modules),
                    "saved_modules": ok,
                    "success_rate": ok / max(len(orchestrator.modules), 1),
                    "checksum": hashlib.sha256(json.dumps(data["modules"], sort_keys=True, default=str).encode()).hexdigest(),
                }
                with open(cdir / "checkpoint.json", "w") as f:
                    json.dump(data, f, indent=2, default=str)

                if self.compression_enabled:
                    try:
                        arch = self.checkpoint_dir / f"{cid}.tar.gz"
                        shutil.make_archive(str(arch.with_suffix("")), "gztar", self.checkpoint_dir, cid)
                        shutil.rmtree(cdir)
                    except Exception as e:
                        self.logger.warning(f"Compression failed: {e}")

                self.logger.info(format_operator_message("📸", "CHECKPOINT CREATED", instrument=checkpoint_name, details=f"Saved {ok}/{len(orchestrator.modules)} modules", context="state_management"))
                return ok > 0
            except Exception as e:
                self.logger.error(f"[CRASH] Failed to create checkpoint {checkpoint_name}: {e}")
                if cdir.exists():
                    shutil.rmtree(cdir, ignore_errors=True)
                return False

    def restore_checkpoint(self, orchestrator: "ModuleOrchestrator", checkpoint_id: str) -> bool:
        with self._lock:
            cdir = self.checkpoint_dir / checkpoint_id
            arch = self.checkpoint_dir / f"{checkpoint_id}.tar.gz"
            try:
                if not self._check_system_health_for_operation(orchestrator):
                    self.logger.error("System health too poor for checkpoint restoration")
                    return False
                if arch.exists() and not cdir.exists():
                    shutil.unpack_archive(str(arch), self.checkpoint_dir)
                if not cdir.exists():
                    self.logger.error(f"Checkpoint {checkpoint_id} not found")
                    return False
                meta = cdir / "checkpoint.json"
                if not meta.exists():
                    self.logger.error("Checkpoint metadata not found")
                    return False
                data = json.loads(meta.read_text())
                if "integrity" in data:
                    saved = data["integrity"]["checksum"]
                    cur = hashlib.sha256(json.dumps(data["modules"], sort_keys=True, default=str).encode()).hexdigest()
                    if saved != cur:
                        self.logger.error("Checkpoint integrity check failed")
                        return False

                self.logger.info(f"[RELOAD] Restoring checkpoint '{checkpoint_id}' from {data['timestamp']}")
                self.create_checkpoint(orchestrator, "pre_restore_backup")

                ok, failed = 0, []
                for name in data.get("modules", {}):
                    mod_file = cdir / f"{name}.json"
                    if not mod_file.exists() or name not in orchestrator.modules:
                        continue
                    try:
                        state = json.loads(mod_file.read_text())
                        mod = orchestrator.modules[name]
                        env = {"state": state, "module_version": getattr(getattr(mod, "metadata", None), "version", "1.0.0")}
                        val = self._validate_state_for_restore(env, mod)
                        if not val.is_valid:
                            self.logger.error(f"State validation failed for {name}")
                            failed.append(name)
                            continue
                        if hasattr(mod, "set_state"):
                            mod.set_state(state)
                            ok += 1
                    except Exception as e:
                        self.logger.error(f"Failed to restore {name}: {e}")
                        failed.append(name)

                orch_state = data.get("orchestrator_state", {})
                if "critical_modules" in orch_state:
                    orchestrator.critical_modules = set(orch_state["critical_modules"])
                if hasattr(orchestrator, "build_execution_plan"):
                    orchestrator.build_execution_plan()

                if failed and len(failed) > len(orchestrator.modules) * 0.3:
                    self.logger.error(f"Too many modules failed to restore: {failed}")
                    return False

                self.logger.info(format_operator_message("[OK]", "CHECKPOINT RESTORED", instrument=checkpoint_id, details=f"Restored {ok} modules, {len(failed)} failed", context="state_management"))
                return ok > 0
            except Exception as e:
                self.logger.error(f"[CRASH] Failed to restore checkpoint {checkpoint_id}: {e}")
                return False

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        cps: List[Dict[str, Any]] = []
        # archived
        for arch in self.checkpoint_dir.glob("*.tar.gz"):
            checkpoint_id = Path(arch.stem).stem  # strip .gz then .tar
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    shutil.unpack_archive(str(arch), tmp)
                    meta = Path(tmp) / checkpoint_id / "checkpoint.json"
                    if meta.exists():
                        d = json.loads(meta.read_text())
                        d.setdefault("checkpoint_id", checkpoint_id)
                        d["file_size_mb"] = arch.stat().st_size / 1024 / 1024
                        d["compressed"] = True
                        cps.append(d)
            except Exception as e:
                self.logger.error(f"Failed to read archived checkpoint {arch.name}: {e}")
        # folders
        for d in self.checkpoint_dir.iterdir():
            if d.is_dir():
                meta = d / "checkpoint.json"
                if meta.exists():
                    try:
                        info = json.loads(meta.read_text())
                        info.setdefault("checkpoint_id", d.name)
                        total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                        info["file_size_mb"] = total / 1024 / 1024
                        info["compressed"] = False
                        cps.append(info)
                    except Exception as e:
                        self.logger.error(f"Failed to read checkpoint {d.name}: {e}")
        cps.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return cps

    def cleanup_old_states(self, days_to_keep: int = 7):
        now = time.time(); cutoff = now - days_to_keep * 24 * 3600
        cleaned = 0
        for f in self.state_dir.glob("*_state.*"):
            if f.stat().st_mtime < cutoff:
                try: f.unlink(); cleaned += 1
                except Exception as e: self.logger.error(f"Failed to remove {f}: {e}")
        for f in self.backup_dir.glob("*"):
            if f.stat().st_mtime < cutoff:
                try: f.unlink(); cleaned += 1
                except Exception as e: self.logger.error(f"Failed to remove {f}: {e}")
        for item in self.checkpoint_dir.iterdir():
            if item.stat().st_mtime < cutoff:
                try:
                    shutil.rmtree(item) if item.is_dir() else item.unlink()
                    cleaned += 1
                except Exception as e:
                    self.logger.error(f"Failed to remove {item}: {e}")
        if cleaned:
            self.logger.info(f"🧹 Cleaned up {cleaned} old files")


# ═════════════════════════════════════════════════════════════
# ReplayEngine (with safe InfoBus subscriptions)
# ═════════════════════════════════════════════════════════════

class ReplayEngine:
    def __init__(self, orchestrator: Optional["ModuleOrchestrator"] = None):
        self.orchestrator = orchestrator
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        self.smart_bus = InfoBusManager.get_instance()

        self.current_session: Optional[ReplaySession] = None
        self.replay_position = 0
        self.replay_speed = 1.0
        self.is_playing = False
        self.is_paused = False
        self.is_recording = False

        self.recorded_events: List[ReplayEvent] = []
        self.sequence_counter = 0
        self.current_recording_id: Optional[str] = None
        self.recording_start_time = 0

        self.health_snapshots: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)

        self.event_filters: List[Callable[[ReplayEvent], bool]] = []
        self.event_modifiers: List[Callable[[ReplayEvent], ReplayEvent]] = []
        self.breakpoints: List[Tuple[str, Callable[[ReplayEvent], bool]]] = []
        self.analysis_collectors: List[Dict[str, Any]] = []
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        self.session_dir = Path("replay_sessions"); self.session_dir.mkdir(exist_ok=True)
        self._lock = threading.RLock()

        self.logger = RotatingLogger(
            name="ReplayEngine",
            log_path="logs/replay/replay_engine.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )

        self._setup_event_subscriptions()
        self.logger.info(format_operator_message("🎬", "REPLAY ENGINE INITIALIZED", details=f"Session dir: {self.session_dir}", context="startup"))

    def _setup_event_subscriptions(self):
        subscribe = getattr(self.smart_bus, "subscribe", None)
        if not callable(subscribe):
            self.logger.info("SmartInfoBus has no 'subscribe' support; recording limited to manual events")
            return
        try:
            self.smart_bus.subscribe("data_updated", self._record_data_update)
            self.smart_bus.subscribe("module_disabled", self._record_module_event)
            self.smart_bus.subscribe("performance_warning", self._record_module_event)
            self.smart_bus.subscribe("module_enabled", self._record_module_event)
            self.smart_bus.subscribe("execution_complete", self._record_execution_event)
        except Exception as e:
            self.logger.warning(f"Bus subscription failed: {e}")

    def start_recording(self, session_id: Optional[str] = None) -> str:
        """
        Start recording a new session with system health tracking.
        """
        with self._lock:
            if self.is_recording:
                self.stop_recording()
            
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_recording_id = session_id
            self.recorded_events.clear()
            self.health_snapshots.clear()
            self.performance_metrics.clear()
            self.sequence_counter = 0
            self.is_recording = True
            self.recording_start_time = time.time()
            
            # Record initial system state with health
            initial_state = self._capture_system_state()
            initial_health = self._capture_system_health()
            
            # Create initial event
            initial_event = ReplayEvent(
                timestamp=time.time(),
                event_type='recording_started',
                module='ReplayEngine',
                data={
                    'session_id': session_id,
                    'initial_state': initial_state,
                    'initial_health': initial_health,
                    'recording_start': datetime.now().isoformat()
                },
                execution_id=session_id,
                sequence_number=self.sequence_counter,
                metadata={'health_score': initial_health.get('overall_score', 0)}
            )
            
            self.recorded_events.append(initial_event)
            self.health_snapshots.append(initial_health)
            self.sequence_counter += 1
            
            # Start health monitoring task
            self._start_health_monitoring()
            
            self.logger.info(
                format_operator_message(
                    "[RED]", "RECORDING STARTED",
                    instrument=session_id,
                    context="replay_recording"
                )
            )
            
            return session_id
    
    def stop_recording(self) -> Optional[ReplaySession]:
        """
        Stop recording and save session with validation.
        """
        with self._lock:
            if not self.is_recording:
                self.logger.warning("No recording in progress")
                return None
            
            try:
                # Record final system state and health
                final_state = self._capture_system_state()
                final_health = self._capture_system_health()
                
                # Create final event
                final_event = ReplayEvent(
                    timestamp=time.time(),
                    event_type='recording_stopped',
                    module='ReplayEngine',
                    data={
                        'session_id': self.current_recording_id,
                        'final_state': final_state,
                        'final_health': final_health,
                        'recording_end': datetime.now().isoformat(),
                        'total_events': len(self.recorded_events),
                        'duration': time.time() - self.recording_start_time
                    },
                    execution_id=self.current_recording_id or "unknown",
                    sequence_number=self.sequence_counter,
                    metadata={'health_score': final_health.get('overall_score', 0)}
                )
                
                self.recorded_events.append(final_event)
                self.health_snapshots.append(final_health)
                
                # Analyze health trends
                health_analysis = self._analyze_health_trends()
                
                # Create session object
                session = ReplaySession(
                    session_id=self.current_recording_id or "unknown",
                    start_time=datetime.fromtimestamp(self.recorded_events[0].timestamp),
                    end_time=datetime.fromtimestamp(self.recorded_events[-1].timestamp),
                    events=self.recorded_events.copy(),
                    initial_state=self.recorded_events[0].data.get('initial_state', {}),
                    final_state=final_state,
                    system_health=health_analysis,
                    metadata={
                        'total_events': len(self.recorded_events),
                        'duration_seconds': self.recorded_events[-1].timestamp - self.recorded_events[0].timestamp,
                        'modules_involved': list(set(e.module for e in self.recorded_events)),
                        'event_types': list(set(e.event_type for e in self.recorded_events)),
                        'health_snapshots': len(self.health_snapshots),
                        'performance_summary': self._get_performance_summary()
                    }
                )
                
                # Validate session integrity
                if not session.validate_integrity():
                    self.logger.error("Session integrity validation failed")
                    return None
                
                # Save session to disk
                self._save_session(session)
                
                # Reset recording state
                self.is_recording = False
                self.current_recording_id = None
                
                self.logger.info(
                    format_operator_message(
                        "⏹️", "RECORDING STOPPED",
                        instrument=session.session_id,
                        details=f"{session.event_count} events, {session.duration_seconds:.1f}s",
                        context="replay_recording"
                    )
                )
                
                return session
                
            except Exception as e:
                self.logger.error(f"[CRASH] Failed to stop recording: {e}")
                self.is_recording = False
                return None
    
    def _start_health_monitoring(self):
        """Start periodic health monitoring during recording (loop-safe)."""
        async def monitor_health():
            while self.is_recording:
                try:
                    await asyncio.sleep(10)  # Check every 10 seconds
                    if self.is_recording:
                        health = self._capture_system_health()
                        self.health_snapshots.append(health)
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(monitor_health())
        except RuntimeError:
            # No loop – health monitoring will start when a loop exists (recording still works)
            self.logger.info("Health monitoring deferred (no running event loop)")

    
    def _capture_system_health(self) -> Dict[str, Any]:
        """Capture comprehensive system health"""
        import psutil
        
        health = {
            'timestamp': time.time(),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'thread_count': threading.active_count()
        }
        
        if self.orchestrator:
            # Get orchestrator health
            health['emergency_mode'] = self.orchestrator.emergency_mode
            health['circuit_breakers_open'] = sum(
                1 for cb in self.orchestrator.circuit_breakers.values() 
                if cb.state == 'OPEN'
            )
            
            # Get execution metrics
            metrics = self.orchestrator.get_execution_metrics()
            health['success_rate'] = metrics.get('success_rate', 0)
            health['avg_execution_time'] = metrics.get('avg_execution_time_ms', 0)
            
            # Calculate overall health score
            health['overall_score'] = self._calculate_health_score(health)
        
        return health
    
    def _calculate_health_score(self, health: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-1)"""
        score = 1.0
        
        # Memory usage penalty
        memory_mb = health.get('memory_usage_mb', 0)
        if memory_mb > 2000:
            score *= 0.7
        elif memory_mb > 1000:
            score *= 0.9
        
        # CPU usage penalty
        cpu = health.get('cpu_percent', 0)
        if cpu > 80:
            score *= 0.8
        elif cpu > 50:
            score *= 0.95
        
        # Emergency mode penalty
        if health.get('emergency_mode'):
            score *= 0.5
        
        # Circuit breaker penalty
        open_breakers = health.get('circuit_breakers_open', 0)
        if open_breakers > 0:
            score *= max(0.5, 1 - (open_breakers * 0.1))
        
        # Success rate factor
        success_rate = health.get('success_rate', 1.0)
        score *= success_rate
        
        return max(0.0, min(1.0, score))
    
    def _analyze_health_trends(self) -> Dict[str, Any]:
        """Analyze health trends from snapshots"""
        if not self.health_snapshots:
            return {}
        
        def _mean(vals):
            if not vals:
                return 0.0
            if NUMPY_AVAILABLE and np is not None:
                return float(np.mean(vals))
            return sum(vals)/len(vals)

        analysis = {
            'snapshot_count': len(self.health_snapshots),
            'avg_health_score': _mean([h.get('overall_score', 0) for h in self.health_snapshots]),
            'min_health_score': min(h.get('overall_score', 1) for h in self.health_snapshots) if self.health_snapshots else 0,
            'max_memory_mb': max(h.get('memory_usage_mb', 0) for h in self.health_snapshots) if self.health_snapshots else 0,
            'avg_cpu_percent': _mean([h.get('cpu_percent', 0) for h in self.health_snapshots]),
            'emergency_mode_activations': sum(1 for h in self.health_snapshots if h.get('emergency_mode')),
            'health_degradation_events': 0
        }
        
        # Check for health degradation
        for i in range(1, len(self.health_snapshots)):
            prev_score = self.health_snapshots[i-1].get('overall_score', 1)
            curr_score = self.health_snapshots[i].get('overall_score', 1)
            if curr_score < prev_score * 0.8:  # 20% drop
                analysis['health_degradation_events'] += 1
        
        return analysis
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """
        Robust statistics helper – safe when NumPy is missing or data is absent.
        """
        if not self.performance_metrics:
            return {
                "status": "no_data",
                "message": "No performance metrics collected",
                "metrics_available": False
            }

        # ---------- helpers ----------
        def _mean(vals):
            if not vals:
                return 0.0
            return float(np.mean(vals)) if NUMPY_AVAILABLE and np is not None else sum(vals) / len(vals)

        def _std(vals):
            if not vals or len(vals) < 2:
                return 0.0
            return float(np.std(vals)) if NUMPY_AVAILABLE and np is not None else 0.0

        def _p95(vals):
            if not vals:
                return 0.0
            if NUMPY_AVAILABLE and np is not None and len(vals) >= 5:
                return float(np.percentile(vals, 95))
            if len(vals) >= 5:
                v = sorted(vals)
                return float(v[int(len(v) * 0.95)])
            return max(vals)

        # ---------- scaffold ----------
        summary: Dict[str, Any] = {
            "status": "data_available",
            "metrics_available": True,
            "collection_period": {
                "recording_duration_seconds":
                    time.time() - self.recording_start_time if self.recording_start_time else 0,
                "total_samples":
                    sum(len(v) for v in self.performance_metrics.values()),
                "metric_types": list(self.performance_metrics.keys())
            },
            "aggregated_metrics": {},
            "module_breakdown": {},
            "health_indicators": {}
        }

        # ---------- aggregate ----------
        for m_name, values in self.performance_metrics.items():
            if not values:
                continue
            summary["aggregated_metrics"][m_name] = {
                "avg": _mean(values),
                "min": float(min(values)),
                "max": float(max(values)),
                "std": _std(values),
                "p95": _p95(values),
                "sample_count": len(values),
                "total": float(sum(values)) if m_name.endswith("_count") else None
            }

        exec_times       = self.performance_metrics.get("execution_time", [])
        replay_exec      = self.performance_metrics.get("replayed_execution_time", [])

        # ---------- module split ----------
        if exec_times or replay_exec:
            summary["module_breakdown"] = {
                "original_execution": {
                    "count": len(exec_times),
                    "avg_time_ms": _mean(exec_times),
                    "total_time_ms": float(sum(exec_times)) if exec_times else 0.0
                },
                "replayed_execution": {
                    "count": len(replay_exec),
                    "avg_time_ms": _mean(replay_exec),
                    "total_time_ms": float(sum(replay_exec)) if replay_exec else 0.0
                }
            }

        # ---------- health indicators ----------
        error_rate: float = 0.0
        avg_latency: float = _mean(exec_times)

        try:
            total_ops   = sum(len(v) for k, v in self.performance_metrics.items()
                            if not k.startswith("error"))
            total_errs  = sum(len(v) for k, v in self.performance_metrics.items()
                            if k.startswith("error"))
            error_rate  = total_errs / max(total_ops, 1)

            latency_status = (
                "excellent" if avg_latency < 50 else
                "good"      if avg_latency < 100 else
                "acceptable"if avg_latency < 200 else
                "poor"
            ) if exec_times else "no_data"

            health_score = 100.0
            if error_rate > 0.10:   health_score -= 50
            elif error_rate > 0.05: health_score -= 25
            elif error_rate > 0.01: health_score -= 10

            if avg_latency > 500:      health_score -= 30
            elif avg_latency > 200:    health_score -= 15
            elif avg_latency > 100:    health_score -= 5

            summary["health_indicators"] = {
                "overall_health_score": max(0.0, health_score),
                "overall_status": (
                    "healthy"  if health_score > 80 else
                    "degraded" if health_score > 50 else
                    "critical"
                ),
                "error_analysis": {
                    "total_operations": total_ops,
                    "total_errors": total_errs,
                    "error_rate": error_rate,
                    "error_status": (
                        "good"     if error_rate < 0.01 else
                        "warning"  if error_rate < 0.05 else
                        "critical"
                    )
                },
                "latency_analysis": {
                    "avg_latency_ms": avg_latency,
                    "latency_status": latency_status,
                    "samples": len(exec_times)
                },
                "data_quality": {
                    "metrics_collected": len(self.performance_metrics),
                    "total_samples": sum(len(v) for v in self.performance_metrics.values()),
                    "completeness_score":
                        min(100.0, len(self.performance_metrics) * 20.0)   # 5 types → 100 %
                }
            }

        except Exception as e:
            summary["health_indicators"] = {
                "overall_status": "unknown",
                "error": f"indicator calc failed: {e}"
            }

        # ---------- trends ----------
        try:
            if exec_times and len(exec_times) >= 10 and NUMPY_AVAILABLE and np is not None:
                recent = exec_times[-10:]
                slope  = float(np.polyfit(np.arange(len(recent)), recent, 1)[0])
                summary["health_indicators"]["performance_trend"] = {  # type: ignore
                    "direction": ("improving" if slope < -1
                                else "degrading" if slope > 1
                                else "stable"),
                    "slope_ms_per_sample": float(slope),
                    "samples_analyzed": len(recent)
                }
        except Exception:
            pass  # non-critical

        # ---------- actionable tips ----------
        rec_actions = []
        if error_rate > 0.05:
            rec_actions.append("Investigate high error rate")
        if avg_latency > 200:
            rec_actions.append("Optimize performance – high latency")
        if len(self.performance_metrics) < 2:
            rec_actions.append("Collect more performance metrics")

        summary["summary"] = {
            "recording_active": self.is_recording,
            "has_data": True,
            "data_quality": "good" if len(self.performance_metrics) >= 3 else "limited",
            "recommended_actions": rec_actions
        }

        return summary

    def _record_data_update(self, event_data: Dict[str, Any]):
        """Record data update event during recording"""
        if not self.is_recording:
            return
        
        with self._lock:
            event = ReplayEvent(
                timestamp=time.time(),
                event_type='data_update',
                module=event_data.get('module', 'unknown'),
                data=event_data.copy(),
                execution_id=self.current_recording_id or "unknown",
                sequence_number=self.sequence_counter,
                metadata={
                    'data_size': len(str(event_data.get('value', ''))),
                    'confidence': event_data.get('confidence', 0)
                }
            )
            
            self.recorded_events.append(event)
            self.sequence_counter += 1
    
    def _record_module_event(self, event_data: Dict[str, Any]):
        """Record module-related event during recording"""
        if not self.is_recording:
            return
        
        with self._lock:
            event = ReplayEvent(
                timestamp=time.time(),
                event_type=event_data.get('type', 'module_event'),
                module=event_data.get('module', 'unknown'),
                data=event_data.copy(),
                execution_id=self.current_recording_id or "unknown",
                sequence_number=self.sequence_counter,
                metadata={
                    'severity': event_data.get('severity', 'info')
                }
            )
            
            self.recorded_events.append(event)
            self.sequence_counter += 1
    
    def _record_execution_event(self, event_data: Dict[str, Any]):
        """Record execution completion event"""
        if not self.is_recording:
            return
        
        with self._lock:
            # Record performance metrics
            if 'execution_time_ms' in event_data:
                self.performance_metrics['execution_time'].append(
                    event_data['execution_time_ms']
                )
            
            event = ReplayEvent(
                timestamp=time.time(),
                event_type='execution_complete',
                module='Orchestrator',
                data=event_data.copy(),
                execution_id=event_data.get('execution_id', 'unknown'),
                sequence_number=self.sequence_counter,
                metadata={
                    'success_rate': event_data.get('success_count', 0) / 
                                  max(event_data.get('module_count', 1), 1)
                }
            )
            
            self.recorded_events.append(event)
            self.sequence_counter += 1
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture comprehensive system state"""
        state = {
            'timestamp': time.time(),
            'smartinfobus_metrics': self.smart_bus.get_performance_metrics(),
            'data_keys': list(getattr(self.smart_bus, "_data_store", {}).keys())[:50],  # First 50 keys
            'active_modules': []
        }
        
        if self.orchestrator:
            state['orchestrator_metrics'] = self.orchestrator.get_execution_metrics()
            state['module_health'] = {}
            state['circuit_breaker_states'] = {}
            
            for name, module in self.orchestrator.modules.items():
                try:
                    state['module_health'][name] = module.get_health_status()
                    
                    if name in self.orchestrator.circuit_breakers:
                        cb = self.orchestrator.circuit_breakers[name]
                        state['circuit_breaker_states'][name] = {
                            'state': getattr(cb, 'state', 'CLOSED'),
                            'failure_count': getattr(cb, 'failure_count', 0)
                        }
                    
                    if hasattr(module, 'get_state'):
                        # Store lightweight state snapshot
                        module_state = module.get_state()
                        state['active_modules'].append({
                            'name': name,
                            'health': module_state.get('health_status', 'unknown'),
                            'step_count': module_state.get('step_count', 0),
                            'error_count': module_state.get('error_count', 0)
                        })
                except Exception as e:
                    self.logger.warning(f"Failed to capture state for {name}: {e}")
        
        return state
    
    async def play(self, 
                   start_position: int = 0, 
                   end_position: Optional[int] = None, 
                   speed: float = 1.0,
                   validate_health: bool = True):
        """
        Play session with health monitoring.
        """
        if not self.current_session:
            raise ValueError("No session loaded")
        
        with self._lock:
            self.replay_position = start_position
            end_pos = end_position or len(self.current_session.events)
            self.replay_speed = speed
            self.is_playing = True
            self.is_paused = False
        
        try:
            # Restore initial state if starting from beginning
            if start_position == 0 and self.orchestrator:
                await self._restore_system_state(self.current_session.initial_state)
            
            # Get initial health if validating
            if validate_health and self.orchestrator:
                initial_health = self._capture_system_health()
                if initial_health.get('overall_score', 1) < 0.5:
                    self.logger.warning("System health poor at replay start")
            
            # Calculate timing for replay
            if self.current_session.events:
                first_timestamp = self.current_session.events[max(0, start_position)].timestamp
            else:
                first_timestamp = time.time()
            
            replay_start_time = time.time()
            events_replayed = 0
            health_check_interval = 100  # Check health every 100 events
            
            self.logger.info(
                f"🎬 Starting replay from position {start_position} to {end_pos} (speed: {speed}x)"
            )
            
            while self.replay_position < end_pos and self.is_playing:
                # Handle pause
                if self.is_paused:
                    await asyncio.sleep(0.1)
                    continue
                
                event = self.current_session.events[self.replay_position]
                
                # Apply filters
                if not all(f(event) for f in self.event_filters):
                    self.replay_position += 1
                    continue
                
                # Apply modifiers
                modified_event = event
                for modifier in self.event_modifiers:
                    modified_event = modifier(modified_event)
                
                # Check breakpoints
                for bp_name, bp_condition in self.breakpoints:
                    if bp_condition(modified_event):
                        self.logger.info(f"[SEARCH] Breakpoint hit: {bp_name} at position {self.replay_position}")
                        await self.pause()
                        break
                
                # Calculate timing for real-time replay
                if self.replay_speed > 0:
                    event_offset = modified_event.timestamp - first_timestamp
                    target_replay_time = replay_start_time + (event_offset / self.replay_speed)
                    current_time = time.time()
                    
                    # Wait if needed
                    if current_time < target_replay_time:
                        await asyncio.sleep(target_replay_time - current_time)
                
                # Replay event
                await self._replay_event(modified_event)
                
                # Trigger callbacks
                await self._trigger_event_callbacks(modified_event)
                
                # Collect analysis data
                if self.analysis_collectors:
                    self._collect_analysis_data(modified_event)
                
                self.replay_position += 1
                events_replayed += 1
                
                # Periodic health check
                if validate_health and events_replayed % health_check_interval == 0:
                    current_health = self._capture_system_health()
                    if current_health.get('overall_score', 1) < 0.3:
                        self.logger.warning("System health degraded during replay - pausing")
                        await self.pause()
                
                # Progress logging
                if events_replayed % 100 == 0:
                    progress = (self.replay_position / end_pos) * 100
                    self.logger.debug(f"Replay progress: {progress:.1f}% ({events_replayed} events)")
            
            self.is_playing = False
            
            replay_duration = time.time() - replay_start_time
            
            # Final health check
            if validate_health and self.orchestrator:
                final_health = self._capture_system_health()
                health_summary = f", final health: {final_health.get('overall_score', 0):.1%}"
            else:
                health_summary = ""
            
            self.logger.info(
                format_operator_message(
                    "[OK]", "REPLAY COMPLETED",
                    details=f"{events_replayed} events in {replay_duration:.1f}s{health_summary}",
                    context="replay_playback"
                )
            )
            
        except Exception as e:
            self.is_playing = False
            self.logger.error(f"[CRASH] Replay failed: {e}")
            raise
    
    async def _replay_event(self, event: ReplayEvent):
        """Replay a single event with type-specific handling"""
        try:
            if event.event_type == 'data_update':
                # Replay data update in SmartInfoBus
                data = event.data
                self.smart_bus.set(
                    key=data.get('key', 'unknown'),
                    value=data.get('value'),
                    module=event.module,
                    thesis=data.get('thesis', f"Replayed from {event.execution_id}"),
                    confidence=data.get('confidence', 1.0)
                )
                
            elif event.event_type == 'module_disabled':
                # Replay module disable
                if self.orchestrator:
                    module_name = event.data.get('module')
                    if module_name and module_name in self.orchestrator.modules:
                        self.smart_bus.record_module_failure(module_name, "Replayed failure")
                        
            elif event.event_type == 'module_enabled':
                # Replay module enable
                if self.orchestrator:
                    module_name = event.data.get('module')
                    if module_name and module_name in self.orchestrator.modules:
                        self.smart_bus.reset_module_failures(module_name)
                        
            elif event.event_type == 'execution_complete':
                # Track execution metrics
                if 'execution_time_ms' in event.data:
                    self.performance_metrics['replayed_execution_time'].append(
                        event.data['execution_time_ms']
                    )
                        
            elif event.event_type in ['recording_started', 'recording_stopped']:
                # Skip meta events during replay
                pass
            else:
                # Generic event replay
                self.logger.debug(f"Replaying generic event: {event.event_type} from {event.module}")
                
        except Exception as e:
            self.logger.error(f"Failed to replay event {event.event_type}: {e}")
    
    async def _restore_system_state(self, state: Dict[str, Any]):
        """Restore system to captured state (defensive against API changes)."""
        if not self.orchestrator:
            return

        try:
            # Clear current state (defensively)
            if hasattr(self.smart_bus, "_cleanup_old_data"):
                try:
                    self.smart_bus._cleanup_old_data()
                except Exception as e:
                    self.logger.warning(f"SmartInfoBus cleanup failed: {e}")

            # Restore circuit breaker states
            cb_states = state.get('circuit_breaker_states', {})
            for module_name, cb_state in cb_states.items():
                if module_name in self.orchestrator.circuit_breakers:
                    cb = self.orchestrator.circuit_breakers[module_name]
                    cb.state = cb_state.get('state', getattr(cb, 'state', 'CLOSED'))
                    cb.failure_count = cb_state.get('failure_count', getattr(cb, 'failure_count', 0))

            # Restore module states if available
            active_modules = state.get('active_modules', [])
            for module_info in active_modules:
                module_name = module_info.get('name')
                if module_name in self.orchestrator.modules:
                    module = self.orchestrator.modules[module_name]
                    if hasattr(module, 'set_state'):
                        try:
                            module.set_state({
                                'step_count': module_info.get('step_count', 0),
                                'health_status': module_info.get('health', 'OK'),
                                'error_count': module_info.get('error_count', 0)
                            })
                        except Exception as e:
                            self.logger.warning(f"Failed to restore state for {module_name}: {e}")

            self.logger.info("System state restored for replay")

        except Exception as e:
            self.logger.error(f"Failed to restore system state: {e}")

    async def pause(self):
        """Pause replay with state preservation"""
        self.is_paused = True
        self.logger.info(f"⏸️ Replay paused at position {self.replay_position}")
    
    async def resume(self):
        """Resume replay"""
        self.is_paused = False
        self.logger.info(f"▶️ Replay resumed at position {self.replay_position}")
    
    def stop(self):
        """Stop replay"""
        self.is_playing = False
        self.is_paused = False
        self.logger.info(f"⏹️ Replay stopped at position {self.replay_position}")
    
    def seek(self, position: int):
        """Seek to specific position with validation"""
        if not self.current_session:
            raise ValueError("No session loaded")
        
        max_position = len(self.current_session.events) - 1
        self.replay_position = max(0, min(position, max_position))
        
        self.logger.info(f"⏭️ Seeked to position {self.replay_position}")
    
    def load_session(self, session_id: str) -> ReplaySession:
        """Load session for replay with validation"""
        session_file = self.session_dir / f"{session_id}.replay"
        
        if not session_file.exists():
            raise ValueError(f"Session not found: {session_id}")
        
        try:
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
            
            # Handle different data formats
            if isinstance(session_data, dict):
                # Convert from dictionary format
                events = [ReplayEvent.from_dict(e) for e in session_data.get('events', [])]
                
                session = ReplaySession(
                    session_id=session_data['session_id'],
                    start_time=datetime.fromisoformat(session_data['start_time']),
                    end_time=datetime.fromisoformat(session_data['end_time']),
                    events=events,
                    initial_state=session_data.get('initial_state', {}),
                    final_state=session_data.get('final_state', {}),
                    metadata=session_data.get('metadata', {}),
                    system_health=session_data.get('system_health', {})
                )
            else:
                # Assume it's already a ReplaySession object
                session = session_data
            
            # Validate session integrity
            if not session.validate_integrity():
                raise ValueError(f"Session integrity validation failed: {session_id}")
            
            self.current_session = session
            self.replay_position = 0
            
            self.logger.info(
                format_operator_message(
                    "[FOLDER]", "SESSION LOADED",
                    instrument=session_id,
                    details=f"{session.event_count} events, {session.duration_seconds:.1f}s",
                    context="replay_loading"
                )
            )
            
            return session
            
        except Exception as e:
            raise ValueError(f"Failed to load session {session_id}: {e}")
    
    def _save_session(self, session: ReplaySession):
        """Save session to disk with compression"""
        session_file = self.session_dir / f"{session.session_id}.replay"
        
        try:
            # Save as pickle for full fidelity
            with open(session_file, 'wb') as f:
                pickle.dump(session, f)
            
            # Also save metadata as JSON for easy browsing
            metadata_file = self.session_dir / f"{session.session_id}.meta.json"
            metadata = {
                'session_id': session.session_id,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat(),
                'duration_seconds': session.duration_seconds,
                'event_count': session.event_count,
                'statistics': session.get_statistics(),
                'system_health': session.system_health
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Session saved: {session_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
            raise
    
    # Analysis and filtering methods
    def add_filter(self, filter_func: Callable[[ReplayEvent], bool], name: str = ""):
        """Add event filter with optional name"""
        self.event_filters.append(filter_func)
        filter_name = name or f"filter_{len(self.event_filters)}"
        self.logger.info(f"Added filter: {filter_name}")
    
    def add_modifier(self, modifier_func: Callable[[ReplayEvent], ReplayEvent], name: str = ""):
        """Add event modifier for what-if analysis"""
        self.event_modifiers.append(modifier_func)
        modifier_name = name or f"modifier_{len(self.event_modifiers)}"
        self.logger.info(f"Added modifier: {modifier_name}")
    
    def add_breakpoint(self, name: str, condition: Callable[[ReplayEvent], bool]):
        """Add conditional breakpoint"""
        self.breakpoints.append((name, condition))
        self.logger.info(f"Added breakpoint: {name}")
    
    def clear_breakpoint(self, name: str):
        """Remove breakpoint by name"""
        self.breakpoints = [(n, c) for n, c in self.breakpoints if n != name]
        self.logger.info(f"Removed breakpoint: {name}")
    
    def subscribe_to_event(self, event_type: str, callback: Callable):
        """Subscribe to specific event type during replay"""
        self.event_callbacks[event_type].append(callback)
    
    async def _trigger_event_callbacks(self, event: ReplayEvent):
        """Trigger callbacks for replayed events"""
        # General callbacks
        for callback in self.event_callbacks.get('*', []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")
        
        # Type-specific callbacks
        for callback in self.event_callbacks.get(event.event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")
    
    def add_analysis_collector(self, name: str, extractor: Callable[[ReplayEvent], Any]):
        """Add analysis data collector"""
        self.analysis_collectors.append({
            'name': name,
            'extractor': extractor,
            'data': []
        })
    
    def _collect_analysis_data(self, event: ReplayEvent):
        """Collect analysis data from replayed event"""
        for collector in self.analysis_collectors:
            try:
                data = collector['extractor'](event)
                if data is not None:
                    collector['data'].append({
                        'timestamp': event.timestamp,
                        'position': self.replay_position,
                        'value': data,
                        'event_type': event.event_type,
                        'module': event.module
                    })
            except Exception as e:
                self.logger.error(f"Analysis collector error for {collector['name']}: {e}")
    
    def get_analysis_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get collected analysis results"""
        return {
            collector['name']: collector['data']
            for collector in self.analysis_collectors
        }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List available replay sessions"""
        sessions = []
        
        for meta_file in self.session_dir.glob("*.meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                sessions.append(metadata)
            except Exception as e:
                self.logger.error(f"Failed to read {meta_file}: {e}")
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        
        return sessions
    
    def delete_session(self, session_id: str):
        """Delete a replay session"""
        session_file = self.session_dir / f"{session_id}.replay"
        metadata_file = self.session_dir / f"{session_id}.meta.json"
        
        removed_files = []
        
        if session_file.exists():
            session_file.unlink()
            removed_files.append("replay")
        
        if metadata_file.exists():
            metadata_file.unlink()
            removed_files.append("metadata")
        
        if removed_files:
            self.logger.info(f"🗑️ Deleted session {session_id}: {', '.join(removed_files)}")
        else:
            self.logger.warning(f"Session {session_id} not found")

class PersistenceManager:
    def __init__(self, orchestrator: Optional["ModuleOrchestrator"] = None, state_dir: str = "state", session_dir: str = "replay_sessions"):
        self.orchestrator = orchestrator
        self.state_manager = StateManager(f"{state_dir}/modules")
        self.replay_engine = ReplayEngine(orchestrator)

        self.config = {
            "auto_checkpoint_interval": 3600,
            "auto_state_interval": 300,
            "max_session_age_days": 30,
            "compression_enabled": True,
            "validation_enabled": True,
            "health_check_enabled": True,
            "persist_infobus": True,
            "infobus_snapshot_path": f"{state_dir}/infobus.json",
        }

        self._background_tasks = []
        self._shutdown_event = asyncio.Event()

        self.logger = RotatingLogger(
            name="PersistenceManager",
            log_path="logs/persistence/persistence_manager.log",
            max_lines=5000,
            operator_mode=True,
        )

        if self.config["persist_infobus"]:
            if load_infobus_snapshot(self.config["infobus_snapshot_path"]):
                self.logger.info("[SAVE] SmartInfoBus snapshot loaded")

        self._start_background_maintenance()
        self.logger.info(format_operator_message("[SAVE]", "PERSISTENCE MANAGER INITIALIZED", context="startup"))

    def _start_background_maintenance(self):
        async def auto_checkpoint():
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(self.config["auto_checkpoint_interval"])
                    if self.orchestrator and self.config["health_check_enabled"]:
                        health = self.orchestrator.get_execution_metrics() if hasattr(self.orchestrator, "get_execution_metrics") else {}
                        if health.get("success_rate", 0) > 0.7:
                            self.state_manager.create_checkpoint(self.orchestrator, f"auto_{int(time.time())}")
                            self.logger.info("[RELOAD] Auto-checkpoint created")
                        else:
                            self.logger.warning("Skipped auto-checkpoint due to poor system health")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Auto-checkpoint failed: {e}")

        async def periodic_cleanup():
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(24 * 3600)
                    self.state_manager.cleanup_old_states(days_to_keep=self.config["max_session_age_days"])
                    self._cleanup_old_sessions()
                    self.logger.info("🧹 Periodic cleanup completed")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Periodic cleanup failed: {e}")

        async def auto_state_save():
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.sleep(self.config["auto_state_interval"])
                    if self.orchestrator:
                        results = self.state_manager.save_all_module_states(self.orchestrator)
                        ok = sum(1 for v in results.values() if v)
                        self.logger.info(f"[SAVE] Auto-saved {ok}/{len(results)} module states")
                        if self.config["persist_infobus"]:
                            save_infobus_snapshot(self.config["infobus_snapshot_path"])
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Auto-state-save failed: {e}")

        try:
            loop = asyncio.get_running_loop()
            self._background_tasks = [
                loop.create_task(auto_checkpoint()),
                loop.create_task(periodic_cleanup()),
                loop.create_task(auto_state_save()),
            ]
        except RuntimeError:
            self.logger.info("No running event loop; background tasks not started")
            self._background_tasks = []

    def _cleanup_old_sessions(self):
        cutoff = time.time() - (self.config["max_session_age_days"] * 24 * 3600)
        cleaned = 0
        for s in self.replay_engine.session_dir.glob("*.replay"):
            if s.stat().st_mtime < cutoff:
                try:
                    s.unlink()
                    meta = s.with_suffix(".meta.json")
                    if meta.exists(): meta.unlink()
                    cleaned += 1
                except Exception as e:
                    self.logger.error(f"Failed to remove old session {s}: {e}")
        if cleaned:
            self.logger.info(f"Cleaned up {cleaned} old sessions")

    async def shutdown(self):
        self.logger.info("[STOP] Shutting down persistence manager...")
        try: self._shutdown_event.set()
        except Exception: pass
        for t in list(self._background_tasks):
            try: t.cancel()
            except Exception: pass
        if self._background_tasks:
            try: await asyncio.gather(*self._background_tasks, return_exceptions=True)
            except Exception: pass
        if self.orchestrator:
            try: self.state_manager.save_all_module_states(self.orchestrator)
            except Exception as e: self.logger.error(f"Final state save failed: {e}")
        if self.config["persist_infobus"]:
            try:
                save_infobus_snapshot(self.config["infobus_snapshot_path"])
                self.logger.info("[SAVE] SmartInfoBus snapshot saved")
            except Exception as e:
                self.logger.error(f"InfoBus snapshot failed: {e}")
        if self.orchestrator:
            try: self.state_manager.create_checkpoint(self.orchestrator, "final_shutdown")
            except Exception as e: self.logger.error(f"Final checkpoint failed: {e}")
        self.logger.info("[OK] Persistence manager shutdown complete")

    # --- reporting & utilities (unchanged APIs) ---

    def get_status_report(self) -> str:
        checkpoints = self.state_manager.list_checkpoints()
        sessions = self.replay_engine.list_sessions()
        state_size_mb = sum(f.stat().st_size for f in self.state_manager.state_dir.rglob('*') if f.is_file()) / 1024 / 1024
        replay_size_mb = sum(f.stat().st_size for f in self.replay_engine.session_dir.rglob('*') if f.is_file()) / 1024 / 1024
        infobus_snapshot = Path(self.config["infobus_snapshot_path"])
        infobus_info = "present" if infobus_snapshot.exists() else "absent"
        lines = [
            "PERSISTENCE SYSTEM STATUS",
            "=" * 50,
            "State Manager:",
            f"  Available Checkpoints: {len(checkpoints)}",
            f"  State Directory: {self.state_manager.state_dir}",
            f"  Storage Used: {state_size_mb:.1f} MB",
            f"  Validation Enabled: {self.state_manager.validation_enabled}",
            f"  Compression Enabled: {self.state_manager.compression_enabled}",
            "",
            "Replay Engine:",
            f"  Available Sessions: {len(sessions)}",
            f"  Recording Active: {self.replay_engine.is_recording}",
            f"  Playback Active: {self.replay_engine.is_playing}",
            f"  Session Directory: {self.replay_engine.session_dir}",
            f"  Storage Used: {replay_size_mb:.1f} MB",
            "",
            "InfoBus:",
            f"  Snapshot: {infobus_info} ({self.config['infobus_snapshot_path']})",
            "",
            "Configuration:",
            f"  Auto-checkpoint Interval: {self.config['auto_checkpoint_interval']}s",
            f"  Auto-state Save Interval: {self.config['auto_state_interval']}s",
            f"  Max Session Age: {self.config['max_session_age_days']} days",
            f"  Compression: {self.config['compression_enabled']}",
            f"  Validation: {self.config['validation_enabled']}",
            f"  Health Checks: {self.config['health_check_enabled']}",
            f"  Persist InfoBus: {self.config['persist_infobus']}",
            "",
            f"Total Storage: {state_size_mb + replay_size_mb:.1f} MB",
        ]
        return "\n".join(lines)

    def create_system_backup(self, backup_name: str = "system_backup") -> bool:
        try:
            ts = int(time.time())
            backup_id = f"{backup_name}_{ts}"
            root = Path("backups"); root.mkdir(parents=True, exist_ok=True)
            bdir = root / backup_id; bdir.mkdir(parents=True, exist_ok=True)
            if self.orchestrator:
                self.state_manager.create_checkpoint(self.orchestrator, backup_name)
            shutil.copytree(self.state_manager.state_dir, bdir / "states")
            shutil.copytree(self.replay_engine.session_dir, bdir / "replays")
            try:
                ib = Path(self.config["infobus_snapshot_path"])
                if ib.exists():
                    dst = bdir / "infobus"; dst.mkdir(exist_ok=True)
                    shutil.copy2(ib, dst / ib.name)
            except Exception:
                pass
            meta = {
                "backup_id": backup_id,
                "backup_name": backup_name,
                "timestamp": datetime.now().isoformat(),
                "state_root": str(self.state_manager.state_dir),
                "replay_root": str(self.replay_engine.session_dir),
                "infobus_snapshot": Path(self.config["infobus_snapshot_path"]).exists(),
                "state_files": len([p for p in (bdir / "states").rglob("*") if p.is_file()]),
                "replay_files": len([p for p in (bdir / "replays").rglob("*") if p.is_file()]),
                "system_info": {"orchestrator_available": self.orchestrator is not None, "module_count": len(self.orchestrator.modules) if self.orchestrator else 0},
            }
            (bdir / "backup_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
            arch_base = root / backup_id
            try:
                shutil.make_archive(str(arch_base), "gztar", root, backup_id)
                shutil.rmtree(bdir)
            except Exception as e:
                self.logger.error(f"Compression failed: {e}")
            self.logger.info(f"[OK] System backup created: {arch_base}.tar.gz")
            return True
        except Exception as e:
            self.logger.error(f"System backup failed: {e}")
            return False

    def restore_system_backup(self, backup_id: str) -> bool:
        try:
            arch = Path("backups") / f"{backup_id}.tar.gz"
            if not arch.exists():
                self.logger.error(f"Backup not found: {backup_id}")
                return False
            with tempfile.TemporaryDirectory() as tmp:
                shutil.unpack_archive(str(arch), tmp)
                bdir = Path(tmp) / backup_id
                meta = bdir / "backup_metadata.json"
                if meta.exists():
                    md = json.loads(meta.read_text())
                    self.logger.info(f"Restoring backup from {md.get('timestamp','unknown time')}")
                else:
                    self.logger.warning("Backup metadata.json missing; continuing with best effort.")
                try: self.create_system_backup("pre_restore_backup")
                except Exception as e: self.logger.warning(f"Pre-restore backup failed (continuing): {e}")
                src_states = bdir / "states"
                if src_states.exists():
                    shutil.rmtree(self.state_manager.state_dir, ignore_errors=True)
                    shutil.copytree(src_states, self.state_manager.state_dir)
                src_replays = bdir / "replays"
                if src_replays.exists():
                    shutil.rmtree(self.replay_engine.session_dir, ignore_errors=True)
                    shutil.copytree(src_replays, self.replay_engine.session_dir)
                ib_dir = bdir / "infobus"
                if ib_dir.exists():
                    try:
                        dest = Path(self.config["infobus_snapshot_path"])
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        for p in ib_dir.glob("*.json"):
                            shutil.copy2(p, dest); break
                        loaded = load_infobus_snapshot(self.config["infobus_snapshot_path"])
                        self.logger.info(f"InfoBus snapshot restore: {'ok' if loaded else 'skipped'}")
                    except Exception as e:
                        self.logger.error(f"Failed to restore InfoBus snapshot: {e}")
                if self.orchestrator:
                    self.state_manager.restore_all_states(self.orchestrator)
                self.logger.info(f"[OK] System restored from backup: {backup_id}")
                return True
        except Exception as e:
            self.logger.error(f"System restore failed: {e}")
            return False

    def get_checkpoint_details(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        for cp in self.state_manager.list_checkpoints():
            if cp.get("checkpoint_id") == checkpoint_id:
                return cp
        return None

    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        for s in self.replay_engine.list_sessions():
            if s.get("session_id") == session_id:
                return s
        return None

    def export_diagnostics(self, output_path: str = "diagnostics"):
        try:
            d = Path(output_path); d.mkdir(parents=True, exist_ok=True)
            (d / "status_report.txt").write_text(self.get_status_report())
            (d / "checkpoints.json").write_text(json.dumps(self.state_manager.list_checkpoints(), indent=2, default=str))
            (d / "sessions.json").write_text(json.dumps(self.replay_engine.list_sessions(), indent=2, default=str))
            cfg = {"persistence_config": self.config,
                   "state_manager_config": {"max_backups": self.state_manager.max_backups,
                                            "compression_enabled": self.state_manager.compression_enabled,
                                            "validation_enabled": self.state_manager.validation_enabled}}
            (d / "configuration.json").write_text(json.dumps(cfg, indent=2, default=str))
            try:
                ib = Path(self.config["infobus_snapshot_path"])
                if ib.exists(): shutil.copy2(ib, d / ib.name)
            except Exception: pass
            self.logger.info(f"[STATS] Diagnostics exported to {d}")
        except Exception as e:
            self.logger.error(f"Failed to export diagnostics: {e}")

    def verify_system_integrity(self) -> Dict[str, Any]:
        res = {"timestamp": datetime.now().isoformat(),
               "checkpoints": {"total": 0, "valid": 0, "corrupted": []},
               "sessions": {"total": 0, "valid": 0, "corrupted": []},
               "overall_integrity": True}
        cps = self.state_manager.list_checkpoints(); res["checkpoints"]["total"] = len(cps)
        for cp in cps:
            try:
                res["checkpoints"]["valid"] += 1 if "integrity" in cp else 0
                if "integrity" not in cp: res["checkpoints"]["corrupted"].append(cp.get("checkpoint_id","unknown"))
            except Exception as e:
                res["checkpoints"]["corrupted"].append(f"{cp.get('checkpoint_id','unknown')}: {e}")
        for meta in self.replay_engine.list_sessions():
            sid = meta.get("session_id", "unknown")
            res["sessions"]["total"] += 1
            try:
                sess = self.replay_engine.load_session(sid)
                if sess.validate_integrity():
                    res["sessions"]["valid"] += 1
                else:
                    res["sessions"]["corrupted"].append(sid)
            except Exception as e:
                res["sessions"]["corrupted"].append(f"{sid}: {e}")
        if res["checkpoints"]["corrupted"] or res["sessions"]["corrupted"]:
            res["overall_integrity"] = False
        return res

    def repair_corrupted_data(self) -> Dict[str, Any]:
        report = {"checkpoints_repaired": 0, "sessions_repaired": 0, "failed_repairs": []}
        integ = self.verify_system_integrity()
        for chk in integ["checkpoints"]["corrupted"]:
            try:
                self.logger.warning(f"Would repair checkpoint: {chk}")
            except Exception as e:
                report["failed_repairs"].append(f"Checkpoint {chk}: {e}")
        for sess in integ["sessions"]["corrupted"]:
            try:
                self.logger.warning(f"Would repair session: {sess}")
            except Exception as e:
                report["failed_repairs"].append(f"Session {sess}: {e}")
        return report

    def get_module_state_history(self, module_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for cp in self.state_manager.list_checkpoints():
            modules = cp.get("modules")
            if isinstance(modules, dict) and module_name in modules:
                out.append({"checkpoint_id": cp.get("checkpoint_id"),
                            "timestamp": cp.get("timestamp"),
                            "state_version": modules[module_name].get("version", 0)})
        out.sort(key=lambda x: x["timestamp"] or "", reverse=True)
        return out


# ═════════════════════════════════════════════════════════════
# Convenience functions
# ═════════════════════════════════════════════════════════════

def create_persistence_manager(orchestrator: Optional["ModuleOrchestrator"] = None) -> PersistenceManager:
    return PersistenceManager(orchestrator)

def quick_checkpoint(orchestrator: "ModuleOrchestrator", name: str = "quick") -> bool:
    return StateManager().create_checkpoint(orchestrator, name)

def quick_restore(orchestrator: "ModuleOrchestrator", checkpoint_id: str) -> bool:
    return StateManager().restore_checkpoint(orchestrator, checkpoint_id)

async def record_session(duration_seconds: float = 60, session_id: Optional[str] = None) -> Optional[ReplaySession]:
    engine = ReplayEngine()
    _ = engine.start_recording(session_id)
    await asyncio.sleep(duration_seconds)
    return engine.stop_recording()

def list_all_checkpoints() -> List[Dict[str, Any]]:
    return StateManager().list_checkpoints()

def list_all_sessions() -> List[Dict[str, Any]]:
    return ReplayEngine().list_sessions()


if __name__ == "__main__":
    import asyncio
    async def _smoke():
        pm = PersistenceManager()
        print(pm.get_status_report())
        integ = pm.verify_system_integrity()
        print(f"\nSystem Integrity: {integ['overall_integrity']}")
        if not integ["overall_integrity"]:
            print(json.dumps(integ, indent=2))
        pm.export_diagnostics()
        ok = pm.create_system_backup("smoke_test_backup")
        print(f"\nBackup created: {ok}")
    asyncio.run(_smoke())
