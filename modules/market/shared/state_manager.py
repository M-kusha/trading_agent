

from __future__ import annotations

import copy
import datetime as _dt
import gzip
import hashlib
import json
import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class StateManager:

    def __init__(
        self,
        logger: Optional[Any] = None,
        hot_reload_enabled: bool = True,
        checkpoint_dir: Optional[str | Path] = None,
        checkpoint_format: str = "json",
        compression: bool = True,
        max_history: int = 20,
        interprocess_lock: bool = True,
    ):
        self.logger = logger
        self.hot_reload_enabled = hot_reload_enabled


        self._lock = threading.RLock()
        self._interprocess_lock_enabled = bool(interprocess_lock)


        self._component_states: Dict[str, Dict[str, Any]] = {}
        self._global_state: Dict[str, Any] = {}
        self._state_history: List[Dict[str, Any]] = []
        self._max_history = int(max_history)


        self._checkpoints: Dict[str, Dict[str, Any]] = {}
        base_dir = Path(checkpoint_dir) if checkpoint_dir else Path("state/market/checkpoints")
        self._checkpoint_dir = base_dir
        self._checkpoint_format = checkpoint_format.lower()
        self._compression = bool(compression)


        self._schema_version = 1
        self._state_version = 0

        self.trace("StateManager initialized", level="DEBUG")


    def trace(self, message: str, level: str = "INFO"):
        if self.logger:
            self.logger.trace(f"[StateManager] {message}", level=level)


    def get_component_state(self, component: str) -> Dict[str, Any]:
        with self._lock:
            self.trace(f"Getting state for component: {component}", level="TRACE")
            return copy.deepcopy(self._component_states.get(component, {}))

    def set_component_state(self, component: str, state: Dict[str, Any]):
        with self._lock:
            self.trace(f"Setting state for component: {component}", level="TRACE")
            prev = self._component_states.get(component)
            if prev is not None:
                self._add_to_history(component, prev)
            self._component_states[component] = copy.deepcopy(state)
            self._bump_version()

    def update_component_state(self, component: str, updates: Dict[str, Any]):
        with self._lock:
            self.trace(f"Updating state for component: {component}", level="TRACE")
            current = self._component_states.get(component, {})
            if current:
                self._add_to_history(component, current)
            updated = copy.deepcopy(current)
            updated.update(copy.deepcopy(updates))
            self._component_states[component] = updated
            self._bump_version()

    def get_global_state(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._global_state)

    def set_global_state(self, state: Dict[str, Any]):
        with self._lock:
            self.trace("Setting global state", level="TRACE")
            if self._global_state:
                self._add_to_history("_global", self._global_state)
            self._global_state = copy.deepcopy(state)
            self._bump_version()

    def rollback(self, component: str, steps: int = 1) -> bool:
        with self._lock:
            self.trace(f"Rolling back {component} by {steps} steps", level="INFO")
            comp_hist = [h for h in reversed(self._state_history) if h['component'] == component]
            if len(comp_hist) < steps:
                self.trace("Insufficient history for rollback", level="WARNING")
                return False
            target_state = comp_hist[steps - 1]['state']
            if component == "_global":
                self._global_state = copy.deepcopy(target_state)
            else:
                self._component_states[component] = copy.deepcopy(target_state)
            self._bump_version()
            self.trace(f"Rollback successful for {component}", level="INFO")
            return True

    def create_checkpoint(self, name: str):
        with self._lock:
            self.trace(f"Creating checkpoint: {name}", level="INFO")
            checkpoint = self._build_checkpoint()
            self._checkpoints[name] = checkpoint
            if self.hot_reload_enabled:
                self._save_checkpoint_to_disk(name, checkpoint)

    def restore_checkpoint(self, name: str) -> bool:
        with self._lock:
            self.trace(f"Restoring checkpoint: {name}", level="INFO")
            if name not in self._checkpoints:
                checkpoint = self._load_checkpoint_from_disk(name)
                if not checkpoint:
                    self.trace(f"Checkpoint not found: {name}", level="WARNING")
                    return False
            else:
                checkpoint = self._checkpoints[name]

            ok = self._apply_checkpoint(checkpoint)
            if ok:
                self.trace(f"Checkpoint restored: {name}", level="INFO")
            return ok

    def export_state(self, filepath: str):
        with self._lock:
            self.trace(f"Exporting state to: {filepath}", level="INFO")
            state = {
                'schema_version': self._schema_version,
                'state_version': self._state_version,
                'component_states': self._component_states,
                'global_state': self._global_state,
                'export_time': _dt.datetime.now().isoformat(),
            }
            self._atomic_json_write(Path(filepath), state)

    def import_state(self, filepath: str) -> bool:
        with self._lock:
            self.trace(f"Importing state from: {filepath}", level="INFO")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                self._component_states = copy.deepcopy(state.get('component_states', {}))
                self._global_state = copy.deepcopy(state.get('global_state', {}))
                self._bump_version()
                self.trace("State imported successfully", level="INFO")
                return True
            except Exception as e:
                self.trace(f"Failed to import state: {e}", level="ERROR")
                return False

    def clear_state(self, component: Optional[str] = None):
        with self._lock:
            if component:
                self.trace(f"Clearing state for component: {component}", level="INFO")
                self._component_states.pop(component, None)
            else:
                self.trace("Clearing all state", level="INFO")
                self._component_states.clear()
                self._global_state.clear()
            self._bump_version()

    def get_state_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                'components': list(self._component_states.keys()),
                'component_count': len(self._component_states),
                'global_keys': list(self._global_state.keys()),
                'history_size': len(self._state_history),
                'checkpoints': list(self._checkpoints.keys()),
                'schema_version': self._schema_version,
                'state_version': self._state_version,
                'checkpoint_dir': str(self._checkpoint_dir),
            }


    def list_checkpoints_on_disk(self) -> List[str]:
        suffix = self._checkpoint_suffix()
        if self._compression:
            suffix += ".gz"
        return sorted([p.stem.replace(suffix.replace('.', ''), '')
                       for p in self._checkpoint_dir.glob(f"*{suffix}")])

    def prune_checkpoints(self, keep_last: int = 5) -> int:
        files = self._sorted_checkpoint_files()
        to_delete = files[:-keep_last] if len(files) > keep_last else []
        deleted = 0
        for f in to_delete:
            try:
                f.unlink(missing_ok=True)
                deleted += 1
            except Exception as e:
                self.trace(f"Failed to delete checkpoint {f.name}: {e}", level="WARNING")
        if deleted:
            self.trace(f"Pruned {deleted} checkpoints", level="INFO")
        return deleted

    def transaction(self):
        return _StateTransaction(self)


    def _bump_version(self):
        self._state_version += 1

    def _add_to_history(self, component: str, state: Dict[str, Any]):

        self._state_history.append({
            'component': component,
            'state': copy.deepcopy(state),
            'timestamp': _dt.datetime.now().isoformat()
        })

        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)

    def _build_checkpoint(self) -> Dict[str, Any]:
        return {
            'schema_version': self._schema_version,
            'state_version': self._state_version,
            'component_states': copy.deepcopy(self._component_states),
            'global_state': copy.deepcopy(self._global_state),
            'timestamp': _dt.datetime.now().isoformat()
        }

    def _apply_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        try:

            assert 'component_states' in checkpoint and 'global_state' in checkpoint
            self._component_states = copy.deepcopy(checkpoint['component_states'])
            self._global_state = copy.deepcopy(checkpoint['global_state'])
            self._bump_version()
            return True
        except Exception as e:
            self.trace(f"Invalid checkpoint structure: {e}", level="ERROR")
            return False


    def _checkpoint_suffix(self) -> str:
        if self._checkpoint_format == "pickle":
            return ".pkl"
        return ".json"

    def _checkpoint_path(self, name: str) -> Path:
        suf = self._checkpoint_suffix()
        if self._compression:
            suf += ".gz"
        return self._checkpoint_dir / f"{name}{suf}"

    def _save_checkpoint_to_disk(self, name: str, checkpoint: Dict[str, Any]):

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self._checkpoint_path(name)
        try:
            if self._checkpoint_format == "pickle":

                payload = pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL)
                self._atomic_bytes_write(path, payload, compressed=self._compression)
            else:

                payload = self._with_checksum(checkpoint)
                self._atomic_json_write(path, payload, compressed=self._compression)
            self.trace(f"Checkpoint saved to disk: {path}", level="TRACE")
        except Exception as e:
            self.trace(f"Failed to save checkpoint: {e}", level="ERROR")

    def _load_checkpoint_from_disk(self, name: str) -> Optional[Dict[str, Any]]:
        path = self._checkpoint_path(name)
        if not path.exists():
            return None
        try:

            with self._file_lock(path.with_suffix(path.suffix + ".lock"), timeout=5.0):
                if self._checkpoint_format == "pickle":
                    data = self._read_bytes(path, compressed=self._compression)
                    checkpoint = pickle.loads(data)
                else:
                    data = self._read_json(path, compressed=self._compression)
                    checkpoint = self._validate_checksum(data)
            self.trace(f"Checkpoint loaded from disk: {path}", level="TRACE")
            return checkpoint
        except Exception as e:
            self.trace(f"Failed to load checkpoint: {e}", level="ERROR")
            return None


    def _atomic_json_write(self, path: Path, obj: Dict[str, Any], compressed: bool = False):
        text = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
        data = text.encode("utf-8")
        self._atomic_bytes_write(path, data, compressed=compressed)

    def _read_json(self, path: Path, compressed: bool = False) -> Dict[str, Any]:
        if compressed:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _atomic_bytes_write(self, path: Path, data: bytes, compressed: bool = False):

        lockfile = path.with_suffix(path.suffix + ".lock")
        with self._file_lock(lockfile, timeout=10.0):

            tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
            try:
                with os.fdopen(tmp_fd, "wb") as tmp:
                    if compressed:
                        with gzip.GzipFile(fileobj=tmp, mode="wb") as gz:
                            gz.write(data)
                            gz.flush()
                            os.fsync(tmp.fileno())
                    else:
                        tmp.write(data)
                        tmp.flush()
                        os.fsync(tmp.fileno())
                os.replace(tmp_path, path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

    def _read_bytes(self, path: Path, compressed: bool = False) -> bytes:
        if compressed:
            with gzip.open(path, "rb") as f:
                return f.read()
        with open(path, "rb") as f:
            return f.read()


    @staticmethod
    def _sha256_of_json(obj: Dict[str, Any]) -> str:

        canon = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(canon).hexdigest()

    def _with_checksum(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            'meta': {
                'schema_version': checkpoint.get('schema_version', self._schema_version),
                'state_version': checkpoint.get('state_version', self._state_version),
                'timestamp': checkpoint.get('timestamp', _dt.datetime.now().isoformat()),
                'checksum': '',
            },
            'data': {
                'component_states': checkpoint['component_states'],
                'global_state': checkpoint['global_state'],
            }
        }
        payload['meta']['checksum'] = self._sha256_of_json(payload['data'])
        return payload

    def _validate_checksum(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        meta = payload.get('meta', {})
        data = payload.get('data', {})
        checksum = meta.get('checksum', '')
        if not checksum:
            raise ValueError("Missing checkpoint checksum")
        calc = self._sha256_of_json(data)
        if calc != checksum:
            raise ValueError("Checkpoint checksum mismatch")
        return {
            'schema_version': meta.get('schema_version', self._schema_version),
            'state_version': meta.get('state_version', 0),
            'component_states': data.get('component_states', {}),
            'global_state': data.get('global_state', {}),
            'timestamp': meta.get('timestamp'),
        }


    class _SoftLock:
        def __init__(self, path: Path):
            self.path = path
            self.acquired = False

        def acquire(self) -> bool:
            try:

                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w") as f:
                    f.write(f"pid={os.getpid()} time={time.time()}\n")
                self.acquired = True
                return True
            except FileExistsError:
                return False

        def release(self):
            if self.acquired:
                try:
                    os.remove(self.path)
                except FileNotFoundError:
                    pass
                self.acquired = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.release()

    def _file_lock(self, lockfile: Path, timeout: float = 10.0):
        class _Locker:
            def __init__(self, outer: StateManager, lf: Path, timeout_s: float):
                self.outer = outer
                self.lf = lf
                self.timeout_s = timeout_s
                self.soft = StateManager._SoftLock(lf)

            def __enter__(self):
                if not self.outer._interprocess_lock_enabled:
                    return self
                end = time.time() + self.timeout_s
                while time.time() < end:
                    if self.soft.acquire():
                        return self
                    time.sleep(0.05)

                self.outer.trace(f"Lock timeout on {self.lf.name}; continuing without interprocess lock", level="WARNING")
                return self

            def __exit__(self, exc_type, exc, tb):
                self.soft.release()

        return _Locker(self, lockfile, timeout)


    def _sorted_checkpoint_files(self) -> List[Path]:
        suf = self._checkpoint_suffix()
        if self._compression:
            suf += ".gz"
        files = list(self._checkpoint_dir.glob(f"*{suf}"))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files


class _StateTransaction:

    def __init__(self, mgr: StateManager):
        self._m = mgr
        self._snapshot: Optional[Dict[str, Any]] = None

    def __enter__(self):
        with self._m._lock:
            self._m.trace("Transaction begin", level="TRACE")

            self._snapshot = {
                'component_states': copy.deepcopy(self._m._component_states),
                'global_state': copy.deepcopy(self._m._global_state),
                'version': self._m._state_version,
            }
        return self

    def __exit__(self, exc_type, exc, tb):
        with self._m._lock:
            if exc_type is not None and self._snapshot is not None:
                self._m.trace(f"Transaction rollback due to error: {exc}", level="WARNING")
                self._m._component_states = self._snapshot['component_states']
                self._m._global_state = self._snapshot['global_state']
                self._m._state_version = self._snapshot['version']
            else:
                self._m.trace("Transaction commit", level="TRACE")
