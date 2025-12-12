from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from config import get_logger, load_app_config, setup_logging
from modules.contracts import CONTRACTS
from modules.core.module_base import BaseModule, ModuleMetadata
from modules.utils.info_bus import InfoBusManager, SmartInfoBus


def _module_path_from_contract_file(contract_file: str) -> str:
    rel = contract_file.replace("\\", "/").lstrip("/")
    if rel.endswith(".py"):
        rel = rel[:-3]
    return "modules." + rel.replace("/", ".")


def load_module_class(contract_name: str) -> Type[BaseModule]:
    contract = CONTRACTS.get(contract_name)
    if not contract:
        raise KeyError(f"Unknown module contract: {contract_name}")
    module_path = _module_path_from_contract_file(contract.file)
    mod = importlib.import_module(module_path)

    matches: List[Type[BaseModule]] = []
    for obj in vars(mod).values():
        if not isinstance(obj, type):
            continue
        if not issubclass(obj, BaseModule) or obj is BaseModule:
            continue
        meta = getattr(obj, "__module_metadata__", None)
        if isinstance(meta, ModuleMetadata) and meta.name == contract_name:
            matches.append(obj)

    if len(matches) == 1:
        return matches[0]
    if hasattr(mod, contract_name):
        obj = getattr(mod, contract_name)
        if isinstance(obj, type) and issubclass(obj, BaseModule):
            return obj
    if matches:
        raise RuntimeError(
            f"Multiple module classes match contract '{contract_name}': {[c.__name__ for c in matches]}"
        )
    raise RuntimeError(f"No BaseModule class found for contract '{contract_name}' in {module_path}")


def _price_df(n: int = 64) -> Any:
    try:
        import numpy as np
        import pandas as pd

        base = np.linspace(100, 101, n)
        return pd.DataFrame(
            {
                "open": base,
                "high": base + 0.5,
                "low": base - 0.5,
                "close": base + 0.1,
                "volume": np.ones(n),
            }
        )
    except Exception:
        base = [100.0 + i * (1.0 / max(n - 1, 1)) for i in range(n)]
        return {
            "open": base,
            "high": [x + 0.5 for x in base],
            "low": [x - 0.5 for x in base],
            "close": [x + 0.1 for x in base],
            "volume": [1.0 for _ in base],
        }


def default_market_data(
    instruments: Sequence[str] = ("EURUSD", "XAUUSD"),
    timeframes: Sequence[str] = ("M15", "H1", "H4", "D1"),
    n: int = 128,
) -> Dict[str, Any]:
    frames = {tf: _price_df(n) for tf in timeframes}
    data: Dict[str, Any] = {}
    for inst in instruments:
        per_tf = {tf: frames[tf].copy() if hasattr(frames[tf], "copy") else frames[tf] for tf in timeframes}
        data[inst] = per_tf
        c = "".join(ch for ch in str(inst) if ch.isalnum()).upper()
        if len(c) == 6:
            data.setdefault(f"{c[:3]}/{c[3:]}", per_tf)
            data.setdefault(f"{c[:3]}_{c[3:]}", per_tf)
    return data


def default_inputs_for_requires(requires: Iterable[str]) -> Dict[str, Any]:
    data = default_market_data()
    out: Dict[str, Any] = {"execution_id": "module_lab_exec_0"}
    prices = {inst: 100.0 for inst in data.keys()}

    for k in requires:
        lk = str(k).lower()
        if k in out:
            continue
        if k in {"market_data", "historical_prices", "multi_timeframe_data"}:
            out[k] = data
        elif k in {"prices", "price_data", "tick_prices"}:
            out[k] = prices
        elif k in {"market_context"}:
            out[k] = {"timestamp": time.time(), "instruments": list(data.keys())}
        elif k in {"positions", "trades", "order_data", "orders", "recent_trades"}:
            out[k] = []
        elif k in {"step_idx", "episode_idx"} or lk.endswith("_idx"):
            out[k] = 0
        elif "config" in lk:
            out[k] = {}
        else:
            out[k] = {}
    return out


class _EventRecorder:
    def __init__(self) -> None:
        self.data_updates: List[Dict[str, Any]] = []
        self.contract_violations: List[Dict[str, Any]] = []

    def on_data_updated(self, event: Dict[str, Any]) -> None:
        self.data_updates.append(dict(event))

    def on_event_logged(self, event: Dict[str, Any]) -> None:
        if event.get("type") == "contract_violation":
            self.contract_violations.append(dict(event))


def _configure_bus(bus: SmartInfoBus, *, contract_mode: str) -> None:
    try:
        bus.config.persistence_enabled = False
    except Exception:
        pass
    try:
        bus.config.contract_enforcement = contract_mode
    except Exception:
        pass


def _instantiate_module(
    module_class: Type[BaseModule],
    *,
    module_config: Optional[Any],
    bus: SmartInfoBus,
) -> BaseModule:
    dependencies = {"bus": bus}
    try:
        return module_class(config=module_config, dependencies=dependencies)
    except TypeError:
        try:
            instance = module_class(config=module_config)
        except TypeError:
            instance = module_class()
        if getattr(instance, "bus", None) is None:
            try:
                instance.bus = bus
            except Exception:
                pass
        return instance


def _publish_outputs(
    bus: SmartInfoBus,
    *,
    module_name: str,
    outputs: Dict[str, Any],
    requires: Sequence[str],
    processing_time_ms: float,
) -> None:
    thesis = outputs.get("_thesis") if isinstance(outputs.get("_thesis"), str) else None
    confidence_raw = outputs.get("_confidence")
    confidence = 1.0
    if isinstance(confidence_raw, (int, float)) and 0.0 <= float(confidence_raw) <= 1.0:
        confidence = float(confidence_raw)

    for key, value in outputs.items():
        if str(key).startswith("_"):
            continue
        bus.set(
            str(key),
            value,
            module=module_name,
            thesis=thesis or f"module_lab:{module_name}",
            confidence=confidence,
            dependencies=list(requires),
            processing_time_ms=processing_time_ms,
        )


async def run_module_once(
    contract_name: str,
    *,
    timeout_s: float = 5.0,
    contract_mode: str = "warn",
) -> Dict[str, Any]:
    os.environ.setdefault("SMARTINFOBUS_AUTOWIRE", "0")

    app_config = load_app_config(mode="training")
    app_config.logging.debug = False
    app_config.logging.level = "INFO"
    setup_logging(app_config.logging)

    logger = get_logger("module_lab")

    InfoBusManager.reset_instance()
    bus = InfoBusManager.get_instance()
    _configure_bus(bus, contract_mode=contract_mode)

    rec = _EventRecorder()
    try:
        bus.subscribe("data_updated", rec.on_data_updated)
        bus.subscribe("event_logged", rec.on_event_logged)
    except Exception:
        pass

    module_class = load_module_class(contract_name)
    meta: ModuleMetadata = getattr(module_class, "__module_metadata__")
    requires = list(meta.requires or [])
    inputs = default_inputs_for_requires(requires)

    module_config: Optional[Any] = None
    try:
        from modules.core.configuration_manager import ConfigurationManager

        cm = ConfigurationManager.get_instance()
        module_config = cm.get_module_config(contract_name)
    except Exception:
        module_config = None

    module = _instantiate_module(module_class, module_config=module_config, bus=bus)

    t0 = time.perf_counter()
    outputs: Dict[str, Any] = {}
    error: Optional[str] = None

    try:
        outputs = await asyncio.wait_for(module.process(**inputs), timeout=timeout_s)
        if not isinstance(outputs, dict):
            outputs = {"_raw": outputs}
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error(f"[module_lab] {contract_name} failed: {error}")
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0

    if error is None:
        try:
            _publish_outputs(
                bus,
                module_name=contract_name,
                outputs=outputs,
                requires=requires,
                processing_time_ms=float(dt_ms),
            )
        except Exception as e:
            error = f"PublishError({type(e).__name__}): {e}"

    try:
        snapshot = bus.export_snapshot(include_values=False, include_history=False, include_events=False, include_metrics=True)
    except Exception:
        snapshot = {}

    provided_keys = [k for k in outputs.keys() if not str(k).startswith("_")]
    self_updates = [e for e in rec.data_updates if str(e.get("module")) == contract_name]
    other_updates = [e for e in rec.data_updates if str(e.get("module")) != contract_name]
    self_violations = [e for e in rec.contract_violations if str(e.get("module")) == contract_name]
    other_violations = [e for e in rec.contract_violations if str(e.get("module")) != contract_name]
    return {
        "module": contract_name,
        "class": module_class.__name__,
        "provides_declared": list(meta.provides or []),
        "requires_declared": requires,
        "inputs_keys": sorted(list(inputs.keys())),
        "outputs_keys": sorted(list(provided_keys)),
        "status": "ok" if error is None else "error",
        "error": error,
        "time_ms": float(dt_ms),
        "bus_updates_self": self_updates,
        "bus_updates_other_count": len(other_updates),
        "bus_updates_other_sample": other_updates[:50],
        "contract_violations_self": self_violations,
        "contract_violations_other_count": len(other_violations),
        "contract_violations_other_sample": other_violations[:50],
        "bus_metrics": snapshot.get("metrics", {}),
    }


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _print_list(items: Sequence[str]) -> None:
    for m in sorted(items):
        print(m)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="module_lab", description="Run SmartInfoBus modules in isolation.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List contract modules")

    show = sub.add_parser("show", help="Show contract metadata for a module")
    show.add_argument("module", help="Contract module name")

    run = sub.add_parser("run", help="Run a single module once and emit a JSON report")
    run.add_argument("module", help="Contract module name")
    run.add_argument("--timeout-s", type=float, default=5.0)
    run.add_argument("--contract-mode", choices=("off", "warn", "strict"), default="warn")
    run.add_argument("--out", type=Path, default=Path("logs/module_lab/module_report.json"))

    run_all = sub.add_parser("run-all", help="Run multiple modules and emit a JSON report")
    run_all.add_argument("--limit", type=int, default=0, help="If >0, run only the first N modules")
    run_all.add_argument("--timeout-s", type=float, default=3.0)
    run_all.add_argument("--contract-mode", choices=("off", "warn", "strict"), default="warn")
    run_all.add_argument("--out", type=Path, default=Path("logs/module_lab/modules_report.json"))

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.cmd == "list":
        _print_list(list(CONTRACTS.keys()))
        return 0

    if args.cmd == "show":
        name = str(args.module)
        c = CONTRACTS.get(name)
        if not c:
            raise SystemExit(f"Unknown module: {name}")
        payload = {"name": c.name, "file": c.file, "provides": c.provides, "requires": c.requires, "meta": c.meta}
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if args.cmd == "run":
        payload = asyncio.run(
            run_module_once(args.module, timeout_s=float(args.timeout_s), contract_mode=str(args.contract_mode))
        )
        write_report(Path(args.out), payload)
        print(f"Wrote {args.out}")
        return 0 if payload.get("status") == "ok" else 1

    if args.cmd == "run-all":
        modules = sorted(CONTRACTS.keys())
        if int(args.limit) > 0:
            modules = modules[: int(args.limit)]

        results: List[Dict[str, Any]] = []
        for name in modules:
            results.append(
                asyncio.run(
                    run_module_once(name, timeout_s=float(args.timeout_s), contract_mode=str(args.contract_mode))
                )
            )

        payload = {
            "generated_at": time.time(),
            "count": len(results),
            "ok": sum(1 for r in results if r.get("status") == "ok"),
            "error": sum(1 for r in results if r.get("status") != "ok"),
            "results": results,
        }
        write_report(Path(args.out), payload)
        print(f"Wrote {args.out}")
        return 0

    raise SystemExit(f"Unhandled command: {args.cmd}")
