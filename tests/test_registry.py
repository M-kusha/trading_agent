"""Module-registry integrity tests.

config/module_registry.yaml drives the live ModuleOrchestrator through dynamic
string imports, so a stale entry is invisible to static analysis. The
orchestrator catches ImportError and only logs it, which meant live trading
could start with modules silently missing.

These tests make registry drift a build failure instead.
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "config" / "module_registry.yaml"


def _registry() -> dict:
    if not REGISTRY_PATH.is_file():
        pytest.skip("module_registry.yaml not found")
    # utf-8 explicitly: the file contains box-drawing characters that break
    # under the Windows default cp1252 codec.
    with io.open(REGISTRY_PATH, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    return loaded.get("modules", loaded)


def _entries() -> list[tuple[str, dict]]:
    return [
        (name, spec)
        for name, spec in _registry().items()
        if isinstance(spec, dict) and "module_path" in spec
    ]


def test_registry_parses_and_is_non_empty():
    entries = _entries()
    assert len(entries) > 0, "registry declares no modules"


def test_every_registry_entry_has_a_source_file():
    """The failure that let NewsSentimentModule survive its own deletion."""
    missing = []
    for name, spec in _entries():
        module_path = spec["module_path"]
        source = REPO_ROOT / (module_path.replace(".", "/") + ".py")
        package = REPO_ROOT / module_path.replace(".", "/") / "__init__.py"
        if not source.is_file() and not package.is_file():
            missing.append(f"{name} -> {module_path}")
    assert not missing, "registry entries with no source file: " + ", ".join(missing)


def test_every_registry_entry_is_importable_by_spec():
    """Every entry must resolve, or fail only because an optional third-party
    package is absent.

    find_spec() imports parent packages, so a registry module whose package
    __init__ pulls in torch raises ModuleNotFoundError on a machine without
    torch. That is an environment gap, not registry drift. Only failures that
    name one of OUR packages indicate a genuinely broken entry.
    """
    FIRST_PARTY = ("modules", "envs", "train", "live", "config", "utils", "backend", "dashboard")

    broken: list[str] = []
    skipped: list[str] = []
    for name, spec in _entries():
        module_path = spec["module_path"]
        try:
            if importlib.util.find_spec(module_path) is None:
                broken.append(f"{name} -> {module_path} (no spec)")
        except ModuleNotFoundError as exc:
            missing = (exc.name or "").split(".")[0]
            if missing in FIRST_PARTY:
                broken.append(f"{name} -> {module_path} (missing {exc.name})")
            else:
                skipped.append(f"{name}: needs {missing}")
        except (ImportError, ValueError) as exc:
            broken.append(f"{name} -> {module_path} ({type(exc).__name__})")

    assert not broken, "broken registry entries: " + ", ".join(broken)
    if skipped:
        pytest.skip(
            f"{len(skipped)} entries need uninstalled third-party packages: "
            + ", ".join(sorted(skipped)[:5])
        )


def test_declared_class_name_matches_entry_key():
    """The orchestrator looks modules up by class name; a mismatch means the
    entry loads nothing."""
    mismatched = []
    for name, spec in _entries():
        source = REPO_ROOT / (spec["module_path"].replace(".", "/") + ".py")
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        if f"class {name}" not in text:
            mismatched.append(f"{name} not defined in {spec['module_path']}")
    assert not mismatched, "registry key / class mismatches: " + ", ".join(mismatched)


def test_no_duplicate_module_paths():
    seen: dict[str, str] = {}
    dupes = []
    for name, spec in _entries():
        path = spec["module_path"]
        if path in seen:
            dupes.append(f"{path} claimed by both {seen[path]} and {name}")
        seen[path] = name
    assert not dupes, "; ".join(dupes)
