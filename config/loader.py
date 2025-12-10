from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .models import (
    EnvironmentConfig,
    LoggingConfig,
    ModeConfig,
    MT5Config,
    PathsConfig,
    RLConfig,
    RiskConfig,
    TradingAgentConfig,
)


CONFIG_DIR = Path(__file__).resolve().parent
BASE_CONFIG_PATH = CONFIG_DIR / "base.yaml"
TRAINING_CONFIG_PATH = CONFIG_DIR / "training.yaml"
LIVE_CONFIG_PATH = CONFIG_DIR / "live.yaml"
PRESETS_CONFIG_PATH = CONFIG_DIR / "presets.yaml"
RISK_POLICY_PATH = CONFIG_DIR / "risk_policy.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_risk_policy(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load risk_policy.yaml into a dict."""
    return _load_yaml(path or RISK_POLICY_PATH)


def _load_base_bundle(mode: str) -> Dict[str, Any]:
    base = _load_yaml(BASE_CONFIG_PATH)
    mode_path = TRAINING_CONFIG_PATH if mode == "training" else LIVE_CONFIG_PATH
    mode_cfg = _load_yaml(mode_path)
    merged = _deep_merge(base, mode_cfg)
    return merged


def _load_preset(name: Optional[str]) -> Dict[str, Any]:
    if not name:
        return {}
    presets = _load_yaml(PRESETS_CONFIG_PATH).get("presets", {})
    return presets.get(name, {})


def load_app_config(
    mode: str = "training",
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> TradingAgentConfig:
    """
    Load and merge the application config.

    Args:
        mode: "training" or "live"
        preset: optional preset name from presets.yaml
        overrides: optional dictionary of explicit overrides (last-write-wins)
    """
    merged = _load_base_bundle(mode)
    merged = _deep_merge(merged, _load_preset(preset))
    merged = _deep_merge(merged, overrides or {})

    risk_policy = load_risk_policy()
    risk_overrides = merged.get("risk_overrides", {})

    cfg = TradingAgentConfig(
        mode=ModeConfig(**merged.get("mode", {})),
        logging=LoggingConfig(**merged.get("logging", {})),
        paths=PathsConfig(**merged.get("paths", {})),
        environment=EnvironmentConfig(**merged.get("environment", {})),
        rl=RLConfig(**merged.get("rl", {})),
        mt5=MT5Config(**merged.get("mt5", {})),
        risk=RiskConfig(policy=risk_policy, overrides=risk_overrides),
    )

    _apply_mt5_env_overrides(cfg)
    return cfg


def _apply_mt5_env_overrides(cfg: TradingAgentConfig) -> None:
    account = os.getenv("MT5_ACCOUNT")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")

    if account:
        try:
            cfg.mt5.account = int(account)
        except ValueError:
            pass  # Invalid account number, keep existing value
    if password:
        cfg.mt5.password = password
    if server:
        cfg.mt5.server = server


def get_config(
    mode: str = "training",
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> TradingAgentConfig:
    """Primary API to fetch the application configuration."""
    return load_app_config(mode=mode, preset=preset, overrides=overrides)


def build_trading_config(
    mode: str = "training",
    preset: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """Helper to go straight from YAML to envs.config.TradingConfig."""
    app_config = load_app_config(mode=mode, preset=preset, overrides=overrides)
    return app_config.to_trading_config()

