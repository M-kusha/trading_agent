#!/usr/bin/env python3
"""
Quick training performance analyzer for Modern PPO runs.

Reads:
  - logs/training/enhanced_metrics_*.jsonl  (rich SB3 + system metrics)
  - logs/training/monitor_0.csv.monitor.csv (episode rewards / lengths)
  - logs/infobus_session.json               (if present; for closed trades, memory gate stats)

Outputs a concise English summary:
  - How far training progressed (timesteps, episodes, reward curve)
  - Account-level performance (PnL, drawdown)
  - Trade statistics (if closed positions are available)
  - Memory / risk gate influence (vetoes, risk multipliers) where possible
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
TRAINING_DIR = LOGS_DIR / "training"


def _load_enhanced_metrics() -> List[Dict[str, Any]]:
    """Load all enhanced_metrics_*.jsonl entries sorted by timestep."""
    entries: List[Dict[str, Any]] = []
    if not TRAINING_DIR.exists():
        return entries

    for path_str in sorted(glob(str(TRAINING_DIR / "enhanced_metrics_*.jsonl"))):
        path = Path(path_str)
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        entries.append(obj)
                    except Exception:
                        continue
        except Exception:
            continue

    # Sort by 'timestep' if available
    def _key(o: Dict[str, Any]) -> float:
        try:
            return float(o.get("timestep", o.get("total_timesteps", 0.0)) or 0.0)
        except Exception:
            return 0.0

    entries.sort(key=_key)
    return entries


def _load_monitor_csv() -> List[Tuple[float, int, float]]:
    """
    Load Monitor CSV if present.

    Stable-Baselines Monitor file format here starts with a JSON header and then:
      r,l,t
      reward,episode_length,total_timesteps
    """
    path = TRAINING_DIR / "monitor_0.csv.monitor.csv"
    if not path.exists():
        return []

    rows: List[Tuple[float, int, float]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return rows

    for line in text:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Expect r,l,t lines
        parts = line.split(",")
        if len(parts) != 3:
            # Some monitor files can get corrupted; skip bad lines
            continue
        try:
            r = float(parts[0])
            l = int(float(parts[1]))
            t = float(parts[2])
            rows.append((r, l, t))
        except Exception:
            continue
    return rows


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _load_infobus_session() -> Dict[str, Any]:
    """
    Load SmartInfoBus session export if available.

    This may contain:
      - closed_positions
      - memory_gate / memory_vote snapshots
      - account_state history
    """
    path = LOGS_DIR / "infobus_session.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    except Exception:
        return {}


@dataclass
class RewardSummary:
    episodes: int = 0
    mean_reward: float = 0.0
    last_reward: float = 0.0
    best_reward: float = 0.0
    reward_trend: Optional[str] = None


@dataclass
class AccountSummary:
    initial_balance: Optional[float] = None
    last_balance: Optional[float] = None
    last_equity: Optional[float] = None
    max_drawdown: Optional[float] = None
    equity_change_pct: Optional[float] = None


@dataclass
class TradeSummary:
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.total_profit - self.total_loss

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return (self.wins / total) if total > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        if self.total_loss == 0.0:
            return math.inf if self.total_profit > 0 else 0.0
        return self.total_profit / self.total_loss


@dataclass
class MemoryRiskSummary:
    memory_veto_count: int = 0
    memory_veto_examples: List[str] = None
    risk_multiplier_samples: List[float] = None

    def __post_init__(self):
        if self.memory_veto_examples is None:
            self.memory_veto_examples = []
        if self.risk_multiplier_samples is None:
            self.risk_multiplier_samples = []


def summarize_rewards(metrics: List[Dict[str, Any]], monitor_rows: List[Tuple[float, int, float]]) -> RewardSummary:
    summary = RewardSummary()

    if metrics:
        last = metrics[-1]
        summary.episodes = int(last.get("episodes", 0) or 0)
        summary.mean_reward = float(last.get("episode_reward_mean", 0.0) or 0.0)
        summary.last_reward = float(last.get("current_episode_reward", 0.0) or 0.0)
        summary.best_reward = float(last.get("best_episode_reward", 0.0) or 0.0)

        # Simple trend: compare early vs late episode_reward_mean
        early = metrics[len(metrics) // 10] if len(metrics) >= 10 else metrics[0]
        early_mean = float(early.get("episode_reward_mean", 0.0) or 0.0)
        delta = summary.mean_reward - early_mean
        if abs(delta) < 1e-3:
            summary.reward_trend = "flat"
        elif delta > 0:
            summary.reward_trend = "improving"
        else:
            summary.reward_trend = "worsening"

    # Fallback: if no enhanced metrics but we have monitor CSV
    if not metrics and monitor_rows:
        summary.episodes = len(monitor_rows)
        rewards = [r for (r, _, _) in monitor_rows]
        if rewards:
            summary.mean_reward = sum(rewards) / len(rewards)
            summary.last_reward = rewards[-1]
            summary.best_reward = max(rewards)
            if len(rewards) > 10:
                early_mean = sum(rewards[: len(rewards) // 10]) / max(len(rewards) // 10, 1)
                delta = summary.mean_reward - early_mean
                summary.reward_trend = "improving" if delta > 0 else "worsening"

    return summary


def summarize_account(metrics: List[Dict[str, Any]]) -> AccountSummary:
    summary = AccountSummary()
    if not metrics:
        return summary

    # Assume initial balance is env_balance at first metric entry
    first = metrics[0]
    last = metrics[-1]
    summary.initial_balance = _safe_float(first.get("env_balance", first.get("env_equity", 0.0)), None)  # type: ignore[arg-type]
    if summary.initial_balance is not None and summary.initial_balance <= 0:
        summary.initial_balance = None

    summary.last_balance = _safe_float(last.get("env_balance", last.get("env_equity", 0.0)), None)  # type: ignore[arg-type]
    summary.last_equity = _safe_float(last.get("env_equity", last.get("env_balance", 0.0)), None)  # type: ignore[arg-type]

    if summary.initial_balance is not None and summary.last_equity is not None:
        try:
            summary.equity_change_pct = (summary.last_equity - summary.initial_balance) / max(
                summary.initial_balance, 1e-9
            )
        except Exception:
            summary.equity_change_pct = None

    # Max drawdown from env_drawdown if present
    max_dd = 0.0
    for m in metrics:
        dd = _safe_float(m.get("env_drawdown", 0.0), 0.0)
        if dd > max_dd:
            max_dd = dd
    summary.max_drawdown = max_dd if max_dd > 0 else None
    return summary


def _extract_closed_positions(store: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Try to find closed positions in the InfoBus session dump.

    The exact keying can vary; this is best-effort and will silently return empty list if not found.
    """
    candidates: List[List[Dict[str, Any]]] = []
    # flat key
    closed = store.get("closed_positions")
    if isinstance(closed, list):
        candidates.append(closed)  # type: ignore[arg-type]

    # nested under trading / position modules
    for key, val in store.items():
        if not isinstance(val, dict):
            continue
        for sub_key in ("closed_positions", "positions_closed", "recent_closed_positions"):
            cp = val.get(sub_key)
            if isinstance(cp, list):
                candidates.append(cp)  # type: ignore[arg-type]

    # choose the largest candidate (most trades)
    if not candidates:
        return []
    longest = max(candidates, key=len)
    # Ensure elements are dict-like
    return [p for p in longest if isinstance(p, dict)]


def summarize_trades(bus_store: Dict[str, Any]) -> TradeSummary:
    summary = TradeSummary()
    closed_positions = _extract_closed_positions(bus_store)
    if not closed_positions:
        return summary

    for pos in closed_positions:
        pnl = _safe_float(pos.get("pnl", pos.get("profit", 0.0)), 0.0)
        if pnl > 0:
            summary.wins += 1
            summary.total_profit += pnl
        elif pnl < 0:
            summary.losses += 1
            summary.total_loss += abs(pnl)
        # Breakeven trades (pnl == 0) are counted towards total_trades but neither wins nor losses
        summary.total_trades += 1

    return summary


def summarize_memory_risk(bus_store: Dict[str, Any]) -> MemoryRiskSummary:
    summary = MemoryRiskSummary()
    if not bus_store:
        return summary

    # Count vetoed trades in any memory_gate snapshots we can find
    def scan_gate(obj: Any):
        if isinstance(obj, dict):
            mg = obj
        else:
            return
        if "memory_gate" in mg and isinstance(mg["memory_gate"], dict):
            gate = mg["memory_gate"]
        else:
            gate = mg
        if isinstance(gate, dict):
            if gate.get("veto"):
                summary.memory_veto_count += 1
                reasons = gate.get("reasons") or gate.get("gate_reasons") or []
                if isinstance(reasons, list) and reasons:
                    msg = "; ".join(str(r) for r in reasons[:2])
                    if msg and msg not in summary.memory_veto_examples:
                        summary.memory_veto_examples.append(msg)
            rm = gate.get("risk_multiplier")
            if isinstance(rm, (int, float)):
                summary.risk_multiplier_samples.append(float(rm))

    # Look across all values in the store
    for key, val in bus_store.items():
        if "memory_gate" in str(key):
            scan_gate(val)
        elif isinstance(val, dict) and "memory_gate" in val:
            scan_gate(val["memory_gate"])

        # Nested dicts that might contain memory_gate
        if isinstance(val, dict):
            for sub_val in val.values():
                if isinstance(sub_val, dict) and "memory_gate" in sub_val:
                    scan_gate(sub_val["memory_gate"])

    return summary


def _format_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.1f}%"


def _format_currency(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    sign = "+" if x >= 0 else "-"
    return f"{sign}${abs(x):,.2f}"


def main() -> int:
    metrics = _load_enhanced_metrics()
    monitor_rows = _load_monitor_csv()
    bus_store = _load_infobus_session()

    reward_summary = summarize_rewards(metrics, monitor_rows)
    account_summary = summarize_account(metrics)
    trade_summary = summarize_trades(bus_store)
    memory_risk_summary = summarize_memory_risk(bus_store)

    # ----- English report -----
    print("\n=== Modern PPO Training Performance Report ===\n")

    # Progress / learning
    if metrics:
        last = metrics[-1]
        total_steps = _safe_float(last.get("total_timesteps", last.get("timestep", 0.0)), 0.0)
        elapsed = _safe_float(last.get("elapsed_time_s", 0.0), 0.0)
        steps_per_second = _safe_float(last.get("steps_per_second", 0.0), 0.0)
        progress_pct = _safe_float(last.get("progress_pct", 0.0), 0.0)

        print(">>> Training Progress")
        print(
            f"- Reached approximately {total_steps:,.0f} timesteps "
            f"({progress_pct * 100:.1f}% of target) over {reward_summary.episodes} episodes."
        )
        if elapsed > 0:
            hours = elapsed / 3600.0
            print(f"- Wall-clock training time so far is about {hours:.2f} hours.")
        if steps_per_second > 0:
            print(f"- Effective training speed is roughly {steps_per_second:,.1f} steps/second.")
    else:
        print(">>> Training Progress")
        print("- No enhanced training metrics found; training may not have run or logs were not written.")

    print()
    print(">>> Reward & Learning Signal")
    print(
        f"- Average episode reward is {reward_summary.mean_reward:.2f}, "
        f"with a best episode around {reward_summary.best_reward:.2f}."
    )
    if reward_summary.reward_trend:
        trend_phrase = {
            "improving": "improving overall",
            "worsening": "trending down overall",
            "flat": "roughly flat so far",
        }.get(reward_summary.reward_trend, reward_summary.reward_trend)
        print(f"- The reward curve appears to be {trend_phrase}.")
    if metrics:
        last = metrics[-1]
        recent_mean = _safe_float(last.get("episode_reward_recent", 0.0), 0.0)
        std = _safe_float(last.get("episode_reward_std", 0.0), 0.0)
        print(
            f"- Recent episode reward (short window) is about {recent_mean:.2f} "
            f"with volatility (std) around {std:.2f}."
        )

    # Account performance
    print()
    print(">>> Account / Equity Performance")
    if account_summary.initial_balance is not None:
        last_balance = account_summary.last_balance or account_summary.initial_balance
        net_pnl = last_balance - account_summary.initial_balance
        direction = "profit" if net_pnl >= 0 else "loss"
        print(
            f"- Starting balance was about ${account_summary.initial_balance:,.2f}; "
            f"current balance is {last_balance:,.2f} "
            f"({direction} of {_format_currency(net_pnl)})."
        )
        if account_summary.equity_change_pct is not None:
            print(
                f"- Overall equity change during training is roughly {_format_pct(account_summary.equity_change_pct)}."
            )
        if account_summary.max_drawdown is not None:
            print(
                f"- Maximum observed drawdown during training is around {_format_pct(account_summary.max_drawdown)}."
            )
    else:
        print("- Could not infer account balance from metrics.")

    # Trade-level stats
    print()
    print(">>> Trade-Level Behaviour")
    if trade_summary.total_trades > 0:
        print(
            f"- The system executed roughly {trade_summary.total_trades} closed trades "
            f"({trade_summary.wins} wins / {trade_summary.losses} losses)."
        )
        print(
            f"- Win rate on those trades is about {_format_pct(trade_summary.win_rate)}, "
            f"with net P&L of {_format_currency(trade_summary.net_pnl)} "
            f"and profit factor {trade_summary.profit_factor:.2f}."
        )
    else:
        print(
            "- No closed trade history was found in the exported InfoBus session; "
            "trade-level stats (win rate, profit factor) are not available from this run."
        )

    # Memory / risk gate
    print()
    print(">>> Memory & Risk Gate Influence")
    if memory_risk_summary.memory_veto_count > 0 or memory_risk_summary.risk_multiplier_samples:
        print(
            f"- The memory/risk gate intervened multiple times; we saw roughly "
            f"{memory_risk_summary.memory_veto_count} explicit veto events in the stored session."
        )
        if memory_risk_summary.risk_multiplier_samples:
            avg_mult = sum(memory_risk_summary.risk_multiplier_samples) / len(
                memory_risk_summary.risk_multiplier_samples
            )
            below_one = [m for m in memory_risk_summary.risk_multiplier_samples if m < 1.0]
            frac_reduced = len(below_one) / max(len(memory_risk_summary.risk_multiplier_samples), 1)
            print(
                f"- On average, the memory gate risk multiplier was about {avg_mult:.2f}; "
                f"in roughly {_format_pct(frac_reduced)} of sampled cases it reduced trade size below the normal level."
            )
        if memory_risk_summary.memory_veto_examples:
            examples = "; ".join(memory_risk_summary.memory_veto_examples[:3])
            print(f"- Example reasons for memory-based vetoes include: \"{examples}\".")
    else:
        print(
            "- No explicit memory_gate snapshots were found in the InfoBus session; "
            "the memory/risk system may not have been fully active during this run, or its outputs were not persisted."
        )

    # PPO / model diagnostics & system health
    if metrics:
        print()
        print(">>> PPO / Model Diagnostics")
        last = metrics[-1]
        approx_kl = _safe_float(last.get("approx_kl", 0.0), 0.0)
        clip_fraction = _safe_float(last.get("clip_fraction", 0.0), 0.0)
        entropy_loss = _safe_float(last.get("entropy_loss", 0.0), 0.0)
        policy_loss = _safe_float(last.get("policy_loss", 0.0), 0.0)
        value_loss = _safe_float(last.get("value_loss", 0.0), 0.0)
        explained_variance = _safe_float(last.get("explained_variance", 0.0), 0.0)
        decision_conf = _safe_float(last.get("decision_confidence", 0.0), 0.0)
        risk_score = _safe_float(last.get("risk_score", 0.0), 0.0)
        risk_level = _safe_float(last.get("risk_level", 0.0), 0.0)

        print(
            f"- Latest PPO diagnostics: approx_KL={approx_kl:.4f}, clip_fraction={clip_fraction:.3f}, "
            f"entropy_loss={entropy_loss:.4f}, policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}."
        )
        print(
            f"- Value function explained variance is {explained_variance:.3f}, "
            "indicating how much of the return the critic currently explains."
        )
        print(
            f"- The system's risk score is {risk_score:.2f} with normalized risk level {risk_level:.2f}, "
            f"and average decision confidence around {decision_conf:.2f}."
        )

        # System health snapshot
        sys_health = _safe_float(last.get("system_health_score", 0.0), 0.0)
        sys_status = str(last.get("system_health_status", "unknown"))
        cpu_pct = _safe_float(last.get("system_cpu_percent", 0.0), 0.0)
        mem_pct = _safe_float(last.get("system_memory_percent", 0.0), 0.0)
        total_modules = int(_safe_float(last.get("total_modules", 0.0), 0.0))
        healthy_modules = int(_safe_float(last.get("healthy_modules", 0.0), 0.0))
        model_device = str(last.get("model_device", "cpu"))
        perf_ms = _safe_float(last.get("performance_avg_ms", 0.0), 0.0)
        perf_succ = _safe_float(last.get("performance_success_rate", 0.0), 0.0)

        print()
        print(">>> System Health & Infrastructure")
        print(
            f"- Reported system health score is {sys_health:.1f} ({sys_status}); "
            f"CPU usage around {cpu_pct:.1f}%, memory usage around {mem_pct:.1f}%."
        )
        print(
            f"- {healthy_modules} out of {total_modules} modules are marked healthy, "
            f"with average module operation time {perf_ms:.2f} ms and success rate {perf_succ:.1f}%."
        )
        print(f"- Model is currently running on device: {model_device}.")

    print("\n(Report generated from existing log files; no training code was modified.)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
