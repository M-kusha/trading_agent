# modules/memory/components/budget.py
"""
Memory Budget Component
Optimizes memory allocation across components based on simple efficiency,
utilization, and recency signals.

Public outputs (contract):
- allocation_strategy
- budget_optimization
- memory_allocation
- memory_efficiency
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np

from .base import MemoryComponent


class BudgetComponent(MemoryComponent):
    """Memory budget optimization component."""

    # Tunables / guardrails
    _RECENT_WINDOW_SEC: int = 30 * 60  # 30 minutes
    _MIN_ALLOCATION: int = 50          # minimum per-component allocation
    _CHANGE_THRESHOLD_FRAC: float = 0.10  # only apply changes ≥10%
    _DEFAULT_COMPONENTS: tuple[str, ...] = (
        "replay",
        "compression",
        "mistakes",
        "neural",
        "playbook",
    )

    def _initialize_component(self) -> None:
        """Initialize budget-specific resources and state."""
        # Read config with safe fallbacks
        cfg = self.config
        self.rebalance_interval: int = int(getattr(cfg, "rebalance_interval", 50)) or 50
        self.utilization_target: float = float(getattr(cfg, "utilization_target", 0.8))
        self.efficiency_weight: float = float(getattr(cfg, "efficiency_weight", 0.7))
        self.recency_weight: float = float(getattr(cfg, "recency_weight", 0.3))
        self.max_memory_size: int = int(getattr(cfg, "max_memory_size", 10_000))

        # Current allocation across components (initial heuristic split)
        self.current_allocation: Dict[str, int] = {
            "replay": int(self.max_memory_size * 0.20),
            "compression": int(self.max_memory_size * 0.10),
            "mistakes": int(self.max_memory_size * 0.20),
            "neural": int(self.max_memory_size * 0.20),
            "playbook": int(self.max_memory_size * 0.30),
        }
        # Ensure minima and total budget constraints
        self._normalize_allocation_inplace(self.current_allocation)

        # Performance tracking per component
        self.component_performance: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "hits": 0,
                "profit": 0.0,
                "efficiency": 0.0,
                "recent_hits": deque(maxlen=100),  # timestamps of successful cycles
            }
        )
        # Make sure all default components are present in perf dict
        for c in self._DEFAULT_COMPONENTS:
            _ = self.component_performance[c]  # instantiate

        # Optimization bookkeeping
        self.optimization_count: int = 0
        self.optimization_history: deque[Dict[str, Any]] = deque(maxlen=50)
        self.allocation_changes: deque[Dict[str, Any]] = deque(maxlen=100)

        # Aggregate metrics
        self.total_profit: float = 0.0
        self.optimality_score: float = 0.5

        self._log_debug(
            "budget_initialized",
            details={
                "rebalance_interval": self.rebalance_interval,
                "utilization_target": self.utilization_target,
                "efficiency_weight": self.efficiency_weight,
                "recency_weight": self.recency_weight,
                "max_memory_size": self.max_memory_size,
            },
        )

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process budget optimization for this cycle."""
        try:
            self._update_component_performance(context)

            optimization_result: Dict[str, Any]
            if self._should_optimize():
                optimization_result = await self._optimize_allocation()
            else:
                optimization_result = {"optimization_performed": False}

            metrics = self._calculate_metrics()
            optimization_result.update(metrics)

            return self._format_output(optimization_result)
        except Exception as e:  # defensive: never break the memory pipeline
            self.log_error("Budget processing failed", e)
            return self._get_fallback_output()

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _update_component_performance(self, context: Dict[str, Any]) -> None:
        """
        Update performance metrics for each component based on upstream summaries.

        Expects context['component_performance'] like:
        {
          "replay": {"errors": int, "status": "success"|"error"|..., ...},
          ...
        }
        """
        comp_perf: Dict[str, Dict[str, Any]] = context.get("component_performance", {})

        now = time.time()
        for component, perf in comp_perf.items():
            if component not in self.component_performance:
                # unknown component: skip, keep only known ones
                continue

            slot = self.component_performance[component]

            # Count "hits" (no errors) and record recency
            errors = int(perf.get("errors", 0))
            status = str(perf.get("status", "unknown"))
            if errors == 0 and status in {"success", "healthy"}:
                slot["hits"] += 1
                slot["recent_hits"].append(now)

                # Naive profit proxy (you can replace with real PnL attribution)
                slot["profit"] += 0.1
                self.total_profit += 0.1

            # Efficiency: profit per successful cycle
            hits = max(1, slot["hits"])
            slot["efficiency"] = float(slot["profit"]) / float(hits)

    def _should_optimize(self) -> bool:
        """Return True when it's time to rebalance (interval-based)."""
        self.optimization_count += 1
        # Guard: avoid modulo-by-zero if misconfigured
        interval = max(1, self.rebalance_interval)
        return self.optimization_count % interval == 0

    async def _optimize_allocation(self) -> Dict[str, Any]:
        """Compute and apply an updated memory allocation."""
        efficiency_scores = self._calculate_efficiency_scores()
        optimal_allocation = self._calculate_optimal_allocation(efficiency_scores)
        changes = self._apply_allocation_changes(optimal_allocation)
        self._record_optimization(efficiency_scores, optimal_allocation, changes)

        return {
            "optimization_performed": True,
            "efficiency_scores": efficiency_scores,
            "optimal_allocation": optimal_allocation,
            "allocation_changes": changes,
        }

    def _calculate_efficiency_scores(self) -> Dict[str, float]:
        """
        Build an efficiency score per component:
        - base efficiency (profit/hit)
        - utilization vs target (hits / allocated capacity)
        - recent activity rate (events in last window)
        """
        scores: Dict[str, float] = {}
        now = time.time()

        for component, size in self.current_allocation.items():
            slot = self.component_performance[component]

            base_eff = float(slot["efficiency"])  # 0..inf
            alloc = max(1, int(size))
            hits = float(slot["hits"])
            utilization = hits / alloc  # normalized by allocation
            util_factor = min(1.0, utilization / max(1e-9, self.utilization_target))

            # Recent hits within window
            recent_hits = [ts for ts in slot["recent_hits"] if (now - ts) <= self._RECENT_WINDOW_SEC]
            # Convert to "per-minute rate" normalized by a nominal cap (10/min → factor 1.0)
            per_min_rate = len(recent_hits) / max(1.0, self._RECENT_WINDOW_SEC / 60.0)
            recency_factor = float(np.clip(per_min_rate / 10.0, 0.0, 1.0))

            score = (
                base_eff * self.efficiency_weight
                + util_factor * (1.0 - self.efficiency_weight) * 0.5
                + recency_factor * self.recency_weight * 0.5
            )
            scores[component] = max(0.0, float(score))

        return scores

    def _calculate_optimal_allocation(self, efficiency_scores: Dict[str, float]) -> Dict[str, int]:
        """Allocate total budget proportionally to efficiency scores with per-component minima."""
        total_budget = max(self._MIN_ALLOCATION * len(self.current_allocation), int(self.max_memory_size))
        # Normalize scores
        total_score = sum(efficiency_scores.values())
        if total_score <= 0.0:
            equal = total_budget // len(self.current_allocation)
            proposal = {c: equal for c in self.current_allocation}
            self._normalize_allocation_inplace(proposal)
            return proposal

        # Reserve minima, then distribute the remainder by proportion
        proposal: Dict[str, int] = {c: self._MIN_ALLOCATION for c in self.current_allocation}
        remaining = max(0, total_budget - self._MIN_ALLOCATION * len(self.current_allocation))
        for comp, score in efficiency_scores.items():
            share = 0 if total_score == 0 else (score / total_score)
            proposal[comp] = proposal.get(comp, self._MIN_ALLOCATION) + int(remaining * share)

        self._normalize_allocation_inplace(proposal)
        return proposal

    def _apply_allocation_changes(self, new_allocation: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """Apply only significant allocation changes and record the diff."""
        changes: Dict[str, Dict[str, int]] = {}
        for comp, new_size in new_allocation.items():
            if comp not in self.current_allocation:
                continue
            old_size = self.current_allocation[comp]
            if old_size <= 0:
                delta_frac = 1.0
            else:
                delta_frac = abs(new_size - old_size) / float(old_size)

            if delta_frac >= self._CHANGE_THRESHOLD_FRAC:
                self.current_allocation[comp] = int(new_size)
                changes[comp] = {"old": int(old_size), "new": int(new_size), "change": int(new_size - old_size)}

        if changes:
            self.allocation_changes.append({"timestamp": time.time(), "changes": changes})

        return changes

    def _record_optimization(
        self,
        efficiency_scores: Dict[str, float],
        optimal_allocation: Dict[str, int],
        changes: Dict[str, Dict[str, int]],
    ) -> None:
        """Persist a snapshot of the optimization step."""
        self.optimization_history.append(
            {
                "timestamp": time.time(),
                "efficiency_scores": dict(efficiency_scores),
                "optimal_allocation": dict(optimal_allocation),
                "changes_applied": len(changes),
                "total_profit": float(self.total_profit),
            }
        )

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Compute budget-level metrics for reporting."""
        self.optimality_score = self._calculate_optimality_score()

        total_hits = float(sum(slot["hits"] for slot in self.component_performance.values()))
        total_size = float(sum(self.current_allocation.values()))
        overall_efficiency = (total_hits / total_size) if total_size > 0 else 0.0

        return {
            "optimality_score": float(self.optimality_score),
            "total_profit": float(self.total_profit),
            "overall_efficiency": float(overall_efficiency),
            "optimization_count": int(self.optimization_count),
        }

    def _calculate_optimality_score(self) -> float:
        """Heuristic score ∈ [0,1] indicating whether profit is trending up recently."""
        recent = list(self.optimization_history)[-5:]
        if len(recent) < 2:
            return 0.5

        trend = 0
        for i in range(1, len(recent)):
            trend += 1 if recent[i]["total_profit"] > recent[i - 1]["total_profit"] else -1

        score = 0.5 + 0.5 * (trend / max(1, len(recent) - 1))
        return float(np.clip(score, 0.0, 1.0))

    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Render outputs matching the system contract."""
        memory_allocation = {
            "trades": self.current_allocation.get("replay", 500),
            "mistakes": self.current_allocation.get("mistakes", 100),
            "plays": self.current_allocation.get("playbook", 200),
        }

        memory_efficiency: Dict[str, Dict[str, float]] = {}
        now = time.time()
        for comp, perf in self.component_performance.items():
            alloc = max(1, self.current_allocation.get(comp, self._MIN_ALLOCATION))
            utilization = float(perf["hits"]) / float(alloc)

            recent_hits = [ts for ts in perf["recent_hits"] if (now - ts) <= self._RECENT_WINDOW_SEC]
            per_min_rate = len(recent_hits) / max(1.0, self._RECENT_WINDOW_SEC / 60.0)

            memory_efficiency[comp] = {
                "efficiency": float(perf["efficiency"]),
                "utilization": float(np.clip(utilization, 0.0, 1.0)),
                "recent_hit_rate_per_min": float(per_min_rate),
                "total_hits": int(perf["hits"]),
                "total_profit": float(perf["profit"]),
            }

        return {
            "allocation_strategy": {
                "allocation_method": "efficiency_based",
                "rebalance_frequency": int(self.rebalance_interval),
                "efficiency_weight": float(self.efficiency_weight),
                "recent_changes": int(len(self.allocation_changes)),
            },
            "budget_optimization": {
                "optimality_score": float(self.optimality_score),
                "total_profit": float(self.total_profit),
                "optimization_count": int(self.optimization_count),
                "last_optimization": time.time(),
            },
            "memory_allocation": memory_allocation,
            "memory_efficiency": memory_efficiency,
        }

    def _get_fallback_output(self) -> Dict[str, Any]:
        """Return a conservative default payload on error."""
        return {
            "allocation_strategy": {
                "allocation_method": "default",
                "rebalance_frequency": int(self.rebalance_interval),
                "efficiency_weight": float(self.efficiency_weight),
                "recent_changes": 0,
            },
            "budget_optimization": {
                "optimality_score": 0.5,
                "total_profit": 0.0,
                "optimization_count": int(self.optimization_count),
                "last_optimization": 0.0,
            },
            "memory_allocation": {"trades": 500, "mistakes": 100, "plays": 200},
            "memory_efficiency": {},
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _normalize_allocation_inplace(self, alloc: Dict[str, int]) -> None:
        """
        Ensure allocations respect minima and total budget exactly.
        Mutates `alloc` in place.
        """
        # Enforce minima
        for c in list(alloc.keys()):
            alloc[c] = max(self._MIN_ALLOCATION, int(alloc[c]))

        total = sum(alloc.values())
        if total == 0:
            per = max(self._MIN_ALLOCATION, self.max_memory_size // max(1, len(alloc)))
            for c in alloc:
                alloc[c] = per
            total = sum(alloc.values())

        # Scale to budget if needed
        if total != self.max_memory_size:
            scale = self.max_memory_size / float(total)
            # First pass: scale
            for c in alloc:
                alloc[c] = max(self._MIN_ALLOCATION, int(round(alloc[c] * scale)))
            # Second pass: fix rounding drift to match exact budget
            drift = self.max_memory_size - sum(alloc.values())
            if drift != 0:
                # Adjust largest buckets by ±1 until drift is corrected
                # (deterministic order by component name)
                keys_sorted: List[str] = sorted(alloc.keys())
                idx = 0
                step = 1 if drift > 0 else -1
                while drift != 0 and keys_sorted:
                    k = keys_sorted[idx % len(keys_sorted)]
                    new_val = alloc[k] + step
                    if new_val >= self._MIN_ALLOCATION:
                        alloc[k] = new_val
                        drift -= step
                    idx += 1
