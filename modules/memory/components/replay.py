# modules/memory/components/replay.py
"""
Historical Replay Component
Analyzes sequences and patterns for learning optimization.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np

from .base import MemoryComponent


class ReplayComponent(MemoryComponent):
    """Historical replay analysis component."""

    # --------------------------- Tunables / Constants -------------------------

    _EPISODE_BUFFER_MAX: int = 50         # rolling episodes kept
    _TOP_PROFITABLE_MAX: int = 100        # keep top-N profitable sequences
    _QUALITY_WINDOW_MIN: int = 10         # min samples to report trend
    _TREND_SEGMENT: int = 5               # recent vs older segment length
    _TIME_STD_SCALE: float = 100.0        # scale factor for temporal regularity
    _EPS: float = 1e-12                   # numerical stability epsilon

    # ------------------------------- Lifecycle --------------------------------

    def _initialize_component(self) -> None:
        """Initialize replay-specific resources and metrics."""
        # Replay configuration
        self.replay_interval: int = int(self.config.replay_interval)
        self.replay_decay: float = float(self.config.replay_decay)
        self.sequence_len: int = int(self.config.sequence_len)
        self.profit_threshold: float = float(self.config.replay_profit_threshold)

        # State
        self.episode_buffer: deque[Dict[str, Any]] = deque(maxlen=self._EPISODE_BUFFER_MAX)
        self.profitable_sequences: List[Dict[str, Any]] = []
        self.sequence_patterns: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "total_pnl": 0.0, "avg_pnl": 0.0, "last_seen": 0.0}
        )
        self.current_sequence: List[Dict[str, Any]] = []

        # Metrics
        self.replay_bonus: float = 0.0
        self.best_sequence_pnl: float = 0.0
        self.sequences_analyzed: int = 0
        self.patterns_identified: int = 0

        # Quality tracking
        self.sequence_quality_scores: deque[float] = deque(maxlen=100)
        self.pattern_evolution: deque[Dict[str, Any]] = deque(maxlen=100)
        self.learning_effectiveness: deque[float] = deque(maxlen=200)

    # --------------------------------- Process --------------------------------

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process historical replay analysis."""
        try:
            sequence_data = self._extract_sequence_data(context)

            # Step-wise sequence handling
            seq_result = self._process_trading_sequence(sequence_data)

            # Episode closure handling
            if bool(sequence_data["episode_data"].get("completed", False)):
                episode_result = self._analyze_episode_patterns(sequence_data)
                seq_result.update(episode_result)

            # Periodic replay recommendations
            episode_idx = int(context.get("episode", 0) or 0)
            if self._should_replay(episode_idx):
                replay_result = self._generate_replay_recommendations()
                seq_result.update(replay_result)

            # Metrics
            self._update_metrics(seq_result)

            return self._format_output(seq_result)
        except Exception as e:
            self.log_error("Replay processing failed", e)
            return self._get_fallback_output()

    # ------------------------------- Extraction --------------------------------

    def _extract_sequence_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sequence-relevant data from context."""
        trades = context.get("trades", []) or []
        actions = context.get("actions", []) or []
        features = context.get("features", None)
        episode_data = context.get("episode_data", {}) or {}

        # Current action (vector-like) -> default neutral
        current_action: Optional[List[float]]
        if actions:
            last = actions[-1]
            if isinstance(last, (list, tuple, np.ndarray)):
                arr = np.asarray(last, dtype=np.float32).reshape(-1)
                current_action = arr.tolist()
            else:
                current_action = [0.0, 0.0]
        else:
            current_action = None

        return {
            "trades": trades,
            "features": features,
            "current_action": current_action,
            "episode_data": episode_data,
            "timestamp": time.time(),
        }

    # --------------------------- Sequence Processing ---------------------------

    def _process_trading_sequence(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Append current step and compute sequence quality."""
        if sequence_data["current_action"] is not None:
            step = {
                "action": sequence_data["current_action"],
                "timestamp": float(sequence_data["timestamp"]),
                "features": sequence_data["features"],
            }
            self.current_sequence.append(step)

        # Trim to configured maximum sequence length
        if len(self.current_sequence) > self.sequence_len:
            self.current_sequence = self.current_sequence[-self.sequence_len :]

        # Quality scoring
        quality = self._calculate_sequence_quality(self.current_sequence)
        self.sequence_quality_scores.append(float(quality))

        self.sequences_analyzed += 1

        return {
            "current_sequence_length": len(self.current_sequence),
            "sequence_quality": float(quality),
            "sequences_processed": self.sequences_analyzed,
        }

    def _calculate_sequence_quality(self, sequence: List[Dict[str, Any]]) -> float:
        """Calculate quality score of a trading sequence ∈ [0,1]."""
        if len(sequence) < 2:
            return 0.0
        try:
            # Action consistency (lower variance of magnitudes -> higher score)
            acts = []
            for step in sequence:
                a = step.get("action", [0.0, 0.0])
                v = np.linalg.norm(np.asarray(a, dtype=np.float32))
                acts.append(float(v))
            if len(acts) >= 2:
                action_variance = float(np.var(acts))
            else:
                action_variance = 0.0
            consistency_score = max(0.0, 1.0 - action_variance)

            # Temporal regularity (more regular spacing -> higher score)
            tss = [float(step.get("timestamp", 0.0)) for step in sequence if "timestamp" in step]
            if len(tss) >= 2:
                diffs = np.diff(np.asarray(tss, dtype=np.float64))
                # Guard against degenerate or identical timestamps
                temporal_std = float(np.std(diffs)) if diffs.size > 0 else 0.0
                temporal_score = max(0.0, 1.0 - temporal_std / self._TIME_STD_SCALE)
            else:
                temporal_score = 1.0

            quality = consistency_score * 0.6 + temporal_score * 0.4
            return float(np.clip(quality, 0.0, 1.0))
        except Exception:
            # In ambiguous situations, return neutral mid-quality
            return 0.5

    # ---------------------------- Episode & Patterns ---------------------------

    def _analyze_episode_patterns(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in a completed episode and reset the sequence."""
        episode_data = sequence_data["episode_data"]
        episode_pnl = float(episode_data.get("pnl", 0.0) or 0.0)

        # Record episode
        episode_record = {
            "sequence": list(self.current_sequence),
            "pnl": episode_pnl,
            "timestamp": time.time(),
        }
        self.episode_buffer.append(episode_record)

        # Profitable sequence tracking
        if episode_pnl > self.profit_threshold:
            self._process_profitable_sequence(episode_record)

        # Extract and record a sequence pattern
        pattern = self._extract_sequence_pattern(self.current_sequence)
        if pattern:
            self._update_pattern_data(pattern, episode_pnl)

        # Reset for next episode
        self.current_sequence = []

        return {
            "episode_analyzed": True,
            "episode_pnl": episode_pnl,
            "pattern_extracted": pattern is not None,
            "profitable_sequence": episode_pnl > self.profit_threshold,
        }

    def _extract_sequence_pattern(self, sequence: List[Dict[str, Any]]) -> Optional[str]:
        """Discretize actions to a compact pattern string (L/S/H)."""
        if len(sequence) < 3:
            return None
        try:
            elems: List[str] = []
            for step in sequence:
                a = step.get("action", [0.0, 0.0])
                a0 = float(np.asarray(a, dtype=np.float32).reshape(-1)[0]) if a is not None else 0.0
                if a0 > 0.5:
                    elems.append("L")  # Long
                elif a0 < -0.5:
                    elems.append("S")  # Short
                else:
                    elems.append("H")  # Hold
            return "".join(elems) if elems else None
        except Exception:
            return None

    def _update_pattern_data(self, pattern: str, pnl: float) -> None:
        """Update statistics for a discovered pattern."""
        data = self.sequence_patterns[pattern]
        data["count"] += 1
        data["total_pnl"] += float(pnl)
        denom = max(1, data["count"])
        data["avg_pnl"] = float(data["total_pnl"] / denom)
        data["last_seen"] = time.time()
        self.patterns_identified = len(self.sequence_patterns)

    def _process_profitable_sequence(self, episode: Dict[str, Any]) -> None:
        """Maintain a bounded set of top profitable sequences."""
        self.profitable_sequences.append(episode)
        # Keep only the best N sequences
        if len(self.profitable_sequences) > self._TOP_PROFITABLE_MAX:
            self.profitable_sequences.sort(key=lambda x: float(x.get("pnl", 0.0)), reverse=True)
            self.profitable_sequences = self.profitable_sequences[: self._TOP_PROFITABLE_MAX]

        # Update best PnL
        pnl_val = float(episode.get("pnl", 0.0))
        if pnl_val > self.best_sequence_pnl:
            self.best_sequence_pnl = pnl_val

    # ------------------------------ Recommendations ----------------------------

    def _should_replay(self, episode: int) -> bool:
        """Decide if replay should be triggered this episode."""
        return self.replay_interval > 0 and episode > 0 and episode % self.replay_interval == 0

    def _generate_replay_recommendations(self) -> Dict[str, Any]:
        """Generate replay recommendations and a decayed bonus."""
        if not self.profitable_sequences:
            return {"replay_recommended": False, "replay_bonus": 0.0}

        # Best sequence by realized PnL
        best_seq = max(self.profitable_sequences, key=lambda x: float(x.get("pnl", 0.0)))
        hours_since = (time.time() - float(best_seq.get("timestamp", time.time()))) / 3600.0
        decay = float(self.replay_decay) ** max(0.0, hours_since)
        bonus = (float(best_seq.get("pnl", 0.0)) / 100.0) * decay
        self.replay_bonus = float(bonus)

        # Best pattern by (avg_pnl * count)
        best_pattern: Optional[str] = None
        if self.sequence_patterns:
            best_pattern = max(
                self.sequence_patterns.items(),
                key=lambda kv: float(kv[1]["avg_pnl"]) * int(kv[1]["count"]),
            )[0]

        return {
            "replay_recommended": True,
            "replay_bonus": float(self.replay_bonus),
            "best_sequence_pnl": float(best_seq.get("pnl", 0.0)),
            "best_pattern": best_pattern,
            "total_patterns": len(self.sequence_patterns),
        }

    # --------------------------------- Metrics ---------------------------------

    def _update_metrics(self, result: Dict[str, Any]) -> None:
        """Update lightweight process metrics."""
        self._last_process = time.time()
        self._process_count += 1

    # --------------------------------- Output ----------------------------------

    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements."""
        avg_quality = float(np.mean(list(self.sequence_quality_scores))) if self.sequence_quality_scores else 0.0
        episodes_processed = len(self.episode_buffer)

        return {
            "learning_progress": {
                "episodes_processed": episodes_processed,
                "profitable_sequences": len(self.profitable_sequences),
                "patterns_learned": self.patterns_identified,
                "learning_rate": float(len(self.profitable_sequences) / max(1, episodes_processed)),
            },
            "pattern_analysis": {
                "total_patterns": len(self.sequence_patterns),
                "profitable_patterns": sum(1 for p in self.sequence_patterns.values() if float(p["avg_pnl"]) > 0.0),
                "best_pattern": result.get("best_pattern"),
                "pattern_confidence": float(self._calculate_pattern_confidence()),
            },
            "replay_sequences": {
                "total_sequences": len(self.profitable_sequences),
                "best_sequence_pnl": float(self.best_sequence_pnl),
                "replay_bonus": float(self.replay_bonus),
                "sequences_analyzed": int(self.sequences_analyzed),
            },
            "sequence_quality": {
                "current_quality": float(result.get("sequence_quality", 0.0)),
                "average_quality": float(avg_quality),
                "quality_trend": self._calculate_quality_trend(),
                "episodes_processed": episodes_processed,
            },
        }

    def _calculate_pattern_confidence(self) -> float:
        """Aggregate pattern confidence ∈ [0,1] from consistency and frequency."""
        if not self.sequence_patterns:
            return 0.0

        confidences: List[float] = []
        for data in self.sequence_patterns.values():
            count = int(data["count"])
            total_pnl = float(data["total_pnl"])
            avg_pnl = float(data["avg_pnl"])
            if count <= 0:
                continue

            # Consistency: stable average relative to total mass (bounded)
            denom = max(self._EPS, abs(total_pnl))
            consistency = 1.0 - min(1.0, abs(avg_pnl) / denom)

            # Frequency: saturate around 10 occurrences
            frequency = min(1.0, count / 10.0)

            confidences.append(float(np.clip(consistency * frequency, 0.0, 1.0)))

        return float(np.mean(confidences)) if confidences else 0.0

    def _calculate_quality_trend(self) -> str:
        """Compute quality trend label using last 10 samples split 5/5."""
        if len(self.sequence_quality_scores) < self._QUALITY_WINDOW_MIN:
            return "insufficient_data"

        recent = np.mean(list(self.sequence_quality_scores)[-self._TREND_SEGMENT :])
        older = np.mean(list(self.sequence_quality_scores)[-self._QUALITY_WINDOW_MIN : -self._TREND_SEGMENT])

        # Use a ±10% band
        if recent > older * 1.10:
            return "improving"
        if recent < older * 0.90:
            return "declining"
        return "stable"

    # -------------------------------- Fallback ---------------------------------

    def _get_fallback_output(self) -> Dict[str, Any]:
        """Conservative payload on error."""
        return {
            "learning_progress": {"episodes_processed": 0, "profitable_sequences": 0, "patterns_learned": 0, "learning_rate": 0.0},
            "pattern_analysis": {"total_patterns": 0, "profitable_patterns": 0, "best_pattern": None, "pattern_confidence": 0.0},
            "replay_sequences": {"total_sequences": 0, "best_sequence_pnl": 0.0, "replay_bonus": 0.0, "sequences_analyzed": 0},
            "sequence_quality": {"current_quality": 0.0, "average_quality": 0.0, "quality_trend": "insufficient_data", "episodes_processed": 0},
        }
