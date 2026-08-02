# modules/memory/shared/pattern_detector.py
"""
Unified Pattern Detector
Advanced pattern detection and mining for memory components.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class Pattern:
    """Pattern representation."""
    id: str
    sequence: List[str]
    frequency: int
    confidence: float
    avg_outcome: float
    metadata: Dict[str, Any]

    def __hash__(self) -> int:  # explicit; dataclass(frozen=True) already supports hashing
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pattern):
            return False
        return self.id == other.id


class UnifiedPatternDetector:
    """
    Unified pattern detection using multiple algorithms.

    Features:
    - Sequential pattern mining (lightweight Apriori-like counting of subsequences)
    - Motif discovery (sliding window)
    - Anomaly detection (pattern-fit based)
    - Pattern clustering (similarity-based greedy assignment)
    - Confidence scoring
    """

    def __init__(self) -> None:
        # Pattern storage
        self.patterns: Dict[str, Pattern] = {}
        self.pattern_index: Dict[str, Set[str]] = defaultdict(set)

        # Configuration (tunable)
        self.min_support: int = 2
        self.min_confidence: float = 0.6
        self.max_pattern_length: int = 10

        # Statistics
        self.total_sequences_processed: int = 0
        self.total_patterns_found: int = 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def detect_patterns(
        self,
        sequences: List[List[Any]],
        outcomes: Optional[List[float]] = None
    ) -> List[Pattern]:
        """
        Detect frequent sequential patterns across sequences.

        Args:
            sequences: List of sequences (elements can be mixed types).
            outcomes: Optional per-sequence outcomes for avg_outcome scoring.

        Returns:
            List of detected Pattern objects that satisfy support/confidence thresholds.
        """
        if not sequences:
            return []

        # Normalize to string tokens
        string_sequences = self._convert_to_strings(sequences)

        # Count subsequences (per-sequence unique counting to avoid within-sequence inflation)
        frequent_counts = self._mine_frequent_patterns(string_sequences)

        # Build patterns with metrics
        patterns: List[Pattern] = []
        for pattern_seq, support in frequent_counts.items():
            if support < self.min_support:
                continue

            pat = self._create_pattern(
                pattern_seq=pattern_seq,
                support=support,
                sequences=string_sequences,
                outcomes=outcomes
            )
            if pat.confidence >= self.min_confidence:
                patterns.append(pat)
                self._store_pattern(pat)

        self.total_sequences_processed += len(sequences)
        self.total_patterns_found += len(patterns)
        return patterns

    def find_motifs(
        self,
        sequence: List[Any],
        window_size: int = 3
    ) -> List[Tuple[int, List[Any]]]:
        """
        Find recurring motifs in a single sequence via a fixed-size sliding window.

        Returns:
            Sorted list of (position, motif_list) with positions where motifs occur.
        """
        if window_size <= 0 or len(sequence) < window_size * 2:
            return []

        motifs: List[Tuple[int, List[Any]]] = []
        seen: Dict[Tuple[Any, ...], List[int]] = defaultdict(list)

        # Slide window through sequence and record positions
        for i in range(len(sequence) - window_size + 1):
            window = tuple(sequence[i:i + window_size])
            seen[window].append(i)

        # Collect motifs that repeat at least twice
        for motif, positions in seen.items():
            if len(positions) >= 2:
                motifs.extend((pos, list(motif)) for pos in positions)

        return sorted(motifs, key=lambda x: x[0])

    def detect_anomalies(
        self,
        sequence: List[Any],
        known_patterns: Optional[List[Pattern]] = None
    ) -> List[Tuple[int, Any, float]]:
        """
        Detect anomalies in a sequence by evaluating local fit to known patterns.

        Returns:
            List of (position, original_element, anomaly_score in [0,1]).
        """
        if not sequence:
            return []

        patterns = known_patterns or list(self.patterns.values())
        if not patterns:
            return []

        string_seq = self._convert_to_strings([sequence])[0]
        anomalies: List[Tuple[int, Any, float]] = []

        for i, _ in enumerate(string_seq):
            fit = self._calculate_pattern_fit(string_seq, i, patterns)
            anomaly_score = 1.0 - float(np.clip(fit, 0.0, 1.0))
            if anomaly_score > 0.7:
                anomalies.append((i, sequence[i], anomaly_score))

        return anomalies

    def match_pattern(
        self,
        sequence: List[Any],
        pattern: Pattern
    ) -> List[int]:
        """
        Find starting indices where a pattern matches in the given sequence.
        """
        if not pattern.sequence:
            return []

        string_seq = self._convert_to_strings([sequence])[0]
        pat = pattern.sequence
        pat_len = len(pat)

        matches: List[int] = []
        for i in range(len(string_seq) - pat_len + 1):
            if string_seq[i:i + pat_len] == pat:
                matches.append(i)
        return matches

    def get_pattern_by_id(self, pattern_id: str) -> Optional[Pattern]:
        """Lookup a stored pattern by ID."""
        return self.patterns.get(pattern_id)

    def get_patterns_by_outcome(self, min_outcome: float) -> List[Pattern]:
        """Retrieve patterns with average outcome >= min_outcome."""
        return [p for p in self.patterns.values() if p.avg_outcome >= min_outcome]

    def get_similar_patterns(
        self,
        pattern: Pattern,
        similarity_threshold: float = 0.7
    ) -> List[Tuple[Pattern, float]]:
        """
        Return patterns with similarity >= threshold, sorted by similarity desc.
        """
        similar: List[Tuple[Pattern, float]] = []
        for other in self.patterns.values():
            if other.id == pattern.id:
                continue
            sim = self._calculate_pattern_similarity(pattern, other)
            if sim >= similarity_threshold:
                similar.append((other, sim))
        return sorted(similar, key=lambda x: x[1], reverse=True)

    def cluster_patterns(self, n_clusters: int = 5) -> Dict[int, List[Pattern]]:
        """
        Cluster patterns into n clusters via greedy assignment on average similarity.
        This is a lightweight heuristic; it doesn't guarantee optimal partitions.
        """
        if n_clusters <= 0 or not self.patterns:
            return {}

        patterns_list = list(self.patterns.values())
        clusters: Dict[int, List[Pattern]] = defaultdict(list)

        # Seed clusters with first patterns (simple, deterministic)
        seeds = min(n_clusters, len(patterns_list))
        for i in range(seeds):
            clusters[i].append(patterns_list[i])

        # Assign remaining to the closest (highest average similarity) cluster
        for pat in patterns_list[seeds:]:
            best_cluster = 0
            best_sim = -1.0
            for cid, pats in clusters.items():
                sims = [self._calculate_pattern_similarity(pat, p) for p in pats]
                avg_sim = float(np.mean(sims)) if sims else 0.0
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_cluster = cid
            clusters[best_cluster].append(pat)

        return dict(clusters)

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _convert_to_strings(self, sequences: List[List[Any]]) -> List[List[str]]:
        """
        Convert arbitrary sequences to strings to standardize mining/logical ops.
        - Numeric scalars are discretized to bins (POS/NEG/LOW/HIGH/ZERO).
        - Vectors (lists/tuples/ndarrays) are discretized per element to H/M/L.
        - Other objects fall back to str(element).
        """
        out: List[List[str]] = []
        for seq in sequences:
            sseq: List[str] = []
            for el in seq:
                if isinstance(el, (list, tuple, np.ndarray)):
                    sseq.append(self._discretize_vector(el))
                elif isinstance(el, (int, float, np.number)):
                    sseq.append(self._discretize_scalar(float(el)))
                else:
                    sseq.append(str(el))
            out.append(sseq)
        return out

    def _discretize_vector(self, vector: Union[Sequence[float], np.ndarray]) -> str:
        """
        Discretize a vector to an H/M/L code per component (first 5 dims).
        Thresholds: > 0.5 => 'H', < -0.5 => 'L', else 'M'.
        """
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return ""
        codes: List[str] = []
        for val in arr[:5]:
            if val > 0.5:
                codes.append("H")
            elif val < -0.5:
                codes.append("L")
            else:
                codes.append("M")
        return "".join(codes)

    def _discretize_scalar(self, value: float) -> str:
        """
        Discretize a scalar into signed bins for coarse patterning.
        """
        if value > 0.5:
            return "POS_HIGH"
        if value > 0.0:
            return "POS_LOW"
        if value < -0.5:
            return "NEG_HIGH"
        if value < 0.0:
            return "NEG_LOW"
        return "ZERO"

    def _mine_frequent_patterns(
        self,
        sequences: List[List[str]]
    ) -> Dict[Tuple[str, ...], int]:
        """
        Count all unique subsequences (per sequence) up to max_pattern_length.

        Returns:
            Dict mapping (pattern_tuple) -> support_count.
        """
        patterns: Dict[Tuple[str, ...], int] = defaultdict(int)

        for seq in sequences:
            if not seq:
                continue

            seen_in_this_seq: Set[Tuple[str, ...]] = set()
            max_len = min(len(seq), self.max_pattern_length)

            # Inclusive upper bound for length
            for length in range(1, max_len + 1):
                # Slide window for contiguous subsequences
                for start in range(len(seq) - length + 1):
                    pat = tuple(seq[start:start + length])
                    if pat not in seen_in_this_seq:
                        patterns[pat] += 1
                        seen_in_this_seq.add(pat)

        return dict(patterns)

    def _create_pattern(
        self,
        pattern_seq: Tuple[str, ...],
        support: int,
        sequences: List[List[str]],
        outcomes: Optional[List[float]]
    ) -> Pattern:
        """
        Construct a Pattern with computed confidence and average outcome.
        Confidence = support / number_of_sequences.
        avg_outcome is computed over sequences containing the pattern (if outcomes provided).
        """
        # Stable, deterministic ID (fast and sufficient)
        pattern_str = "_".join(pattern_seq)
        pattern_id = hashlib.blake2s(pattern_str.encode("utf-8"), digest_size=8).hexdigest()

        confidence = support / max(len(sequences), 1)

        avg_outcome = 0.0
        if outcomes:
            matched: List[float] = []
            # Guard length mismatch gracefully
            limit = min(len(sequences), len(outcomes))
            for i in range(limit):
                if self._contains_pattern(sequences[i], pattern_seq):
                    matched.append(float(outcomes[i]))
            if matched:
                avg_outcome = float(np.mean(matched))

        return Pattern(
            id=pattern_id,
            sequence=list(pattern_seq),
            frequency=support,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            avg_outcome=avg_outcome,
            metadata={
                "length": len(pattern_seq),
                "first_element": pattern_seq[0] if pattern_seq else None,
                "last_element": pattern_seq[-1] if pattern_seq else None,
            },
        )

    @staticmethod
    def _contains_pattern(sequence: List[str], pattern: Tuple[str, ...]) -> bool:
        """Check if a contiguous pattern occurs in a sequence."""
        if not pattern or not sequence:
            return False
        pat_len = len(pattern)
        if pat_len > len(sequence):
            return False
        for i in range(len(sequence) - pat_len + 1):
            if tuple(sequence[i:i + pat_len]) == pattern:
                return True
        return False

    def _store_pattern(self, pattern: Pattern) -> None:
        """Insert/overwrite pattern and update inverted index by element."""
        self.patterns[pattern.id] = pattern
        for elem in pattern.sequence:
            self.pattern_index[elem].add(pattern.id)

    def _calculate_pattern_fit(
        self,
        sequence: List[str],
        position: int,
        patterns: List[Pattern]
    ) -> float:
        """
        Compute how well the token at `position` could participate in any known pattern.
        Returns max confidence among patterns that could align at this position.
        """
        if not patterns or not sequence:
            return 0.0

        best: float = 0.0
        n = len(sequence)

        for pat in patterns:
            pat_seq = pat.sequence
            L = len(pat_seq)
            if L == 0 or L > n:
                continue

            # Try alignments where pat[i] would map to sequence[position]
            for i in range(L):
                start = position - i
                end = start + L
                if start < 0 or end > n:
                    continue
                # Check contiguous match
                if sequence[start:end] == pat_seq:
                    best = max(best, pat.confidence)
                    # Early exit if perfect
                    if best >= 1.0:
                        return 1.0

        return float(np.clip(best, 0.0, 1.0))

    def _calculate_pattern_similarity(self, pattern1: Pattern, pattern2: Pattern) -> float:
        """
        Compute similarity between two patterns combining:
        - Jaccard similarity of token sets
        - Length similarity
        - Outcome proximity (squashed)
        """
        if pattern1.id == pattern2.id:
            return 1.0
        if pattern1.sequence == pattern2.sequence:
            # identical token sequence regardless of metadata
            return 1.0

        seq1 = pattern1.sequence
        seq2 = pattern2.sequence

        set1, set2 = set(seq1), set(seq2)
        if not set1 and not set2:
            jaccard = 1.0
        else:
            union = len(set1 | set2)
            inter = len(set1 & set2)
            jaccard = (inter / union) if union > 0 else 0.0

        max_len = max(len(seq1), len(seq2))
        min_len = min(len(seq1), len(seq2))
        length_sim = (min_len / max_len) if max_len > 0 else 1.0

        outcome_diff = abs(float(pattern1.avg_outcome) - float(pattern2.avg_outcome))
        # Soft proximity: larger diffs lower similarity, but smoothly
        outcome_sim = 1.0 / (1.0 + outcome_diff / 100.0)

        similarity = jaccard * 0.5 + length_sim * 0.2 + outcome_sim * 0.3
        return float(np.clip(similarity, 0.0, 1.0))
