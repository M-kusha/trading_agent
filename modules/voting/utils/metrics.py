
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def safe_mean(values: List[Any], default: float = 0.0) -> float:
    if not values:
        return default

    try:
        cleaned = [safe_float(v) for v in values if v is not None]
        if not cleaned:
            return default
        return float(np.mean(cleaned))
    except Exception:
        return default


def safe_std(values: List[Any], default: float = 0.0) -> float:
    if not values or len(values) < 2:
        return default

    try:
        cleaned = [safe_float(v) for v in values if v is not None]
        if len(cleaned) < 2:
            return default
        return float(np.std(cleaned))
    except Exception:
        return default


def clip_value(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return float(np.clip(safe_float(value), low, high))
    except Exception:
        return (low + high) / 2


def calculate_agreement_score(
    actions: List[str],
    weights: Optional[List[float]] = None
) -> Tuple[float, str]:
    if not actions:
        return 0.0, "abstain"


    counts: Dict[str, float] = {}
    total_weight = 0.0

    for i, action in enumerate(actions):
        normalized = str(action).lower().strip()
        if normalized not in ("long", "short", "hold", "abstain"):
            normalized = "abstain"

        weight = weights[i] if weights and i < len(weights) else 1.0
        weight = safe_float(weight, 1.0)

        counts[normalized] = counts.get(normalized, 0.0) + weight
        total_weight += weight

    if total_weight == 0:
        return 0.0, "abstain"


    majority_action, majority_weight = max(counts.items(), key=lambda kv: kv[1])


    agreement_score = majority_weight / total_weight

    return clip_value(agreement_score), majority_action


def calculate_weighted_consensus(
    proposals: List[Dict[str, Any]],
    weight_key: str = "confidence"
) -> Dict[str, Any]:
    if not proposals:
        return {
            "consensus_action": "abstain",
            "consensus_score": 0.0,
            "weighted_confidence": 0.0,
            "vote_distribution": {},
            "participating_count": 0,
        }


    action_weights: Dict[str, float] = {}
    action_counts: Dict[str, int] = {}
    total_weight = 0.0

    for p in proposals:
        action = str(p.get("action", "abstain")).lower()
        weight = safe_float(p.get(weight_key, 1.0), 1.0)

        action_weights[action] = action_weights.get(action, 0.0) + weight
        action_counts[action] = action_counts.get(action, 0) + 1
        total_weight += weight

    if total_weight == 0:
        return {
            "consensus_action": "abstain",
            "consensus_score": 0.0,
            "weighted_confidence": 0.0,
            "vote_distribution": action_counts,
            "participating_count": len(proposals),
        }


    consensus_action, consensus_weight = max(
        action_weights.items(), key=lambda kv: kv[1]
    )
    consensus_score = consensus_weight / total_weight


    confidences = [safe_float(p.get("confidence", 0.5)) for p in proposals]
    weighted_confidence = safe_mean(confidences)

    return {
        "consensus_action": consensus_action,
        "consensus_score": clip_value(consensus_score),
        "weighted_confidence": clip_value(weighted_confidence),
        "vote_distribution": action_counts,
        "participating_count": len(proposals),
    }


def calculate_directional_consensus(proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not proposals:
        return {
            "direction": "neutral",
            "strength": 0.0,
            "directional_count": 0,
            "abstain_count": 0,
        }

    long_weight = 0.0
    short_weight = 0.0
    directional_count = 0
    abstain_count = 0

    for p in proposals:
        action = str(p.get("action", "abstain")).lower()
        confidence = safe_float(p.get("confidence", 0.5))
        strength = safe_float(p.get("signal_strength", 0.5))
        weight = confidence * strength

        if action == "long":
            long_weight += weight
            directional_count += 1
        elif action == "short":
            short_weight += weight
            directional_count += 1
        elif action in ("hold", "abstain"):
            abstain_count += 1

    total = long_weight + short_weight
    if total == 0:
        return {
            "direction": "neutral",
            "strength": 0.0,
            "directional_count": directional_count,
            "abstain_count": abstain_count,
        }

    if long_weight > short_weight:
        direction = "long"
        strength = (long_weight - short_weight) / total
    elif short_weight > long_weight:
        direction = "short"
        strength = (short_weight - long_weight) / total
    else:
        direction = "neutral"
        strength = 0.0

    return {
        "direction": direction,
        "strength": clip_value(strength),
        "directional_count": directional_count,
        "abstain_count": abstain_count,
    }


def calculate_diversity_index(values: List[Any]) -> float:
    if not values:
        return 0.0


    counts: Dict[Any, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1

    n = len(values)
    if n <= 1:
        return 0.0


    sum_squares = sum((c / n) ** 2 for c in counts.values())
    diversity = 1.0 - sum_squares

    return clip_value(diversity)


def calculate_expert_diversity(proposals: List[Dict[str, Any]]) -> float:
    if not proposals or len(proposals) < 2:
        return 0.0


    actions = [str(p.get("action", "abstain")).lower() for p in proposals]
    confidences = [safe_float(p.get("confidence", 0.5)) for p in proposals]
    strengths = [safe_float(p.get("signal_strength", 0.5)) for p in proposals]


    action_diversity = calculate_diversity_index(actions)


    conf_std = safe_std(confidences)
    conf_diversity = min(1.0, conf_std * 4)


    strength_std = safe_std(strengths)
    strength_diversity = min(1.0, strength_std * 4)


    diversity = (
        0.5 * action_diversity +
        0.3 * conf_diversity +
        0.2 * strength_diversity
    )

    return clip_value(diversity)


def calculate_correlation(
    series1: List[float],
    series2: List[float]
) -> float:
    if not series1 or not series2:
        return 0.0


    min_len = min(len(series1), len(series2))
    if min_len < 2:
        return 0.0

    try:
        s1 = np.array([safe_float(v) for v in series1[:min_len]])
        s2 = np.array([safe_float(v) for v in series2[:min_len]])


        if np.std(s1) == 0 or np.std(s2) == 0:
            return 0.0

        corr = np.corrcoef(s1, s2)[0, 1]
        return float(corr) if np.isfinite(corr) else 0.0

    except Exception:
        return 0.0


def calculate_pairwise_correlations(
    expert_histories: Dict[str, List[float]]
) -> Dict[Tuple[str, str], float]:
    correlations: Dict[Tuple[str, str], float] = {}
    experts = list(expert_histories.keys())

    for i, exp1 in enumerate(experts):
        for exp2 in experts[i+1:]:
            hist1 = expert_histories.get(exp1, [])
            hist2 = expert_histories.get(exp2, [])

            if hist1 and hist2:
                corr = calculate_correlation(hist1, hist2)
                correlations[(exp1, exp2)] = corr

    return correlations


def detect_suspicious_correlations(
    correlations: Dict[Tuple[str, str], float],
    threshold: float = 0.85
) -> List[Tuple[str, str, float]]:
    suspicious: List[Tuple[str, str, float]] = []

    for (exp1, exp2), corr in correlations.items():
        if abs(corr) >= threshold:
            suspicious.append((exp1, exp2, corr))


    suspicious.sort(key=lambda x: abs(x[2]), reverse=True)

    return suspicious


def calculate_collusion_score(
    proposals: List[Dict[str, Any]],
    correlation_threshold: float = 0.85
) -> float:
    if not proposals or len(proposals) < 2:
        return 0.0


    actions = [str(p.get("action", "abstain")).lower() for p in proposals]
    unique_actions = set(actions)

    if len(unique_actions) == 1 and len(proposals) >= 3:

        action_uniformity = 0.8
    else:

        action_counts: Dict[str, int] = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        max_count = max(action_counts.values())
        action_uniformity = max_count / len(actions)


    confidences = [safe_float(p.get("confidence", 0.5)) for p in proposals]
    conf_std = safe_std(confidences)
    confidence_uniformity = max(0.0, 1.0 - conf_std * 4)


    strengths = [safe_float(p.get("signal_strength", 0.5)) for p in proposals]
    strength_std = safe_std(strengths)
    strength_uniformity = max(0.0, 1.0 - strength_std * 4)


    collusion_score = (
        0.5 * action_uniformity +
        0.3 * confidence_uniformity +
        0.2 * strength_uniformity
    )


    if collusion_score < 0.6:
        collusion_score *= 0.5

    return clip_value(collusion_score)


def calculate_fragility_score(
    consensus_score: float,
    diversity_score: float,
    uncertainty: float = 0.0,
    market_volatility: float = 0.5
) -> float:

    consensus_factor = 1.0 - clip_value(consensus_score)


    diversity_deviation = abs(diversity_score - 0.4)
    diversity_factor = min(1.0, diversity_deviation * 2)


    uncertainty_factor = clip_value(uncertainty)


    volatility_factor = clip_value(market_volatility) * 0.5


    fragility = (
        0.35 * consensus_factor +
        0.25 * diversity_factor +
        0.25 * uncertainty_factor +
        0.15 * volatility_factor
    )

    return clip_value(fragility)


def calculate_voting_quality(
    consensus_score: float,
    diversity_score: float,
    participation_ratio: float,
    collusion_score: float = 0.0
) -> Dict[str, Any]:

    consensus = clip_value(consensus_score)
    diversity = clip_value(diversity_score)
    participation = clip_value(participation_ratio)
    collusion = clip_value(collusion_score)


    diversity_quality = 1.0 - abs(diversity - 0.4) * 2
    diversity_quality = max(0.0, diversity_quality)


    collusion_penalty = collusion * 0.5


    quality_score = (
        0.35 * consensus +
        0.25 * diversity_quality +
        0.25 * participation +
        0.15 * (1.0 - collusion_penalty)
    )


    if quality_score >= 0.7:
        quality_level = "high"
    elif quality_score >= 0.5:
        quality_level = "medium"
    elif quality_score >= 0.3:
        quality_level = "low"
    else:
        quality_level = "invalid"

    return {
        "quality_score": clip_value(quality_score),
        "quality_level": quality_level,
        "components": {
            "consensus": consensus,
            "diversity": diversity,
            "diversity_quality": diversity_quality,
            "participation": participation,
            "collusion": collusion,
        },
    }


def group_proposals_by_instrument(
    proposals: List[Dict[str, Any]],
    instrument_key: str = "instrument",
) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for p in proposals:
        inst_raw = p.get(instrument_key)
        if inst_raw is None:
            groups["_unknown"].append(p)
            continue

        inst = str(inst_raw).strip().upper()
        if not inst:
            inst = "_unknown"

        groups[inst].append(p)

    return dict(groups)


def calculate_per_instrument_metrics(
    proposals: List[Dict[str, Any]],
    instrument_key: str = "instrument",
    weight_key: str = "confidence",
) -> Dict[str, Dict[str, Any]]:
    if not proposals:
        return {}

    grouped = group_proposals_by_instrument(proposals, instrument_key=instrument_key)
    total = max(1, len(proposals))

    per_inst: Dict[str, Dict[str, Any]] = {}

    for inst, inst_props in grouped.items():
        if not inst_props:
            continue

        consensus = calculate_weighted_consensus(inst_props, weight_key=weight_key)
        directional = calculate_directional_consensus(inst_props)
        diversity = calculate_expert_diversity(inst_props)
        collusion = calculate_collusion_score(inst_props)

        participation_ratio = len(inst_props) / total

        quality = calculate_voting_quality(
            consensus_score=consensus.get("consensus_score", 0.0),
            diversity_score=diversity,
            participation_ratio=participation_ratio,
            collusion_score=collusion,
        )

        per_inst[inst] = {
            "consensus": consensus,
            "directional": directional,
            "diversity": diversity,
            "collusion": collusion,
            "quality": quality,
        }

    return per_inst
