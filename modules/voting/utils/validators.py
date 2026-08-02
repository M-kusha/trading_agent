
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from modules.voting.core.constants import VotingAction


def validate_action(action: Any) -> Tuple[bool, str]:
    if action is None:
        return False, "abstain"

    if isinstance(action, VotingAction):
        return True, action.value

    if isinstance(action, str):
        normalized = action.lower().strip()
        valid_actions = {"long", "short", "hold", "abstain"}
        if normalized in valid_actions:
            return True, normalized

    return False, "abstain"


validate_voting_action = validate_action


def validate_confidence(confidence: Any) -> float:
    try:
        val = float(confidence)
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.0


def validate_signal_strength(strength: Any) -> float:
    try:
        val = float(strength)
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.0


def validate_proposal(proposal: Any) -> Dict[str, Any]:
    if not isinstance(proposal, dict):
        return {
            "action": "abstain",
            "confidence": 0.0,
            "signal_strength": 0.0,
            "reason": "Invalid proposal type",
            "valid": False,
        }

    is_valid_action, action = validate_action(proposal.get("action"))
    confidence = validate_confidence(proposal.get("confidence", 0.0))
    signal_strength = validate_signal_strength(
        proposal.get("signal_strength", proposal.get("intensity", 0.0))
    )

    return {
        "action": action,
        "confidence": confidence,
        "signal_strength": signal_strength,
        "reason": str(proposal.get("reason", proposal.get("thesis", ""))),
        "expert": str(proposal.get("expert", proposal.get("module", "unknown"))),
        "timestamp": proposal.get("timestamp", datetime.now().isoformat()),
        "valid": is_valid_action and confidence > 0,
    }


def validate_proposals(proposals: List[Any]) -> List[Dict[str, Any]]:
    if not isinstance(proposals, (list, tuple)):
        return []

    validated = []
    for p in proposals:
        v = validate_proposal(p)
        if v.get("valid"):
            validated.append(v)

    return validated


def sanitize_bus_key(key: str) -> str:
    if not isinstance(key, str):
        return "unknown_key"


    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', key)


    if sanitized and not sanitized[0].isalpha():
        sanitized = "key_" + sanitized

    return sanitized or "unknown_key"


def validate_timestamp(timestamp: Any, max_age_seconds: float = 60.0) -> Tuple[bool, str]:
    if not timestamp:
        return False, datetime.now().isoformat()

    try:
        if isinstance(timestamp, datetime):
            ts = timestamp
        elif isinstance(timestamp, str):

            ts_str = timestamp.replace("Z", "+00:00")
            ts = datetime.fromisoformat(ts_str)
        else:
            return False, datetime.now().isoformat()

        age = (datetime.now() - ts.replace(tzinfo=None)).total_seconds()
        is_fresh = abs(age) < max_age_seconds

        return is_fresh, ts.isoformat()

    except Exception:
        return False, datetime.now().isoformat()


def validate_expert_name(name: Any) -> str:
    if not name or not isinstance(name, str):
        return "UnknownExpert"


    cleaned = re.sub(r'[^a-zA-Z0-9]', '', str(name))

    return cleaned or "UnknownExpert"


def validate_instrument(instrument: Any) -> Optional[str]:
    if not instrument or not isinstance(instrument, str):
        return None


    normalized = instrument.strip().upper()


    if len(normalized) >= 6 and normalized.isalpha():
        return normalized


    if len(normalized) >= 3:
        return normalized

    return None


def validate_weight(weight: Any) -> float:
    try:
        val = float(weight)
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.0


def validate_weights(weights: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(weights, dict):
        return {}

    validated = {}
    for key, value in weights.items():
        if isinstance(key, str) and key:
            validated[key] = validate_weight(value)

    return validated
