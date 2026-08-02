"""
Voting Validators
=================
Input validation utilities for the voting system.
Ensures data integrity throughout the voting pipeline.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from modules.voting.core.constants import VotingAction


def validate_action(action: Any) -> Tuple[bool, str]:
    """
    Validate a voting action.
    
    Args:
        action: The action to validate
        
    Returns:
        Tuple of (is_valid, normalized_action)
    """
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


# Alias for consistency with naming convention
validate_voting_action = validate_action


def validate_confidence(confidence: Any) -> float:
    """
    Validate and normalize a confidence value.
    
    Args:
        confidence: The confidence value to validate
        
    Returns:
        Normalized confidence in range [0.0, 1.0]
    """
    try:
        val = float(confidence)
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.0


def validate_signal_strength(strength: Any) -> float:
    """
    Validate and normalize signal strength.
    
    Args:
        strength: The signal strength to validate
        
    Returns:
        Normalized strength in range [0.0, 1.0]
    """
    try:
        val = float(strength)
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.0


def validate_proposal(proposal: Any) -> Dict[str, Any]:
    """
    Validate and normalize a voting proposal.
    
    Args:
        proposal: Dict-like proposal to validate
        
    Returns:
        Normalized proposal dict
    """
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
    """
    Validate a list of proposals.
    
    Args:
        proposals: List of proposals to validate
        
    Returns:
        List of validated proposals
    """
    if not isinstance(proposals, (list, tuple)):
        return []
    
    validated = []
    for p in proposals:
        v = validate_proposal(p)
        if v.get("valid"):
            validated.append(v)
    
    return validated


def sanitize_bus_key(key: str) -> str:
    """
    Sanitize a SmartInfoBus key name.
    
    Args:
        key: The key to sanitize
        
    Returns:
        Sanitized key name
    """
    if not isinstance(key, str):
        return "unknown_key"
    
    # Remove invalid characters, keep alphanumeric and underscores
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', key)
    
    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = "key_" + sanitized
    
    return sanitized or "unknown_key"


def validate_timestamp(timestamp: Any, max_age_seconds: float = 60.0) -> Tuple[bool, str]:
    """
    Validate a timestamp and check freshness.
    
    Args:
        timestamp: ISO format timestamp string
        max_age_seconds: Maximum age in seconds
        
    Returns:
        Tuple of (is_fresh, normalized_timestamp)
    """
    if not timestamp:
        return False, datetime.now().isoformat()
    
    try:
        if isinstance(timestamp, datetime):
            ts = timestamp
        elif isinstance(timestamp, str):
            # Handle various ISO formats
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
    """
    Validate and normalize an expert name.
    
    Args:
        name: Expert name to validate
        
    Returns:
        Normalized expert name
    """
    if not name or not isinstance(name, str):
        return "UnknownExpert"
    
    # Remove whitespace and special characters
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', str(name))
    
    return cleaned or "UnknownExpert"


def validate_instrument(instrument: Any) -> Optional[str]:
    """
    Validate an instrument symbol.
    
    Args:
        instrument: Instrument symbol to validate
        
    Returns:
        Validated instrument or None
    """
    if not instrument or not isinstance(instrument, str):
        return None
    
    # Normalize to uppercase
    normalized = instrument.strip().upper()
    
    # Basic forex/commodity pattern check
    if len(normalized) >= 6 and normalized.isalpha():
        return normalized
    
    # Handle symbols with numbers (indices, etc.)
    if len(normalized) >= 3:
        return normalized
    
    return None


def validate_weight(weight: Any) -> float:
    """
    Validate a voting weight.
    
    Args:
        weight: Weight value to validate
        
    Returns:
        Normalized weight in range [0.0, 1.0]
    """
    try:
        val = float(weight)
        return max(0.0, min(1.0, val))
    except (TypeError, ValueError):
        return 0.0


def validate_weights(weights: Dict[str, Any]) -> Dict[str, float]:
    """
    Validate a dict of weights.
    
    Args:
        weights: Dict of expert_name -> weight
        
    Returns:
        Dict of validated weights
    """
    if not isinstance(weights, dict):
        return {}
    
    validated = {}
    for key, value in weights.items():
        if isinstance(key, str) and key:
            validated[key] = validate_weight(value)
    
    return validated
