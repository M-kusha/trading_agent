#!/usr/bin/env python3
"""
Numeric Utilities for PPO Meta Module
======================================

Shared numeric helper functions used across the meta module.
Centralized to avoid duplication between ppo_types.py and arbiter_logic.py.
"""

from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Best-effort conversion to float with a safe default.
    
    Handles None, strings, and invalid values gracefully.
    
    Args:
        value: Any value to convert to float
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clip(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    Clamp value to [min_value, max_value].
    
    Args:
        value: Value to clamp
        min_value: Lower bound (default 0.0)
        max_value: Upper bound (default 1.0)
        
    Returns:
        Clamped value
    """
    return max(min_value, min(max_value, value))


# Aliases for backward compatibility with underscore-prefixed versions
_safe_float = safe_float
_clip = clip
