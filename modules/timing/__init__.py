"""
Timing Features Module
======================

Pure utility module for computing entry timing features.
No SmartInfoBus dependencies - can be used in both training and live.

Usage:
    from modules.timing.timing_features import compute_timing_features, TimingFeatures
"""

from .timing_features import (
    TimingFeatures,
    TimingConfig,
    compute_timing_features,
    timing_features_to_array,
    TIMING_FEATURE_DIM,
)

__all__ = [
    "TimingFeatures",
    "TimingConfig",
    "compute_timing_features",
    "timing_features_to_array",
    "TIMING_FEATURE_DIM",
]
