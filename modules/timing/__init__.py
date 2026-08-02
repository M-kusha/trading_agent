"""
Timing Features Module
======================

Pure utility module for computing entry timing features.
No SmartInfoBus dependencies - can be used in both training and live.

Usage:
    from modules.timing.timing_features import compute_timing_features, TimingFeatures
"""

from .timing_features import (
    TIMING_FEATURE_DIM,
    TimingConfig,
    TimingFeatures,
    compute_timing_features,
    timing_features_to_array,
)

__all__ = [
    "TIMING_FEATURE_DIM",
    "TimingConfig",
    "TimingFeatures",
    "compute_timing_features",
    "timing_features_to_array",
]
