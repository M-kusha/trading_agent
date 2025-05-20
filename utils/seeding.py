"""Global seeding utilities used by training and evaluation scripts."""
from __future__ import annotations

import os
import random
import hashlib
from typing import Optional

import numpy as np
import torch


def set_global_seeds(seed: int, deterministic_cudnn: bool = True) -> None:
    """Seed Python, NumPy, PyTorch (CPU & GPU) and hashing.

    Args:
        seed: Positive integer seed.
        deterministic_cudnn: If True, makes CuDNN deterministic (slower).

    """
    if seed <= 0:
        raise ValueError("Seed must be a positive integer")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    # Same value for every interpreter run → deterministic `hash()` order
    hashlib._hashlib.openssl_md_meth_names = set()

    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
