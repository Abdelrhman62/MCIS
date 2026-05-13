"""Deterministic seeding for reproducibility.

Sets the seed on every relevant RNG so that
    - DataLoader shuffling
    - PyTorch parameter init
    - dropout / NumPy ops in custom layers
    - scikit-multilearn iterative stratification (via global np/random)

are all deterministic for a given ``seed``.

Note on iterstrat 0.2.0 bug:
    ``IterativeStratification`` ignores its own ``random_state``. The fix is
    to seed both ``np.random`` and Python's ``random`` immediately before
    calling ``.split()``. This module's ``set_seed`` does that.
"""
from __future__ import annotations

import os
import random as _python_random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs.

    Args:
        seed: Integer seed.
        deterministic: If True, also enable PyTorch deterministic algorithms
            and disable cudnn benchmarking. Off-by-default for benchmarks
            because deterministic mode can be 10–30% slower on GPU.
    """
    _python_random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch is optional at import time (some Phase 0 scripts don't need it).
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            # MPS doesn't honor manual_seed everywhere yet, but we set what we can.
            torch.mps.manual_seed(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Strict mode for newer PyTorch — raises on non-deterministic ops.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, RuntimeError):
                pass
    except ImportError:
        pass


def seed_iterstrat(seed: int = 42) -> None:
    """Workaround for scikit-multilearn ``IterativeStratification`` bug.

    Call IMMEDIATELY before ``mskf.split()``. Construct the ``IterativeStratification``
    instance WITHOUT a ``random_state`` argument (sklearn ≥1.6 rejects it when
    ``shuffle=False``).
    """
    _python_random.seed(seed)
    np.random.seed(seed)
