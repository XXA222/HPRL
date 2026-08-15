"""Gymnasium import boundary with a tiny test-only fallback.

Production RL training still requires the ``freqai_rl`` extra.  The fallback
keeps source validation and deterministic environment tests available in
minimal/read-only installations where Gymnasium is intentionally absent.
"""
# Conditional optional-dependency import boundary is intentional.
# ruff: noqa: I001

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised in full FreqAI-RL installations
    import gymnasium as gym
    from gymnasium import spaces

    GYMNASIUM_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback is exercised by package smoke tests
    GYMNASIUM_AVAILABLE = False

    class _Env:
        metadata: dict = {}

        def reset(self, *, seed=None, options=None):
            self.np_random = np.random.default_rng(seed)
            return None

        def close(self):
            return None

    class _Discrete:
        def __init__(self, n: int):
            if n < 1:
                raise ValueError("n must be positive")
            self.n = int(n)

        def contains(self, value) -> bool:
            try:
                item = int(value)
            except (TypeError, ValueError):
                return False
            return 0 <= item < self.n

        def sample(self, mask=None):
            valid = np.arange(self.n) if mask is None else np.flatnonzero(mask)
            if not len(valid):
                raise ValueError("no valid action")
            return int(np.random.default_rng().choice(valid))

    class _Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = np.dtype(dtype)

        def contains(self, value) -> bool:
            array = np.asarray(value)
            return array.shape == self.shape and np.isfinite(array).all()

    class _Spaces:
        Discrete = _Discrete
        Box = _Box

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()
