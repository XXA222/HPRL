"""Independent high-performance reinforcement learning subsystem for Hedge.

HPRL is parallel to the existing FreqAI reinforcement-learning stacks. Importing this package is
side-effect free and does not import torch until a tensor or algorithm component is requested.
"""

from __future__ import annotations

HPRL_API_VERSION = "2.5.2"
HPRL_RELEASE = "clean-mainline-v1.2.1-gpu-adapt2-memory-v1.5-action-reward-v1.6-perf-v2.5.2"
HPRL_DEVICE_POLICY = "cpu-cuda-auto"

__all__ = ["HPRL_API_VERSION", "HPRL_DEVICE_POLICY", "HPRL_RELEASE"]
