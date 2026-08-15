"""Online feature normalization with serializable Welford statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class RunningNormalizer:
    size: int
    clip: float = 10.0
    epsilon: float = 1e-8
    count: int = field(init=False, default=0)
    mean: FloatArray = field(init=False)
    m2: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError("size must be positive")
        if self.clip <= 0 or self.epsilon <= 0:
            raise ValueError("clip and epsilon must be positive")
        self.count = 0
        self.mean = np.zeros(self.size, dtype=np.float64)
        self.m2 = np.zeros(self.size, dtype=np.float64)

    def update(self, values: npt.ArrayLike) -> None:
        array = self._coerce(values)
        for row in array.reshape(-1, self.size):
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.m2 += delta * delta2

    @property
    def variance(self) -> FloatArray:
        if self.count < 2:
            return np.ones(self.size, dtype=np.float64)
        return np.maximum(self.m2 / (self.count - 1), self.epsilon)

    def normalize(self, values: npt.ArrayLike, *, update: bool = False) -> npt.NDArray[np.float32]:
        array = self._coerce(values)
        if update:
            self.update(array)
        normalized = (array - self.mean) / np.sqrt(self.variance + self.epsilon)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)

    def _coerce(self, values: npt.ArrayLike) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        if array.shape == (self.size,):
            pass
        elif array.ndim >= 1 and array.shape[-1] == self.size:
            pass
        else:
            raise ValueError(f"expected final dimension {self.size}, got {array.shape}")
        if not np.isfinite(array).all():
            array = np.nan_to_num(array, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return array

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "clip": self.clip,
            "epsilon": self.epsilon,
            "count": self.count,
            "mean": self.mean.tolist(),
            "m2": self.m2.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> RunningNormalizer:
        normalizer = cls(
            size=int(state["size"]),
            clip=float(state["clip"]),
            epsilon=float(state["epsilon"]),
        )
        normalizer.count = int(state["count"])
        normalizer.mean = np.asarray(state["mean"], dtype=np.float64)
        normalizer.m2 = np.asarray(state["m2"], dtype=np.float64)
        if normalizer.mean.shape != (normalizer.size,) or normalizer.m2.shape != (normalizer.size,):
            raise ValueError("normalizer state shape mismatch")
        return normalizer
