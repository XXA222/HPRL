"""Runtime inference shield for stale, uncertain, or risk-invalid Hedge actions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .actions import DEFAULT_ACTION_CATALOG, HedgeActions
from .config import HedgeRLConfig


@dataclass(frozen=True, slots=True)
class InferenceDecision:
    requested_action: HedgeActions
    executed_action: HedgeActions
    confidence: float
    normalized_entropy: float
    shielded: bool
    reasons: tuple[str, ...]


class HedgeInferenceGuard:
    def __init__(self, config: HedgeRLConfig) -> None:
        self.config = config
        self.action_count = len(DEFAULT_ACTION_CATALOG)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        # Masked actions intentionally use a very negative sentinel. Their exponentials
        # are mathematically negligible and may underflow to 0.0. Freqtrade's test
        # suite raises on every NumPy floating-point warning, so suppress only this
        # expected underflow while leaving overflow/invalid/divide errors strict.
        with np.errstate(under="ignore"):
            exponent = np.exp(shifted)
        return exponent / max(float(exponent.sum()), np.finfo(float).tiny)

    def decide(
        self,
        logits: npt.ArrayLike,
        *,
        action_mask: npt.ArrayLike,
        feature_age_steps: int,
    ) -> InferenceDecision:
        values = np.asarray(logits, dtype=np.float64).reshape(-1)
        mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
        if values.shape != (self.action_count,) or mask.shape != (self.action_count,):
            raise ValueError(f"logits and action_mask must each contain {self.action_count} values")
        reasons: list[str] = []
        if not np.isfinite(values).all():
            values = np.nan_to_num(values, nan=-1e9, posinf=1e9, neginf=-1e9)
            reasons.append("NONFINITE_LOGITS")
        if not mask.any():
            mask = np.zeros_like(mask)
            mask[HedgeActions.HOLD] = True
            reasons.append("EMPTY_ACTION_MASK")
        requested = HedgeActions(int(np.argmax(values)))
        masked_logits = np.where(mask, values, -1e12)
        probabilities = self._softmax(masked_logits)
        selected = HedgeActions(int(np.argmax(probabilities)))
        confidence = float(probabilities[selected])
        positive = probabilities[probabilities > 0]
        entropy = -float(np.sum(positive * np.log(positive)))
        valid_count = int(mask.sum())
        normalized_entropy = entropy / math.log(valid_count) if valid_count > 1 else 0.0
        if not mask[requested]:
            reasons.append("REQUESTED_ACTION_MASKED")
        if feature_age_steps > self.config.max_feature_age_steps:
            reasons.append("STALE_FEATURES")
        if confidence < self.config.confidence_threshold:
            reasons.append("LOW_CONFIDENCE")
        executed = selected
        if any(
            reason in reasons
            for reason in ("STALE_FEATURES", "LOW_CONFIDENCE", "NONFINITE_LOGITS")
        ):
            executed = HedgeActions.HOLD
        return InferenceDecision(
            requested_action=requested,
            executed_action=executed,
            confidence=confidence,
            normalized_entropy=normalized_entropy,
            shielded=executed != requested or bool(reasons),
            reasons=tuple(dict.fromkeys(reasons)),
        )
