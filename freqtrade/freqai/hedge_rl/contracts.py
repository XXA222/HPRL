"""Deterministic contracts for Hedge ML/RL configuration and action governance.

This module contains the development rounds 21-30.  The contracts are intentionally
small, serializable, and dependency-light so they can be used by training, inference,
source validation, and production adapters without importing the full Freqtrade runtime.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from enum import IntEnum
from hashlib import sha256
from typing import Any

import numpy as np
import numpy.typing as npt

from .actions import (
    DEFAULT_ACTION_CATALOG,
    HedgeActionCatalog,
    HedgeActions,
    HedgeActionSpec,
    LegCommand,
    Urgency,
)
from .config import HedgeRLConfig
from .constraints import HedgeActionMasker
from .state import HedgeAccountState


# Round 21: deterministic seed derivation -------------------------------------------------
@dataclass(frozen=True, slots=True)
class SeedLedger:
    """Derive stable, label-specific seeds without relying on Python's randomized hash."""

    root_seed: int
    namespace: str = "hedge-rl"

    def __post_init__(self) -> None:
        if not isinstance(self.root_seed, int) or self.root_seed < 0:
            raise ValueError("root_seed must be a non-negative integer")
        if not self.namespace.strip():
            raise ValueError("namespace cannot be empty")

    def child(self, label: str, *, index: int = 0) -> int:
        if not str(label).strip() or index < 0:
            raise ValueError("label must be non-empty and index non-negative")
        payload = f"{self.namespace}\0{self.root_seed}\0{label}\0{index}".encode()
        # NumPy and most RL libraries accept unsigned 32-bit seeds.
        return int.from_bytes(sha256(payload).digest()[:4], "big", signed=False)

    def snapshot(self, labels: tuple[str, ...]) -> dict[str, int]:
        if len(set(labels)) != len(labels):
            raise ValueError("seed labels must be unique")
        return {label: self.child(label) for label in labels}


# Round 22: versioned configuration contract ----------------------------------------------
_VERSION_RE = re.compile(r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$")


@dataclass(frozen=True, order=True, slots=True)
class ConfigSchemaVersion:
    major: int
    minor: int
    patch: int = 0

    @classmethod
    def parse(cls, value: str) -> ConfigSchemaVersion:
        match = _VERSION_RE.fullmatch(str(value).strip())
        if match is None:
            raise ValueError("schema version must use MAJOR.MINOR.PATCH")
        return cls(*(int(match.group(name)) for name in ("major", "minor", "patch")))

    def compatible_with(self, runtime: ConfigSchemaVersion) -> bool:
        """A runtime accepts the same major and a schema no newer than itself."""

        return self.major == runtime.major and self <= runtime

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


# Round 23: narrowly scoped environment overrides -----------------------------------------
def _parse_env_scalar(raw: str, expected: type[Any]) -> Any:
    text = raw.strip()
    if expected is bool:
        lowered = text.lower()
        if lowered not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"cannot parse Boolean environment value {raw!r}")
        return lowered in {"true", "1", "yes"}
    if expected is int:
        return int(text)
    if expected is float:
        value = float(text)
        if not np.isfinite(value):
            raise ValueError("environment override must be finite")
        return value
    if expected is tuple:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("tuple override must be a JSON array")
        return tuple(float(item) for item in parsed)
    if expected is str:
        return text
    raise TypeError(f"unsupported environment override type: {expected}")


def apply_environment_overrides(
    config: HedgeRLConfig,
    *,
    environ: Mapping[str, str] | None = None,
    prefix: str = "FREQTRADE_HEDGE_RL_",
) -> HedgeRLConfig:
    """Apply only known scalar config fields; unknown variables are ignored.

    Reward-weight objects are deliberately excluded from environment overrides to avoid
    silently changing a production reward definition with a misspelled variable.
    """

    source = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    allowed = {item.name: item for item in fields(config) if item.name != "reward_weights"}
    for name, descriptor in allowed.items():
        key = prefix + name.upper()
        if key not in source:
            continue
        current = getattr(config, name)
        expected = tuple if isinstance(current, tuple) else type(current)
        values[name] = _parse_env_scalar(source[key], expected)
    return replace(config, **values)


# Round 24: canonical action catalogue digest ---------------------------------------------
def canonical_action_payload(
    catalog: HedgeActionCatalog = DEFAULT_ACTION_CATALOG,
) -> list[dict[str, Any]]:
    return [
        {
            "id": int(spec.action),
            "name": spec.action.name,
            "long_command": spec.long_command.value,
            "short_command": spec.short_command.value,
            "long_fraction": format(spec.long_fraction, ".17g"),
            "short_fraction": format(spec.short_fraction, ".17g"),
            "urgency": spec.urgency.value,
        }
        for spec in catalog.specs()
    ]


def canonical_action_signature(catalog: HedgeActionCatalog = DEFAULT_ACTION_CATALOG) -> str:
    encoded = json.dumps(canonical_action_payload(catalog), sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode()).hexdigest()


# Round 25: explicit action risk tiers -----------------------------------------------------
class ActionRiskTier(IntEnum):
    HOLD = 0
    REDUCE = 1
    OPEN_SMALL = 2
    OPEN_MEDIUM = 3
    REBALANCE = 4
    EMERGENCY = 5


def classify_action_risk(spec: HedgeActionSpec) -> ActionRiskTier:
    if spec.action is HedgeActions.HOLD:
        return ActionRiskTier.HOLD
    if spec.action in {HedgeActions.CLOSE_BOTH, HedgeActions.EMERGENCY_REDUCE_BOTH}:
        return ActionRiskTier.EMERGENCY
    commands = {spec.long_command, spec.short_command}
    if commands <= {LegCommand.HOLD, LegCommand.REDUCE, LegCommand.CLOSE}:
        return ActionRiskTier.REDUCE
    if LegCommand.INCREASE in commands and LegCommand.REDUCE in commands:
        return ActionRiskTier.REBALANCE
    largest = max(spec.long_fraction, spec.short_fraction)
    return ActionRiskTier.OPEN_SMALL if largest <= 0.10 else ActionRiskTier.OPEN_MEDIUM


# Round 26: strict action serialization ----------------------------------------------------
def serialize_action_spec(spec: HedgeActionSpec) -> dict[str, Any]:
    return {
        "action": int(spec.action),
        "name": spec.action.name,
        "long_command": spec.long_command.value,
        "short_command": spec.short_command.value,
        "long_fraction": spec.long_fraction,
        "short_fraction": spec.short_fraction,
        "urgency": spec.urgency.value,
    }


def deserialize_action_spec(payload: Mapping[str, Any]) -> HedgeActionSpec:
    required = {
        "action",
        "name",
        "long_command",
        "short_command",
        "long_fraction",
        "short_fraction",
        "urgency",
    }
    if set(payload) != required:
        raise ValueError(f"action payload fields must be exactly {sorted(required)}")
    action = HedgeActions(int(payload["action"]))
    if payload["name"] != action.name:
        raise ValueError("action name does not match action id")
    return HedgeActionSpec(
        action=action,
        long_command=LegCommand(str(payload["long_command"])),
        short_command=LegCommand(str(payload["short_command"])),
        long_fraction=float(payload["long_fraction"]),
        short_fraction=float(payload["short_fraction"]),
        urgency=Urgency(str(payload["urgency"])),
    )


# Round 27: LONG/SHORT symmetry contract ---------------------------------------------------
_MIRROR_ACTIONS: dict[HedgeActions, HedgeActions] = {
    HedgeActions.HOLD: HedgeActions.HOLD,
    HedgeActions.LONG_OPEN_SMALL: HedgeActions.SHORT_OPEN_SMALL,
    HedgeActions.LONG_OPEN_MEDIUM: HedgeActions.SHORT_OPEN_MEDIUM,
    HedgeActions.LONG_ADD_SMALL: HedgeActions.SHORT_ADD_SMALL,
    HedgeActions.LONG_ADD_MEDIUM: HedgeActions.SHORT_ADD_MEDIUM,
    HedgeActions.LONG_REDUCE_SMALL: HedgeActions.SHORT_REDUCE_SMALL,
    HedgeActions.LONG_REDUCE_MEDIUM: HedgeActions.SHORT_REDUCE_MEDIUM,
    HedgeActions.LONG_CLOSE: HedgeActions.SHORT_CLOSE,
    HedgeActions.SHORT_OPEN_SMALL: HedgeActions.LONG_OPEN_SMALL,
    HedgeActions.SHORT_OPEN_MEDIUM: HedgeActions.LONG_OPEN_MEDIUM,
    HedgeActions.SHORT_ADD_SMALL: HedgeActions.LONG_ADD_SMALL,
    HedgeActions.SHORT_ADD_MEDIUM: HedgeActions.LONG_ADD_MEDIUM,
    HedgeActions.SHORT_REDUCE_SMALL: HedgeActions.LONG_REDUCE_SMALL,
    HedgeActions.SHORT_REDUCE_MEDIUM: HedgeActions.LONG_REDUCE_MEDIUM,
    HedgeActions.SHORT_CLOSE: HedgeActions.LONG_CLOSE,
    HedgeActions.BOTH_OPEN_SMALL: HedgeActions.BOTH_OPEN_SMALL,
    HedgeActions.BOTH_REDUCE_SMALL: HedgeActions.BOTH_REDUCE_SMALL,
    HedgeActions.REBALANCE_TO_LONG: HedgeActions.REBALANCE_TO_SHORT,
    HedgeActions.REBALANCE_TO_SHORT: HedgeActions.REBALANCE_TO_LONG,
    HedgeActions.CLOSE_BOTH: HedgeActions.CLOSE_BOTH,
    HedgeActions.EMERGENCY_REDUCE_BOTH: HedgeActions.EMERGENCY_REDUCE_BOTH,
}


def mirror_action(action: int | HedgeActions) -> HedgeActions:
    return _MIRROR_ACTIONS[HedgeActions(int(action))]


# Round 28: safe action downgrade ----------------------------------------------------------
_DOWNGRADE_CHAINS: dict[HedgeActions, tuple[HedgeActions, ...]] = {
    HedgeActions.LONG_OPEN_MEDIUM: (HedgeActions.LONG_OPEN_SMALL, HedgeActions.HOLD),
    HedgeActions.SHORT_OPEN_MEDIUM: (HedgeActions.SHORT_OPEN_SMALL, HedgeActions.HOLD),
    HedgeActions.LONG_ADD_MEDIUM: (HedgeActions.LONG_ADD_SMALL, HedgeActions.HOLD),
    HedgeActions.SHORT_ADD_MEDIUM: (HedgeActions.SHORT_ADD_SMALL, HedgeActions.HOLD),
    HedgeActions.REBALANCE_TO_LONG: (HedgeActions.BOTH_REDUCE_SMALL, HedgeActions.HOLD),
    HedgeActions.REBALANCE_TO_SHORT: (HedgeActions.BOTH_REDUCE_SMALL, HedgeActions.HOLD),
    HedgeActions.BOTH_OPEN_SMALL: (HedgeActions.HOLD,),
}


def safe_downgrade(action: int | HedgeActions, action_mask: npt.ArrayLike) -> HedgeActions:
    requested = HedgeActions(int(action))
    mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
    if mask.shape != (len(DEFAULT_ACTION_CATALOG),):
        raise ValueError("action mask has an incompatible shape")
    if mask[int(requested)]:
        return requested
    for candidate in _DOWNGRADE_CHAINS.get(requested, (HedgeActions.HOLD,)):
        if mask[int(candidate)]:
            return candidate
    if mask[int(HedgeActions.HOLD)]:
        return HedgeActions.HOLD
    valid = np.flatnonzero(mask)
    if not len(valid):
        raise ValueError("action mask contains no valid action")
    risk_sorted = sorted(
        (HedgeActions(int(item)) for item in valid),
        key=lambda item: (classify_action_risk(DEFAULT_ACTION_CATALOG.decode(item)), int(item)),
    )
    return risk_sorted[0]


# Round 29: invalid action audit journal ---------------------------------------------------
@dataclass(frozen=True, slots=True)
class InvalidActionAudit:
    tick: int
    requested: HedgeActions
    executed: HedgeActions
    reasons: tuple[str, ...]
    observation_signature: str

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("tick cannot be negative")
        if not self.reasons:
            raise ValueError("invalid-action audit requires at least one reason")
        if not re.fullmatch(r"[0-9a-f]{64}", self.observation_signature):
            raise ValueError("observation_signature must be a lowercase SHA-256 digest")


class InvalidActionJournal:
    def __init__(self, *, capacity: int = 10_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._records: list[InvalidActionAudit] = []

    @staticmethod
    def observation_signature(observation: npt.ArrayLike) -> str:
        array = np.asarray(observation, dtype=np.float32)
        return sha256(array.tobytes(order="C")).hexdigest()

    def append(self, record: InvalidActionAudit) -> None:
        self._records.append(record)
        if len(self._records) > self.capacity:
            del self._records[: len(self._records) - self.capacity]

    def records(self) -> tuple[InvalidActionAudit, ...]:
        return tuple(self._records)


# Round 30: full action-mask reason matrix -------------------------------------------------
@dataclass(frozen=True, slots=True)
class MaskReasonRow:
    action: HedgeActions
    allowed: bool
    reasons: tuple[str, ...]
    projected_gross: float
    projected_net: float


def build_mask_reason_matrix(
    config: HedgeRLConfig,
    *,
    account: HedgeAccountState,
    mark: float,
) -> tuple[MaskReasonRow, ...]:
    masker = HedgeActionMasker(config)
    rows: list[MaskReasonRow] = []
    for spec in DEFAULT_ACTION_CATALOG.specs():
        decision = masker.evaluate(int(spec.action), account=account, mark=mark)
        rows.append(
            MaskReasonRow(
                action=spec.action,
                allowed=decision.allowed,
                reasons=decision.reasons,
                projected_gross=decision.projected_gross_exposure,
                projected_net=decision.projected_net_exposure,
            )
        )
    if len(rows) != len(DEFAULT_ACTION_CATALOG) or not rows[int(HedgeActions.HOLD)].allowed:
        raise RuntimeError("action reason matrix violates the HOLD/action-count invariant")
    return tuple(rows)
