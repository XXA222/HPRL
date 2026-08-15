"""Typed configuration for the R3.3 production control plane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from freqtrade.exceptions import OperationalException

_ALLOWED = frozenset({"enabled", "confirmation_ttl_seconds", "max_pending_confirmations"})


@dataclass(frozen=True, slots=True)
class HedgeControlPlaneConfig:
    enabled: bool = False
    confirmation_ttl_seconds: int = 120
    max_pending_confirmations: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise OperationalException("hedge.control_plane.enabled must be boolean")
        if (
            not isinstance(self.confirmation_ttl_seconds, int)
            or isinstance(self.confirmation_ttl_seconds, bool)
            or not 10 <= self.confirmation_ttl_seconds <= 3600
        ):
            raise OperationalException(
                "hedge.control_plane.confirmation_ttl_seconds must be in [10, 3600]"
            )
        if (
            not isinstance(self.max_pending_confirmations, int)
            or isinstance(self.max_pending_confirmations, bool)
            or not 1 <= self.max_pending_confirmations <= 100_000
        ):
            raise OperationalException(
                "hedge.control_plane.max_pending_confirmations must be in [1, 100000]"
            )


def hedge_control_plane_config_from_mapping(
    config: Mapping[str, Any],
) -> HedgeControlPlaneConfig:
    hedge = config.get("hedge", {})
    if not isinstance(hedge, Mapping):
        raise OperationalException("hedge must be a JSON object")
    raw = hedge.get("control_plane", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise OperationalException("hedge.control_plane must be a JSON object")
    unknown = sorted(set(raw) - _ALLOWED)
    if unknown:
        raise OperationalException(
            "Unknown hedge.control_plane configuration key(s): " + ", ".join(unknown)
        )
    return HedgeControlPlaneConfig(**dict(raw))
