"""Immutable tensor-market dataset and offline transition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .contracts import OfflineTransition
from .device import require_torch, torch_device
from .errors import HPRLShapeError


@dataclass(frozen=True, slots=True)
class TensorMarketDataset:
    """Market tensors with strict no-lookahead separation.

    ``features[t]`` is observable at decision time ``t`` while ``forward_returns[t]`` is
    only consumed
    after the action for ``t`` has been chosen.
    """

    features: object
    forward_returns: object
    funding_rates: object | None = None
    available_notional: object | None = None
    symbols: tuple[str, ...] = ()

    def validate(self) -> "TensorMarketDataset":
        torch = require_torch()
        tensors_required = (self.features, self.forward_returns)
        if any(not torch.is_tensor(value) for value in tensors_required):
            raise HPRLShapeError("features and forward_returns must be torch tensors")
        optional_tensors = (self.funding_rates, self.available_notional)
        if any(value is not None and not torch.is_tensor(value) for value in optional_tensors):
            raise HPRLShapeError("optional market data must be torch tensors")
        if self.features.ndim != 3:
            raise HPRLShapeError("features must have [time, symbol, feature] shape")
        if self.forward_returns.ndim != 2:
            raise HPRLShapeError("forward_returns must have [time, symbol] shape")
        if self.features.shape[:2] != self.forward_returns.shape:
            raise HPRLShapeError("features and forward_returns time/symbol dimensions differ")
        if self.features.shape[0] < 2:
            raise HPRLShapeError("HPRL market dataset needs at least two time steps")
        if self.features.shape[1] < 1 or self.features.shape[2] < 1:
            raise HPRLShapeError("market dataset needs at least one symbol and one feature")
        if (
            self.funding_rates is not None
            and self.funding_rates.shape != self.forward_returns.shape
        ):
            raise HPRLShapeError("funding_rates shape must match forward_returns")
        if (
            self.available_notional is not None
            and self.available_notional.shape != self.forward_returns.shape
        ):
            raise HPRLShapeError("available_notional shape must match forward_returns")
        if self.symbols and len(self.symbols) != self.features.shape[1]:
            raise HPRLShapeError("symbol count does not match market tensor")
        if self.symbols and (
            any(not isinstance(symbol, str) or not symbol.strip() for symbol in self.symbols)
            or len(set(self.symbols)) != len(self.symbols)
        ):
            raise HPRLShapeError("symbols must be unique, non-empty strings")
        tensors = [self.features, self.forward_returns]
        optional = (self.funding_rates, self.available_notional)
        tensors.extend(value for value in optional if value is not None)
        devices = {str(value.device) for value in tensors}
        if len(devices) != 1:
            raise HPRLShapeError("all market tensors must reside on the same device")
        if any(not torch.isfinite(value).all() for value in tensors):
            raise HPRLShapeError("market tensors must contain only finite values")
        if self.available_notional is not None and (self.available_notional <= 0).any():
            raise HPRLShapeError("available_notional must be positive")
        return self

    def to(self, device: str) -> "TensorMarketDataset":
        torch = require_torch()
        target = torch_device(device)
        return TensorMarketDataset(
            features=self.features.to(device=target, dtype=torch.float32),
            forward_returns=self.forward_returns.to(device=target, dtype=torch.float32),
            funding_rates=(
                None
                if self.funding_rates is None
                else self.funding_rates.to(device=target, dtype=torch.float32)
            ),
            available_notional=(
                None
                if self.available_notional is None
                else self.available_notional.to(device=target, dtype=torch.float32)
            ),
            symbols=self.symbols,
        ).validate()


def _strict_json_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("done must be a JSON boolean")
    return value


class OfflineTransitionDataset:
    """Simple, deterministic offline dataset bridge independent of legacy FreqAI RL."""

    def __init__(
        self,
        transitions: Iterable[OfflineTransition],
        *,
        action_unit: str = "policy_code",
    ) -> None:
        values = tuple(transitions)
        if not values:
            raise ValueError("offline dataset cannot be empty")
        obs = len(values[0].observation)
        act = len(values[0].action)
        if any(len(row.observation) != obs or len(row.next_observation) != obs for row in values):
            raise HPRLShapeError("offline observation dimensions are inconsistent")
        if any(len(row.action) != act for row in values):
            raise HPRLShapeError("offline action dimensions are inconsistent")
        unit = action_unit.strip().lower() if isinstance(action_unit, str) else ""
        if unit not in {"policy_code", "margin_budget", "notional_exposure"}:
            raise ValueError(
                "action_unit must be policy_code/margin_budget/notional_exposure"
            )
        self._transitions = values
        self.observation_dim = obs
        self.action_dim = act
        self.action_unit = unit

    def __len__(self) -> int:
        return len(self._transitions)

    def tensors(
        self,
        device: str = "cpu",
        *,
        chunk_rows: int = 8192,
    ) -> dict[str, object]:
        """Materialize transition tensors with bounded temporary memory.

        The old implementation constructed five full nested Python lists before
        asking Torch to convert them.  Large offline datasets could therefore hold
        the dataclass rows, multiple Python list graphs and final tensors at once.
        This version preallocates final tensors and converts one bounded chunk at a
        time.
        """
        if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
            raise ValueError("chunk_rows must be a positive integer")
        torch = require_torch()
        target = torch_device(device)
        rows = self._transitions
        count = len(rows)
        result = {
            "obs": torch.empty((count, self.observation_dim), dtype=torch.float32, device=target),
            "action": torch.empty((count, self.action_dim), dtype=torch.float32, device=target),
            "reward": torch.empty((count, 1), dtype=torch.float32, device=target),
            "next_obs": torch.empty(
                (count, self.observation_dim), dtype=torch.float32, device=target
            ),
            "done": torch.empty((count, 1), dtype=torch.float32, device=target),
        }
        for start in range(0, count, chunk_rows):
            end = min(start + chunk_rows, count)
            chunk = rows[start:end]
            # Build only one bounded set of Python lists on CPU.  Copying to CUDA
            # happens chunk-wise, so peak host temporary memory is O(chunk_rows).
            chunk_values = {
                "obs": torch.tensor(
                    [row.observation for row in chunk], dtype=torch.float32
                ),
                "action": torch.tensor(
                    [row.action for row in chunk], dtype=torch.float32
                ),
                "reward": torch.tensor(
                    [[row.reward] for row in chunk], dtype=torch.float32
                ),
                "next_obs": torch.tensor(
                    [row.next_observation for row in chunk], dtype=torch.float32
                ),
                "done": torch.tensor(
                    [[row.done] for row in chunk], dtype=torch.float32
                ),
            }
            for key, value in chunk_values.items():
                # copy_ supports CPU->CUDA directly.  Avoid ``value.to(target)``
                # because it creates a second temporary CUDA tensor before copying
                # into the preallocated final storage.
                result[key][start:end].copy_(value, non_blocking=False)
        return result

    def release_source(self) -> None:
        """Release Python transition objects after one-shot tensorization."""
        self._transitions = ()

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        action_unit: str = "policy_code",
    ) -> "OfflineTransitionDataset":
        import json

        transitions: list[OfflineTransition] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    transitions.append(
                        OfflineTransition(
                            observation=tuple(float(x) for x in row["observation"]),
                            action=tuple(float(x) for x in row["action"]),
                            reward=float(row["reward"]),
                            next_observation=tuple(float(x) for x in row["next_observation"]),
                            done=_strict_json_bool(row["done"]),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid offline transition at JSONL line {line_no}") from exc
        return cls(transitions, action_unit=action_unit)
