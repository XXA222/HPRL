"""Canonical serialization and fingerprints for reproducible optimization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from pathlib import Path


def json_safe(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [json_safe(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
        return items
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite floats cannot be fingerprinted")
        return str(Decimal(str(value)))
    return value


def canonical_json(value: object) -> bytes:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def fingerprint(value: object) -> str:
    return sha256(canonical_json(value)).hexdigest()


def parameter_fingerprint(parameters: Mapping[str, object]) -> str:
    return fingerprint({"schema": "hedge-parameter-v1", "parameters": parameters})


def study_fingerprint(
    *,
    parameter_specs: Sequence[object],
    objective_specs: Sequence[object],
    constraint_specs: Sequence[object],
    dataset_fingerprint: str,
    seed: int,
    sampler: str,
    extra_definition: object = None,
) -> str:
    return fingerprint(
        {
            "schema": "hedge-optimization-study-v1",
            "parameters": tuple(parameter_specs),
            "objectives": tuple(objective_specs),
            "constraints": tuple(constraint_specs),
            "dataset_fingerprint": dataset_fingerprint,
            "seed": seed,
            "sampler": sampler,
            "extra_definition": extra_definition,
        }
    )
