from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

from .contracts import BacktestEvaluation, Candidate
from .decimal_utils import canonical_json, json_value, to_decimal


@dataclass(frozen=True, slots=True)
class CachedEvaluation:
    candidate: Candidate
    dataset_fingerprint: str
    metrics: Mapping[str, Decimal | int | bool | str]
    objective_score: Decimal
    feasible: bool
    violations: tuple[str, ...]
    elapsed_seconds: Decimal
    evaluated_at: datetime

    @classmethod
    def from_evaluation(cls, item: BacktestEvaluation) -> CachedEvaluation:
        return cls(
            candidate=item.candidate,
            dataset_fingerprint=item.dataset_fingerprint,
            metrics=item.metrics,
            objective_score=item.objective_score,
            feasible=item.feasible,
            violations=item.violations,
            elapsed_seconds=item.elapsed_seconds,
            evaluated_at=item.evaluated_at,
        )

    def to_evaluation(self) -> BacktestEvaluation:
        return BacktestEvaluation(
            candidate=self.candidate,
            dataset_fingerprint=self.dataset_fingerprint,
            result=None,
            metrics=self.metrics,
            objective_score=self.objective_score,
            feasible=self.feasible,
            violations=self.violations,
            elapsed_seconds=self.elapsed_seconds,
            evaluated_at=self.evaluated_at,
        )


class EvaluationCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(dataset_fingerprint: str, candidate: Candidate, engine_fingerprint: str) -> str:
        return sha256(
            canonical_json(
                {
                    "dataset": dataset_fingerprint,
                    "candidate": candidate.parameters,
                    "engine": engine_fingerprint,
                }
            )
        ).hexdigest()

    def path(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("cache key must be lowercase SHA-256 hex")
        return self.directory / f"{key}.json"

    def put(self, key: str, evaluation: BacktestEvaluation) -> Path:
        payload = {
            "schema_version": "hedge-bt-cache-v1",
            "candidate": {
                "candidate_id": evaluation.candidate.candidate_id,
                "parameters": evaluation.candidate.parameters,
                "ordinal": evaluation.candidate.ordinal,
            },
            "dataset_fingerprint": evaluation.dataset_fingerprint,
            "metrics": evaluation.metrics,
            "objective_score": evaluation.objective_score,
            "feasible": evaluation.feasible,
            "violations": evaluation.violations,
            "elapsed_seconds": evaluation.elapsed_seconds,
            "evaluated_at": evaluation.evaluated_at,
        }
        target = self.path(key)
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                json_value(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        temporary.replace(target)
        return target

    def get(self, key: str) -> CachedEvaluation | None:
        target = self.path(key)
        if not target.exists():
            return None
        with target.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("schema_version") != "hedge-bt-cache-v1":
            raise ValueError(f"unsupported cache schema in {target}")
        candidate_raw = raw["candidate"]
        candidate = Candidate(
            candidate_id=str(candidate_raw["candidate_id"]),
            parameters=candidate_raw["parameters"],
            ordinal=int(candidate_raw["ordinal"]),
        )
        metrics = {
            str(name): _restore_scalar(value) for name, value in raw["metrics"].items()
        }
        feasible = raw["feasible"]
        if not isinstance(feasible, bool):
            raise TypeError("cache feasible must be a boolean")
        return CachedEvaluation(
            candidate=candidate,
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            metrics=metrics,
            objective_score=to_decimal(raw["objective_score"]),
            feasible=feasible,
            violations=tuple(str(item) for item in raw.get("violations", ())),
            elapsed_seconds=to_decimal(raw.get("elapsed_seconds", "0")),
            evaluated_at=datetime.fromisoformat(raw["evaluated_at"]).astimezone(UTC),
        )


def _restore_scalar(value: object) -> Decimal | int | bool | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return str(value)
    try:
        return to_decimal(value)
    except ValueError:
        return value
