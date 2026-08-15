from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SearchCheckpoint:
    run_id: str
    dataset_fingerprint: str
    engine_fingerprint: str
    completed_candidate_ids: tuple[str, ...]
    updated_at: datetime


class CheckpointStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def save(
        self,
        *,
        run_id: str,
        dataset_fingerprint: str,
        engine_fingerprint: str,
        completed_candidate_ids: Iterable[str],
    ) -> None:
        payload = {
            "schema_version": "hedge-bt-checkpoint-v1",
            "run_id": run_id,
            "dataset_fingerprint": dataset_fingerprint,
            "engine_fingerprint": engine_fingerprint,
            "completed_candidate_ids": sorted(set(completed_candidate_ids)),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def load(self) -> SearchCheckpoint | None:
        if not self.path.exists():
            return None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "hedge-bt-checkpoint-v1":
            raise ValueError("unsupported checkpoint schema")
        return SearchCheckpoint(
            run_id=str(raw["run_id"]),
            dataset_fingerprint=str(raw["dataset_fingerprint"]),
            engine_fingerprint=str(raw["engine_fingerprint"]),
            completed_candidate_ids=tuple(str(item) for item in raw["completed_candidate_ids"]),
            updated_at=datetime.fromisoformat(raw["updated_at"]).astimezone(UTC),
        )
