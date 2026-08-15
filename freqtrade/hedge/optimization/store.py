"""SQLite-backed resumable study storage with deterministic trial identity."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from contextlib import closing
from decimal import Decimal
from pathlib import Path

from freqtrade.hedge.optimization.fingerprint import json_safe
from freqtrade.hedge.optimization.types import TrialRecord, TrialStatus


_SCHEMA = """
CREATE TABLE IF NOT EXISTS hedge_optimization_studies (
    study_name TEXT PRIMARY KEY,
    study_fingerprint TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS hedge_optimization_trials (
    study_name TEXT NOT NULL,
    trial_id INTEGER NOT NULL,
    parameter_hash TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    status TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    objective_values_json TEXT NOT NULL,
    scalar_score TEXT,
    constraint_violations_json TEXT NOT NULL,
    error TEXT,
    duration_seconds TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    config_fingerprint TEXT NOT NULL,
    worker TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (study_name, trial_id),
    UNIQUE (study_name, parameter_hash),
    FOREIGN KEY (study_name) REFERENCES hedge_optimization_studies(study_name)
);
CREATE INDEX IF NOT EXISTS ix_hedge_optimization_trial_status
    ON hedge_optimization_trials(study_name, status);
"""


def _dump(value: object) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _decimal_map(raw: str) -> dict[str, Decimal]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("stored metric payload is not an object")
    return {str(key): Decimal(str(value)) for key, value in data.items()}


class StudyStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            with connection:
                connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def initialize_study(
        self,
        *,
        study_name: str,
        study_fingerprint: str,
        dataset_fingerprint: str,
        definition: Mapping[str, object],
    ) -> None:
        if not study_name.strip():
            raise ValueError("study name cannot be empty")
        with closing(self._connect()) as connection:
            with connection:
                row = connection.execute(
                    "SELECT study_fingerprint, dataset_fingerprint FROM hedge_optimization_studies "
                    "WHERE study_name=?",
                    (study_name,),
                ).fetchone()
                if row is not None:
                    if row[0] != study_fingerprint or row[1] != dataset_fingerprint:
                        raise ValueError(
                            "existing study name has a different definition or dataset fingerprint"
                        )
                    connection.execute(
                        "UPDATE hedge_optimization_studies SET updated_at=CURRENT_TIMESTAMP "
                        "WHERE study_name=?",
                        (study_name,),
                    )
                    return
                connection.execute(
                    "INSERT INTO hedge_optimization_studies "
                    "(study_name, study_fingerprint, dataset_fingerprint, definition_json) "
                    "VALUES (?, ?, ?, ?)",
                    (study_name, study_fingerprint, dataset_fingerprint, _dump(definition)),
                )

    def save_trial(self, study_name: str, trial: TrialRecord) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO hedge_optimization_trials (
                        study_name, trial_id, parameter_hash, parameters_json, status,
                        metrics_json, objective_values_json, scalar_score,
                        constraint_violations_json, error, duration_seconds,
                        dataset_fingerprint, config_fingerprint, worker
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(study_name, parameter_hash) DO UPDATE SET
                        trial_id=excluded.trial_id,
                        parameters_json=excluded.parameters_json,
                        status=excluded.status,
                        metrics_json=excluded.metrics_json,
                        objective_values_json=excluded.objective_values_json,
                        scalar_score=excluded.scalar_score,
                        constraint_violations_json=excluded.constraint_violations_json,
                        error=excluded.error,
                        duration_seconds=excluded.duration_seconds,
                        dataset_fingerprint=excluded.dataset_fingerprint,
                        config_fingerprint=excluded.config_fingerprint,
                        worker=excluded.worker,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (
                        study_name,
                        trial.trial_id,
                        trial.parameter_hash,
                        _dump(trial.parameters),
                        trial.status.value,
                        _dump(trial.metrics),
                        _dump(trial.objective_values),
                        None if trial.scalar_score is None else str(trial.scalar_score),
                        _dump(trial.constraint_violations),
                        trial.error,
                        str(trial.duration_seconds),
                        trial.dataset_fingerprint,
                        trial.config_fingerprint,
                        trial.worker,
                    ),
                )

    def load_trials(self, study_name: str) -> tuple[TrialRecord, ...]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT trial_id, parameter_hash, parameters_json, status, metrics_json,
                       objective_values_json, scalar_score, constraint_violations_json,
                       error, duration_seconds, dataset_fingerprint, config_fingerprint, worker
                FROM hedge_optimization_trials
                WHERE study_name=? ORDER BY trial_id
                """,
                (study_name,),
            ).fetchall()
        output: list[TrialRecord] = []
        for row in rows:
            parameters = json.loads(row[2])
            objective_values = tuple(Decimal(str(value)) for value in json.loads(row[5]))
            violations = tuple(str(value) for value in json.loads(row[7]))
            output.append(
                TrialRecord(
                    trial_id=int(row[0]),
                    parameter_hash=str(row[1]),
                    parameters=parameters,
                    status=TrialStatus(str(row[3])),
                    metrics=_decimal_map(row[4]),
                    objective_values=objective_values,
                    scalar_score=None if row[6] is None else Decimal(str(row[6])),
                    constraint_violations=violations,
                    error=None if row[8] is None else str(row[8]),
                    duration_seconds=Decimal(str(row[9])),
                    dataset_fingerprint=str(row[10]),
                    config_fingerprint=str(row[11]),
                    worker=str(row[12]),
                )
            )
        return tuple(output)

    def completed_by_parameter_hash(self, study_name: str) -> dict[str, TrialRecord]:
        terminal = {
            TrialStatus.COMPLETE,
            TrialStatus.INFEASIBLE,
            TrialStatus.PRUNED,
        }
        return {
            item.parameter_hash: item
            for item in self.load_trials(study_name)
            if item.status in terminal
        }

    def save_many(self, study_name: str, trials: Iterable[TrialRecord]) -> None:
        for trial in trials:
            self.save_trial(study_name, trial)
