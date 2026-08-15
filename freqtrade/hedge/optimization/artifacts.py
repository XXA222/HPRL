"""Atomic, reproducible optimization result artifacts."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from freqtrade.hedge.optimization.fingerprint import json_safe
from freqtrade.hedge.optimization.types import OptimizationResult, TrialRecord


@dataclass(frozen=True, slots=True)
class OptimizationArtifacts:
    summary_json: Path
    trials_csv: Path
    pareto_json: Path
    best_parameters_json: Path | None
    manifest_json: Path


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def _trial_payload(trial: TrialRecord) -> dict[str, object]:
    return {
        "trial_id": trial.trial_id,
        "parameter_hash": trial.parameter_hash,
        "parameters": trial.parameters,
        "status": trial.status.value,
        "metrics": trial.metrics,
        "objective_values": trial.objective_values,
        "scalar_score": trial.scalar_score,
        "constraint_violations": trial.constraint_violations,
        "error": trial.error,
        "duration_seconds": trial.duration_seconds,
        "dataset_fingerprint": trial.dataset_fingerprint,
        "config_fingerprint": trial.config_fingerprint,
        "worker": trial.worker,
    }


def _csv_bytes(trials: Iterable[TrialRecord]) -> bytes:
    trials = tuple(trials)
    parameter_names = sorted({name for trial in trials for name in trial.parameters})
    metric_names = sorted({name for trial in trials for name in trial.metrics})
    fields = (
        ["trial_id", "status", "parameter_hash", "scalar_score", "duration_seconds", "error"]
        + [f"param:{name}" for name in parameter_names]
        + [f"metric:{name}" for name in metric_names]
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for trial in trials:
        row: dict[str, object] = {
            "trial_id": trial.trial_id,
            "status": trial.status.value,
            "parameter_hash": trial.parameter_hash,
            "scalar_score": "" if trial.scalar_score is None else str(trial.scalar_score),
            "duration_seconds": str(trial.duration_seconds),
            "error": trial.error or "",
        }
        row.update({f"param:{name}": trial.parameters.get(name, "") for name in parameter_names})
        row.update({f"metric:{name}": trial.metrics.get(name, "") for name in metric_names})
        writer.writerow({key: str(value) for key, value in row.items()})
    return stream.getvalue().encode("utf-8")


def export_optimization_result(
    result: OptimizationResult,
    output_directory: Path | str,
) -> OptimizationArtifacts:
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "optimization-summary.json"
    csv_path = output / "optimization-trials.csv"
    pareto_path = output / "optimization-pareto.json"
    best_path = output / "optimization-best-parameters.json"
    manifest_path = output / "optimization-artifact-manifest.json"

    summary = {
        "schema_version": "hedge-optimization-result-v1",
        "study_name": result.study_name,
        "study_fingerprint": result.study_fingerprint,
        "dataset_fingerprint": result.dataset_fingerprint,
        "resumed_trials": result.resumed_trials,
        "best_trial_id": result.best_trial_id,
        "pareto_trial_ids": result.pareto_trial_ids,
        "objectives": result.objective_specs,
        "trial_count": len(result.trials),
        "trials": tuple(_trial_payload(item) for item in result.trials),
    }
    _atomic_bytes(summary_path, _json_bytes(summary))
    _atomic_bytes(csv_path, _csv_bytes(result.trials))
    pareto_trials = tuple(
        _trial_payload(item) for item in result.trials if item.trial_id in result.pareto_trial_ids
    )
    _atomic_bytes(
        pareto_path,
        _json_bytes(
            {
                "schema_version": "hedge-optimization-pareto-v1",
                "study_fingerprint": result.study_fingerprint,
                "trials": pareto_trials,
            }
        ),
    )

    best = next(
        (item for item in result.trials if item.trial_id == result.best_trial_id),
        None,
    )
    if best is None:
        if best_path.exists():
            best_path.unlink()
        final_best: Path | None = None
    else:
        _atomic_bytes(
            best_path,
            _json_bytes(
                {
                    "schema_version": "hedge-optimization-best-parameters-v1",
                    "study_fingerprint": result.study_fingerprint,
                    "dataset_fingerprint": result.dataset_fingerprint,
                    "trial_id": best.trial_id,
                    "parameter_hash": best.parameter_hash,
                    "parameters": best.parameters,
                    "metrics": best.metrics,
                    "objective_values": best.objective_values,
                    "scalar_score": best.scalar_score,
                }
            ),
        )
        final_best = best_path

    artifact_paths = [summary_path, csv_path, pareto_path]
    if final_best is not None:
        artifact_paths.append(final_best)
    manifest = {
        "schema_version": "hedge-optimization-artifact-manifest-v1",
        "study_fingerprint": result.study_fingerprint,
        "files": {
            path.name: {
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "size": path.stat().st_size,
            }
            for path in artifact_paths
        },
    }
    _atomic_bytes(manifest_path, _json_bytes(manifest))
    return OptimizationArtifacts(
        summary_json=summary_path,
        trials_csv=csv_path,
        pareto_json=pareto_path,
        best_parameters_json=final_best,
        manifest_json=manifest_path,
    )
