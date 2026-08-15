"""Post-process completed research outputs into canonical comparison metrics."""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any

_ALIASES: dict[str, tuple[str, ...]] = {
    "sharpe": ("sharpe", "sharpe_ratio"),
    "sortino": ("sortino", "sortino_ratio"),
    "drawdown": ("max_drawdown_account", "max_drawdown", "drawdown"),
    "profit": ("profit_total", "total_return_ratio", "net_return", "profit"),
    "profit_abs": ("profit_total_abs", "total_profit_abs", "profit_abs"),
    "trades": ("total_trades", "trades", "trade_count"),
    "win_rate": ("win_rate", "winrate", "win_ratio"),
    "reward": ("reward", "episode_reward", "mean_reward"),
    "loss": ("loss", "eval_loss", "validation_loss"),
    "accuracy": ("accuracy",),
    "f1": ("f1", "f1_score"),
    "mae": ("mae",),
    "rmse": ("rmse",),
}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _canonical_from_mapping(mapping: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias not in mapping:
                continue
            value = _number(mapping[alias])
            if value is not None:
                result[canonical] = value
                break
    if "win_rate" not in result:
        wins = _number(mapping.get("wins"))
        losses = _number(mapping.get("losses"))
        draws = _number(mapping.get("draws")) or 0.0
        if wins is not None and losses is not None and wins + losses + draws > 0:
            result["win_rate"] = wins / (wins + losses + draws)
    return result


def _append_strategy_candidates(
    rows: list[dict[str, Any]],
    strategies: object,
    *,
    strategy: str,
) -> None:
    if not isinstance(strategies, dict):
        return
    selected = strategies.get(strategy) if strategy else None
    if isinstance(selected, dict):
        rows.append(selected)
    for value in strategies.values():
        if isinstance(value, dict) and value not in rows:
            rows.append(value)


def _append_comparison_candidates(
    rows: list[dict[str, Any]],
    comparison: object,
    *,
    strategy: str,
) -> None:
    if not isinstance(comparison, list):
        return
    if strategy:
        selected = next(
            (
                value
                for value in comparison
                if isinstance(value, dict)
                and str(value.get("key", "")) == strategy
            ),
            None,
        )
        if selected is not None:
            rows.append(selected)
    rows.extend(
        value
        for value in comparison
        if isinstance(value, dict)
    )


def _append_named_candidates(
    rows: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    for key in ("metrics", "summary", "best", "result", "evaluation"):
        value = payload.get(key)
        if not isinstance(value, dict):
            continue
        nested_metrics = value.get("metrics")
        if isinstance(nested_metrics, dict):
            rows.append(nested_metrics)
        rows.append(value)


def _candidate_mappings(
    payload: Any,
    *,
    strategy: str = "",
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []

    rows: list[dict[str, Any]] = []
    report = payload.get("report")
    if isinstance(report, dict):
        rows.append(report)
    _append_strategy_candidates(
        rows,
        payload.get("strategy"),
        strategy=strategy,
    )
    _append_comparison_candidates(
        rows,
        payload.get("strategy_comparison"),
        strategy=strategy,
    )
    _append_named_candidates(rows, payload)
    rows.append(payload)
    return rows


def extract_metrics(payload: Any, *, strategy: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    for mapping in _candidate_mappings(payload, strategy=strategy):
        for key, value in _canonical_from_mapping(mapping).items():
            result.setdefault(key, value)
    return result


def _json_from_zip(path: Path, *, max_member_bytes: int) -> Any | None:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [
                info
                for info in archive.infolist()
                if info.filename.lower().endswith(".json") and info.file_size <= max_member_bytes
            ]
            members.sort(key=lambda item: ("config" in item.filename.lower(), item.filename))
            for info in members:
                try:
                    return json.loads(archive.read(info).decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
    except (OSError, zipfile.BadZipFile):
        return None
    return None


def extract_metrics_from_file(
    path: Path,
    *,
    strategy: str = "",
    max_json_bytes: int = 64 * 1024 * 1024,
) -> dict[str, float]:
    try:
        if path.suffix.lower() == ".json":
            if path.stat().st_size > max_json_bytes:
                return {}
            payload = json.loads(path.read_text(encoding="utf-8"))
            return extract_metrics(payload, strategy=strategy)
        if path.suffix.lower() == ".zip":
            payload = _json_from_zip(path, max_member_bytes=max_json_bytes)
            return {} if payload is None else extract_metrics(payload, strategy=strategy)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return {}


def extract_metrics_from_directory(path: Path, *, strategy: str = "") -> dict[str, float]:
    root = path.expanduser().resolve()
    if not root.is_dir():
        return {}
    result: dict[str, float] = {}
    candidates = sorted(
        (
            item
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in {".json", ".zip"}
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates[:100]:
        for key, value in extract_metrics_from_file(candidate, strategy=strategy).items():
            result.setdefault(key, value)
    return result
