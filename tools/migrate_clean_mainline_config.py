#!/usr/bin/env python3
"""One-time local config migration for the Clean Mainline operations key."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any

from freqtrade.hedge.config_migration import migrate_legacy_operations_alias


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a JSON object.")
    return payload


def migrate(path: Path, *, write: bool) -> tuple[bool, Path | None]:
    payload = load_json(path)
    hedge = payload.get("hedge")
    if not isinstance(hedge, dict):
        return False, None

    before = deepcopy(payload)
    changed = migrate_legacy_operations_alias(hedge)
    if not changed:
        return False, None

    if not write:
        return True, None

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(f"{path.name}.pre-clean-mainline-{stamp}.bak")
    shutil.copy2(path, backup)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if before == payload:
        raise AssertionError("migration reported a change but the payload is unchanged")
    return True, backup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the config after creating a timestamped backup.",
    )
    args = parser.parse_args()

    path = args.config.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    changed, backup = migrate(path, write=args.write)
    result = {
        "config": str(path),
        "migration_required": changed,
        "written": bool(changed and args.write),
        "backup": str(backup) if backup else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
