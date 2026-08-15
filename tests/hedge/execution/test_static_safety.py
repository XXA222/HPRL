from __future__ import annotations

import ast
from pathlib import Path


def _production_files() -> tuple[Path, ...]:
    package_root = Path(__file__).resolve().parents[3]
    overlay = package_root / "overlay"
    root = overlay if overlay.is_dir() else package_root
    paths = [
        *(root / "freqtrade/hedge/contracts").glob("*.py"),
        *(root / "freqtrade/hedge/execution").glob("*.py"),
        *(root / "freqtrade/hedge/telemetry").glob("*.py"),
        root / "freqtrade/rpc/api_server/hedge_schemas.py",
        root / "freqtrade/rpc/api_server/hedge_readonly.py",
        root / "freqtrade/rpc/api_server/hedge_ws.py",
        root / "freqtrade/rpc/api_server/hedge_auth.py",
    ]
    files = tuple(sorted(path for path in paths if path.is_file()))
    assert files, "direction-five production source files were not found"
    assert any(path.name == "service.py" for path in files)
    assert any(path.name == "hedge_readonly.py" for path in files)
    return files


def test_overlay_stays_inside_direction_five_exclusive_paths() -> None:
    package_root = Path(__file__).resolve().parents[3]
    overlay = package_root / "overlay"
    if not overlay.is_dir():
        return
    allowed_prefixes = (
        "freqtrade/hedge/execution/",
        "freqtrade/hedge/telemetry/",
        "freqtrade/hedge/contracts/",
    )
    allowed_files = {
        "freqtrade/rpc/api_server/hedge_schemas.py",
        "freqtrade/rpc/api_server/hedge_readonly.py",
        "freqtrade/rpc/api_server/hedge_ws.py",
        "freqtrade/rpc/api_server/hedge_auth.py",
    }
    delivered = {
        path.relative_to(overlay).as_posix()
        for path in overlay.rglob("*.py")
    }
    assert delivered
    assert all(
        relative in allowed_files or relative.startswith(allowed_prefixes)
        for relative in delivered
    )


def test_source_contains_no_real_order_call_or_forbidden_dependency() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in _production_files()
    )
    # Direction five now deliberately composes durable SQL adapters.  The safety
    # invariant is that domain/execution code does not call exchange SDK/network
    # clients directly; persistence imports are allowed only as explicit adapter
    # composition, not as an exchange side effect.
    forbidden = (
        ".create_order(",
        "import ccxt",
        "import requests",
        "import httpx",
        "from requests",
        "from httpx",
    )
    assert not any(token in source for token in forbidden)


def test_every_production_file_parses_and_contains_no_assert_statement() -> None:
    for path in _production_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), path
