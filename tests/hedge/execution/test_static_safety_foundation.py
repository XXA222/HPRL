from pathlib import Path


def test_overlay_stays_inside_direction_five_exclusive_paths() -> None:
    package_root = Path(__file__).resolve().parents[3]
    overlay = package_root / "overlay"
    allowed_prefixes = (
        "freqtrade/hedge/execution/",
        "freqtrade/hedge/telemetry/",
    )
    allowed_files = {
        "freqtrade/rpc/api_server/hedge_schemas.py",
        "freqtrade/rpc/api_server/hedge_readonly.py",
        "freqtrade/rpc/api_server/hedge_ws.py",
        "freqtrade/rpc/api_server/hedge_auth.py",
    }
    delivered = {path.relative_to(overlay).as_posix() for path in overlay.rglob("*.py")}
    assert all(
        relative in allowed_files or relative.startswith(allowed_prefixes) for relative in delivered
    )


def test_overlay_contains_no_real_order_call_or_persistence_dependency() -> None:
    package_root = Path(__file__).resolve().parents[3]
    overlay = package_root / "overlay"
    source = "\n".join(path.read_text(encoding="utf-8") for path in overlay.rglob("*.py"))
    assert ".create_order(" not in source
    assert "freqtrade.persistence" not in source
    assert "import ccxt" not in source
    assert "import requests" not in source
    assert "import httpx" not in source
