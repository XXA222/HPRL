from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_verifier():
    verifier_path = Path(__file__).with_name("_readonly_surface_audit.py")
    spec = importlib.util.spec_from_file_location(
        "direction2_readonly_verifier",
        verifier_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load readonly verifier: {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_surface_has_no_persistence_or_order_write_dependency():
    project_root = Path(__file__).resolve().parents[3]
    hedge_root = project_root / "overlay" / "freqtrade" / "hedge"
    if not hedge_root.exists():
        hedge_root = project_root / "freqtrade" / "hedge"
    verifier = _load_verifier()
    assert verifier.verify(hedge_root) == []


def test_runtime_surface_check_does_not_scan_unrelated_persistence(tmp_path):
    hedge_root = tmp_path / "freqtrade" / "hedge"
    (hedge_root / "exchange").mkdir(parents=True)
    (hedge_root / "readonly").mkdir()
    (hedge_root / "persistence.py").write_text(
        "import sqlalchemy\n",
        encoding="utf-8",
    )
    (hedge_root / "exchange" / "base.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (hedge_root / "readonly" / "service.py").write_text(
        "from ..exchange.base import VALUE\n",
        encoding="utf-8",
    )
    verifier = _load_verifier()
    assert verifier.verify(hedge_root) == []


def test_runtime_surface_check_detects_transitive_persistence_import(tmp_path):
    hedge_root = tmp_path / "freqtrade" / "hedge"
    (hedge_root / "exchange").mkdir(parents=True)
    (hedge_root / "readonly").mkdir()
    (hedge_root / "bridge.py").write_text(
        "from freqtrade.persistence import Trade\n",
        encoding="utf-8",
    )
    (hedge_root / "exchange" / "base.py").write_text(
        "from ..bridge import Trade\n",
        encoding="utf-8",
    )
    verifier = _load_verifier()
    findings = verifier.verify(hedge_root)
    assert any(
        "Forbidden persistence import" in finding
        for finding in findings
    )


def test_runtime_surface_check_resolves_package_relative_imports(tmp_path):
    hedge_root = tmp_path / "freqtrade" / "hedge"
    exchange_root = hedge_root / "exchange"
    readonly_root = hedge_root / "readonly"
    exchange_root.mkdir(parents=True)
    readonly_root.mkdir()
    (exchange_root / "__init__.py").write_text(
        "from .bridge import VALUE\n",
        encoding="utf-8",
    )
    (exchange_root / "bridge.py").write_text(
        "from freqtrade.persistence import Trade\nVALUE = Trade\n",
        encoding="utf-8",
    )
    verifier = _load_verifier()
    findings = verifier.verify(hedge_root)
    assert any(
        "Forbidden persistence import" in finding
        for finding in findings
    )

def test_runtime_surface_allows_readonly_query_order_endpoint(tmp_path):
    hedge_root = tmp_path / "freqtrade" / "hedge"
    (hedge_root / "exchange").mkdir(parents=True)
    (hedge_root / "readonly").mkdir()
    (hedge_root / "exchange" / "client.py").write_text(
        "async def probe(transport):\n"
        "    return await transport.request('GET', '/fapi/v1/order')\n",
        encoding="utf-8",
    )
    verifier = _load_verifier()
    assert verifier.verify(hedge_root) == []


def test_runtime_surface_still_rejects_order_write_request(tmp_path):
    hedge_root = tmp_path / "freqtrade" / "hedge"
    (hedge_root / "exchange").mkdir(parents=True)
    (hedge_root / "readonly").mkdir()
    (hedge_root / "exchange" / "client.py").write_text(
        "async def probe(transport):\n"
        "    return await transport.request('POST', '/fapi/v1/order')\n",
        encoding="utf-8",
    )
    verifier = _load_verifier()
    findings = verifier.verify(hedge_root)
    assert any("Forbidden exchange write request" in finding for finding in findings)

