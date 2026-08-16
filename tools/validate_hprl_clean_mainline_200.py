"""Two-hundred-point acceptance matrix for HPRL on Clean Mainline V1.2.1.

The full 200-point mode is deliberately non-duplicative at the reporting layer:
138 executable HPRL pytest cases, 41 byte-exact protected legacy-RL files, and 21
Clean-Mainline GPU/integration/governance checks. A dependency-light ``--source-only``
mode executes the 41 protected-file plus 21 governance checks when pytest is intentionally
absent from a runtime environment. Source-only mode reports 62/62 and never pretends to be
a full 200/200 test run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_VERSION = "freqtrade-hedge-clean-mainline-v1.2.1-20260812"
EXPECTED_TESTS = 138
EXPECTED_PROTECTED = 41
EXPECTED_INTEGRATION = 21
EXPECTED_TOTAL = 200
_BASELINE_HASHES = {
    "pyproject.toml": (
        "77d81c56bafdaf9cfda60633d91bb738295f045362d793790b630f989e3eb375"
    ),
    "requirements-freqai-rl.txt": (
        "022a36227a45d10447fdad7b5655666f5197e32a1f1351df121c31bb5a14d933"
    ),
    "requirements-hedge-mlrl.txt": (
        "07ce57a0f1f0d940c8454d3e6401b3790c1baf28180e7eb8ddfafe47e31bd924"
    ),
    "CLEAN-MAINLINE-VERSION.txt": (
        "66093a1c05d676a376ff3141af4f4049f594bdf145d42a6c85203f6831371f37"
    ),
}
_PROTECTED_PACKED = (
    "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"
    "RRNJN7/7938Auxhtog=="
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_hashes() -> dict[str, str]:
    raw = zlib.decompress(base64.b64decode(_PROTECTED_PACKED)).decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError("invalid embedded protected-RL hash map")
    return {str(key): str(value) for key, value in payload.items()}


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _pytest_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("pytest") is not None
    except Exception:
        return False


def _collect_hprl_tests() -> tuple[list[str], set[str], str, dict[str, object]]:
    diagnostics: dict[str, object] = {
        "pytest_available": _pytest_available(),
        "collect_returncode": None,
        "execute_returncode": None,
        "collected": 0,
        "selected": 0,
        "recognition_mode": "not-run",
    }
    if not diagnostics["pytest_available"]:
        return [], set(), "pytest is not installed in the selected Python runtime", diagnostics

    collect = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "--confcutdir=tests/hedge/hprl",
            "tests/hedge/hprl",
        ]
    )
    diagnostics["collect_returncode"] = collect.returncode
    nodeids = [
        line.strip()
        for line in collect.stdout.splitlines()
        if line.startswith("tests/hedge/hprl/") and "::" in line
    ]
    diagnostics["collected"] = len(nodeids)
    selected = nodeids[:EXPECTED_TESTS]
    diagnostics["selected"] = len(selected)

    if collect.returncode != 0 or len(selected) != EXPECTED_TESTS:
        detail = (collect.stdout + collect.stderr)[-4000:]
        return nodeids, set(), detail, diagnostics

    execute = _run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-vv",
            "--color=no",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "--confcutdir=tests/hedge/hprl",
            *selected,
        ]
    )
    diagnostics["execute_returncode"] = execute.returncode

    # Pytest's process return code is the authoritative suite result.  Terminal output
    # is a human interface and changes between pytest/plugin versions; a successful
    # subprocess means every selected nodeid passed.
    if execute.returncode == 0:
        diagnostics["recognition_mode"] = "process-returncode"
        passed = set(selected)
    else:
        # On a real failure, retain best-effort per-node diagnostics.  ANSI/control
        # sequences are stripped so this remains useful across pytest versions.
        diagnostics["recognition_mode"] = "failure-output-fallback"
        passed: set[str] = set()
        ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        for raw_line in execute.stdout.splitlines():
            line = ansi.sub("", raw_line).strip()
            if not line.startswith("tests/hedge/hprl/") or " PASSED" not in line:
                continue
            passed.add(line.split(" PASSED", 1)[0].strip())

    detail = (collect.stdout + collect.stderr + execute.stdout + execute.stderr)[-4000:]
    return nodeids, passed, detail, diagnostics


def _source_text() -> str:
    paths = sorted((ROOT / "freqtrade/hedge/hprl").rglob("*.py"))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


def _manifest_paths() -> set[str]:
    payload = json.loads((ROOT / "CLEAN-MAINLINE-MANIFEST.json").read_text(encoding="utf-8"))
    return {str(row["path"]) for row in payload["files"]}


def _command_passes(command: list[str]) -> tuple[bool, str]:
    result = _run(command)
    detail = (result.stdout + result.stderr)[-2000:]
    return result.returncode == 0, detail


def _integration_checks() -> list[tuple[str, Callable[[], object]]]:
    def compatibility_probe() -> bool:
        from freqtrade.hedge.hprl.compatibility import assert_clean_mainline_compatible

        return assert_clean_mainline_compatible(ROOT).compatible

    def registry_exact() -> bool:
        from freqtrade.hedge.hprl.registry import available_algorithms

        return available_algorithms() == (
            "fast_dsac",
            "fast_td3",
            "rebrac_v2",
            "simba_sac",
            "xqc",
        )

    def cli_gpu_policy() -> tuple[bool, str]:
        result = _run([sys.executable, "-m", "freqtrade.hedge.hprl", "inspect"])
        return (
            result.returncode == 0
            and '"device_policy": "cpu-cuda-auto"' in result.stdout
            and '"default_device": "auto"' in result.stdout
            and '"existing_rl_modified": false' in result.stdout,
            result.stdout[-1500:],
        )

    def cpu_smoke() -> tuple[bool, str]:
        result = _run(
            [sys.executable, "-m", "freqtrade.hedge.hprl", "smoke", "--device", "cpu"]
        )
        return result.returncode == 0 and "device=cpu" in result.stdout, result.stdout[-1500:]

    def cpu_train_smoke() -> tuple[bool, str]:
        result = _run(
            [
                sys.executable,
                "-m",
                "freqtrade.hedge.hprl",
                "train-smoke",
                "--device",
                "cpu",
                "--algorithm",
                "fast_td3",
            ]
        )
        return (
            result.returncode == 0
            and '"status": "PASS"' in result.stdout
            and '"updates":' in result.stdout,
            result.stdout[-1800:],
        )

    def auto_device_resolves() -> bool:
        from freqtrade.hedge.hprl.device import resolve_device

        return resolve_device("auto").resolved == (
            "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
        )

    def manifest_has_all_hprl() -> bool:
        expected = {
            path.relative_to(ROOT).as_posix()
            for root in (ROOT / "freqtrade/hedge/hprl", ROOT / "tests/hedge/hprl")
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        expected.update(
            {
                "config_examples/hprl.example.json",
                "config_examples/hprl.gpu.example.json",
                "requirements-hprl.txt",
                "tools/validate_hprl_clean_mainline_200.py",
            }
        )
        return expected <= _manifest_paths()

    def signal_ratio_semantics() -> bool:
        import torch
        from freqtrade.hedge.hprl.adapters import CleanMainlineSignalAdapter

        adapter = CleanMainlineSignalAdapter(("BTC/USDT:USDT",), "check", max_leg_exposure=0.5)
        signal = adapter.decode(torch.tensor([0.5, 0.0]))[0]
        return abs(signal.target_net_ratio - 0.5) <= 1e-6

    def core_authorities_unchanged() -> bool:
        return all(_sha256(ROOT / name) == expected for name, expected in _BASELINE_HASHES.items())

    source = _source_text()
    device_text = (ROOT / "freqtrade/hedge/hprl/device.py").read_text(encoding="utf-8")
    runtime_text = (ROOT / "freqtrade/hedge/hprl/runtime.py").read_text(encoding="utf-8")
    replay_text = (ROOT / "freqtrade/hedge/hprl/replay.py").read_text(encoding="utf-8")
    networks_text = (ROOT / "freqtrade/hedge/hprl/networks.py").read_text(encoding="utf-8")
    generated_names = (
        "HPRL-QUALITY-200-CHECK-RESULT.json",
        "HPRL-DEEP-200-CHECK-RESULT.json",
        "HPRL-SOURCE-MANIFEST.json",
    )
    default_example = (ROOT / "config_examples/hprl.example.json").read_text(encoding="utf-8")
    gpu_example = (ROOT / "config_examples/hprl.gpu.example.json").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements-hprl.txt").read_text(encoding="utf-8")
    checks: list[tuple[str, Callable[[], object]]] = [
        (
            "version_matches_github_backup",
            lambda: (ROOT / "CLEAN-MAINLINE-VERSION.txt").read_text(encoding="utf-8").strip()
            == EXPECTED_VERSION,
        ),
        (
            "hprl_release_and_policy_are_gpu_adapt2",
            lambda: "clean-mainline-v1.2.1-gpu-adapt2"
            in (ROOT / "freqtrade/hedge/hprl/__init__.py").read_text(encoding="utf-8")
            and "cpu-cuda-auto"
            in (ROOT / "freqtrade/hedge/hprl/__init__.py").read_text(encoding="utf-8"),
        ),
        (
            "default_config_uses_auto_device",
            lambda: '"device": "auto"' in default_example
            and '"replay_device": "auto"' in default_example,
        ),
        (
            "gpu_config_enables_cuda_and_amp",
            lambda: '"device": "cuda"' in gpu_example
            and '"mixed_precision": true' in gpu_example
            and '"allow_tf32": true' in gpu_example,
        ),
        (
            "requirements_reuse_project_torch_without_second_pin",
            lambda: "requirements-freqai-rl.txt" in requirements
            and re.search(r"(?m)^\s*torch[=<>!~]", requirements) is None,
        ),
        ("algorithm_registry_exact", registry_exact),
        ("compatibility_probe_passes", compatibility_probe),
        ("signal_net_ratio_uses_equity_semantics", signal_ratio_semantics),
        ("cli_reports_cpu_cuda_auto", cli_gpu_policy),
        ("cpu_gradient_train_smoke_passes", cpu_train_smoke),
        (
            "cuda_and_current_amp_apis_are_implemented",
            lambda: "torch.cuda.is_available" in device_text
            and "torch.amp.GradScaler" in device_text
            and "torch.amp.autocast" in device_text,
        ),
        (
            "deprecated_torch_cuda_amp_api_is_absent",
            lambda: "torch.cuda.amp." not in source,
        ),
        (
            "runtime_is_config_device_driven",
            lambda: "device = config.training.device" in runtime_text
            and "build_online_runtime" in runtime_text
            and "build_offline_runtime" in runtime_text,
        ),
        (
            "categorical_hot_path_has_no_support_item_sync",
            lambda: "target.clamp(self.value_min" not in networks_text
            and "support_min = self.support[0]" in networks_text,
        ),
        (
            "replay_supports_gpu_residency_and_pinned_cpu",
            lambda: "pin_memory" in replay_text
            and "non_blocking=True" in replay_text
            and "torch_device(device)" in replay_text,
        ),
        (
            "no_legacy_rl_imports",
            lambda: all(
                token not in source
                for token in (
                    "freqtrade.freqai.RL",
                    "freqtrade.freqai.hedge_rl",
                    "hedge.native.rl",
                    "hedge.research.rl",
                )
            ),
        ),
        (
            "no_exchange_write_api_tokens",
            lambda: all(
                token not in source
                for token in (
                    "create_order(",
                    "cancel_order(",
                    "edit_order(",
                    "fapiPrivate",
                    "ccxt.",
                )
            ),
        ),
        (
            "no_generated_hprl_authority_files",
            lambda: all(not (ROOT / name).exists() for name in generated_names),
        ),
        ("manifest_includes_gpu_hprl", manifest_has_all_hprl),
        (
            "clean_validator_passes",
            lambda: _command_passes(
                [sys.executable, "tools/validate_clean_mainline.py"]
            ),
        ),
        ("clean_mainline_core_authorities_unchanged", core_authorities_unchanged),
    ]
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="run only 41 protected legacy-RL hashes plus 21 integration/governance checks",
    )
    args = parser.parse_args()
    results: list[dict[str, object]] = []
    pytest_diagnostics: dict[str, object] = {
        "pytest_available": _pytest_available(),
        "mode": "source-only" if args.source_only else "full",
    }

    if not args.source_only:
        nodeids, passed_nodeids, pytest_detail, collected_diagnostics = _collect_hprl_tests()
        pytest_diagnostics.update(collected_diagnostics)
        if len(nodeids) != EXPECTED_TESTS:
            nodeids = nodeids[:EXPECTED_TESTS]
            while len(nodeids) < EXPECTED_TESTS:
                nodeids.append(f"MISSING-HPRL-TEST-{len(nodeids) + 1}")
        for nodeid in nodeids:
            results.append(
                {
                    "category": "pytest",
                    "name": nodeid,
                    "passed": nodeid in passed_nodeids,
                    "detail": "" if nodeid in passed_nodeids else pytest_detail,
                }
            )

    expected_hashes = _protected_hashes()
    for relative in sorted(expected_hashes):
        path = ROOT / relative
        passed = path.is_file() and _sha256(path) == expected_hashes[relative]
        results.append(
            {
                "category": "legacy-rl-hash",
                "name": relative,
                "passed": passed,
                "detail": "byte-exact GitHub V1.2.1 baseline",
            }
        )

    for name, check in _integration_checks():
        try:
            value = check()
            if isinstance(value, tuple):
                passed, detail = bool(value[0]), str(value[1])
            else:
                passed, detail = bool(value), ""
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "category": "integration-governance",
                "name": name,
                "passed": passed,
                "detail": detail,
            }
        )

    for index, row in enumerate(results, start=1):
        row["round"] = index
    passed_count = sum(bool(row["passed"]) for row in results)
    expected_total = EXPECTED_PROTECTED + EXPECTED_INTEGRATION if args.source_only else EXPECTED_TOTAL
    expected_pytest = 0 if args.source_only else EXPECTED_TESTS
    payload = {
        "schema": (
            "hprl-clean-mainline-source-62-v1"
            if args.source_only
            else "hprl-clean-mainline-adaptation-200-v2"
        ),
        "baseline": EXPECTED_VERSION,
        "expected": expected_total,
        "executed": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "status": (
            "PASS"
            if len(results) == expected_total and passed_count == expected_total
            else "FAIL"
        ),
        "composition": {
            "pytest": expected_pytest,
            "legacy_rl_hashes": EXPECTED_PROTECTED,
            "integration_governance": EXPECTED_INTEGRATION,
        },
        "pytest_diagnostics": pytest_diagnostics,
        "rounds": results,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {key: payload[key] for key in ("expected", "executed", "passed", "failed", "status")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
