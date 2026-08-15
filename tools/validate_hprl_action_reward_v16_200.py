"""Execute the HPRL Action/Reward V1.6 200-point acceptance suite."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


EXPECTED = 200


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _clean_output(text: str) -> str:
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    return ansi.sub("", text)


def _run_pytest(root: Path) -> tuple[int, str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/hedge/hprl",
        "--confcutdir=tests/hedge/hprl",
        "-o",
        "addopts=",
        "--disable-warnings",
    ]
    proc = subprocess.run(
        cmd,
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return proc.returncode, _clean_output(proc.stdout), _clean_output(proc.stderr)


def _parse_count(stdout: str) -> tuple[int, int]:
    passed_match = re.search(r"(\d+) passed", stdout)
    failed_match = re.search(r"(\d+) failed", stdout)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = _project_root()
    returncode, stdout, stderr = _run_pytest(root)
    passed, failed = _parse_count(stdout)
    executed = passed + failed
    status = "PASS" if returncode == 0 and passed == EXPECTED and failed == 0 else "FAIL"
    result = {
        "schema": "hprl-action-reward-v16-200",
        "expected": EXPECTED,
        "executed": executed,
        "passed": passed,
        "failed": failed,
        "status": status,
    }
    if status != "PASS":
        result["pytest_returncode"] = returncode
        result["pytest_stdout_tail"] = stdout[-4000:]
        result["pytest_stderr_tail"] = stderr[-4000:]
    text = json.dumps(result, ensure_ascii=True, sort_keys=True)
    print(f"HPRL ACTION REWARD V1.6 200: {passed}/{EXPECTED} PASS; FAIL={failed}")
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
