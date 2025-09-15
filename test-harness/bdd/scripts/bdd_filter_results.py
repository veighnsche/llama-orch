#!/usr/bin/env python3
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "cargo" / "switchyard"
BDD_ENV = os.environ.copy()

CMD = [
    "cargo",
    "test",
    "-p",
    "switchyard",
    "--features",
    "bdd",
    "--test",
    "bdd",
]

LOG_DIR = ROOT.parent.parent / "target"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LASTRUN = LOG_DIR / "bdd-lastrun.log"

# Matchers for extracting failure-relevant lines from cucumber output
FAIL_MARK = re.compile(r"^\s*✘\s+")
FAIL_DEFINED = re.compile(r"^\s*Defined:\s+(SPEC/[^:]+\.feature)")
FAIL_MATCHED = re.compile(r"^\s*Matched:\s+")
FAIL_STEP = re.compile(r"^\s*Step failed:")
FAIL_PANIC = re.compile(r"^\s*Step panicked\.")
SUMMARY = re.compile(
    r"""
    ^\[Summary\]
    |^\s*\d+\s+features$
    |^\s*\d+\s+scenarios.*$
    |^\s*\d+\s+steps.*$
    """,
    re.X,
)


def run_bdd(feature_path: str | None = None) -> subprocess.CompletedProcess:
    env = BDD_ENV.copy()
    if feature_path:
        env["SWITCHYARD_BDD_FEATURE_PATH"] = feature_path
    return subprocess.run(
        CMD,
        cwd=str(ROOT.parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )


def filter_failures_only(output: str) -> str:
    lines = output.splitlines()
    filtered: list[str] = []
    for i, line in enumerate(lines):
        if (
            FAIL_MARK.search(line)
            or FAIL_DEFINED.search(line)
            or FAIL_MATCHED.search(line)
            or FAIL_STEP.search(line)
            or FAIL_PANIC.search(line)
            or SUMMARY.search(line)
        ):
            filtered.append(line)
            # Include assertion context lines that often follow a panic
            # Copy next 3 lines if they exist and aren't a new feature header
            if FAIL_STEP.search(line) or FAIL_PANIC.search(line):
                for j in range(1, 4):
                    if i + j < len(lines):
                        nxt = lines[i + j]
                        if nxt.startswith("Feature:"):
                            break
                        filtered.append(nxt)
    if not filtered:
        # If nothing matched, return original output so users still see something useful
        return output
    return "\n".join(filtered) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Run BDD tests with optional failure-only output filter.")
    ap.add_argument(
        "--features",
        help="Optional path under SPEC/features to run (file or directory). Defaults to all.",
        default=None,
    )
    ap.add_argument(
        "--fail-only",
        action="store_true",
        help="Print only failing steps and summary lines (filters out passing steps).",
    )
    args = ap.parse_args()

    proc = run_bdd(feature_path=args.features)
    LASTRUN.write_text(proc.stdout)

    if args.fail_only:
        sys.stdout.write(filter_failures_only(proc.stdout))
    else:
        sys.stdout.write(proc.stdout)

    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
