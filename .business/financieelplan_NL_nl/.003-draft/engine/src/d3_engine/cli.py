#!/usr/bin/env python3
"""D3 Engine CLI (scaffold)

Parses arguments, emits JSONL progress to stdout, and writes a minimal
run_summary.{json,md} to the output directory. Real simulation logic lives
in submodules under d3_engine.core/, services/, and pipelines/.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from d3_engine.core import runner


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jsonl(event: str, **kwargs):
    rec = {"ts": _ts(), "level": "INFO", "event": event}
    rec.update(kwargs)
    print(json.dumps(rec), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="D3 Engine CLI (scaffold)")
    parser.add_argument("--inputs", required=True, help="Path to .003-draft/inputs")
    parser.add_argument("--out", required=True, help="Path to outputs directory")
    parser.add_argument("--pipelines", default="public,private", help="Comma-separated: public,private")
    parser.add_argument("--seed", type=int, default=None, help="Master seed (optional)")
    parser.add_argument("--fail-on-warning", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--max-concurrency", type=int, default=0, help="Optional parallelism hint")
    args = parser.parse_args()

    inputs = Path(args.inputs)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    _jsonl("run_start", inputs=str(inputs), out=str(out_dir), pipelines=args.pipelines, seed=args.seed)

    # Execute orchestrated run
    pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    result = runner.execute(inputs, out_dir, pipelines, args.seed, args.fail_on_warning)

    # Write run summary from runner result
    summary = {
        "ts": _ts(),
        **result,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "run_summary.md").write_text("# Run Summary\n\nThis is a scaffold run. Implement engine logic in d3_engine.* modules.\n")

    _jsonl("run_done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
