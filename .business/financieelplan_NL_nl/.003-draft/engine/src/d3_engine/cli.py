#!/usr/bin/env python3
"""D3 Engine CLI (scaffold)

Parses arguments and emits JSONL progress to stdout. Real simulation logic
and writing of run summaries/outputs live in d3_engine.core.runner.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from d3_engine.core import runner
from d3_engine.core.validator import ValidationError


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

    try:
        # Execute orchestrated run
        pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
        _ = runner.execute(inputs, out_dir, pipelines, args.seed, args.fail_on_warning, args.max_concurrency)

        _jsonl("run_done")
        return 0
    except KeyboardInterrupt:
        _jsonl("error", code=3, kind="INTERRUPTED", message="Execution interrupted by user (Ctrl-C)")
        return 3
    except ValidationError as ve:
        _jsonl("error", code=2, kind="VALIDATION_ERROR", message=str(ve))
        return 2
    except Exception as e:
        _jsonl("error", code=3, kind="RUNTIME_ERROR", message=str(e))
        return 3


if __name__ == "__main__":
    sys.exit(main())
