#!/usr/bin/env python3
"""
Dev-only guard: ensure no source file exceeds line limits.
- fp (CLI): <= 200 lines
- any other .py source under this folder: <= 300 lines
Exit non-zero on violation.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAX_FP = 200
MAX_PY = 300

violations = []

for dirpath, _, filenames in os.walk(ROOT):
    for fn in filenames:
        if not fn.endswith('.py') and fn != 'fp':
            continue
        p = Path(dirpath) / fn
        try:
            with p.open('r', encoding='utf-8', errors='ignore') as f:
                count = sum(1 for _ in f)
        except Exception as e:
            print(f"WARN: cannot read {p}: {e}")
            continue
        if fn == 'fp':
            if count > MAX_FP:
                violations.append((str(p), count, MAX_FP))
        else:
            if count > MAX_PY:
                violations.append((str(p), count, MAX_PY))

if violations:
    print("Line length violations:")
    for path, cnt, limit in violations:
        print(f"- {path}: {cnt} > {limit}")
    sys.exit(1)

print("OK: line limits respected.")
