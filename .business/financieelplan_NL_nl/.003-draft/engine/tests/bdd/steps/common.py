import json
import csv
import shlex
import shutil
import subprocess
import sys
import hashlib
from pathlib import Path

import yaml
import pytest
from pytest_bdd import given, when, then, parsers


@given(parsers.parse("the inputs directory {path}"))
def given_inputs_dir(ctx, draft_dir: Path, path: str):
    # Resolve common relative token ../inputs to the draft's inputs
    if path.strip() == "../inputs":
        p = draft_dir / "inputs"
    else:
        p = Path(path)
        if not p.is_absolute():
            p = (draft_dir / p).resolve()
    assert p.exists(), f"Inputs directory not found: {p}"
    ctx["inputs"] = p


@given("a fresh outputs directory")
def given_fresh_outputs_dir(ctx, tmp_path: Path):
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    ctx["out"] = out


@given("another fresh outputs directory")
def given_another_fresh_outputs_dir(ctx, tmp_path: Path):
    out2 = tmp_path / "out2"
    out2.mkdir(parents=True, exist_ok=True)
    ctx["out2"] = out2


@given("I copy the inputs to a temporary workspace")
def given_copy_inputs(ctx, draft_dir: Path, tmp_path: Path):
    src = draft_dir / "inputs"
    ws = tmp_path / "ws"
    dst = ws / "inputs"
    shutil.copytree(src, dst)
    ctx["inputs"] = dst
    ctx["workspace"] = ws


@given(parsers.parse('I set simulation.yaml key "{dotted}" to {value}'))
def given_set_sim_yaml_key(ctx, dotted: str, value: str):
    # supports numeric values currently
    inputs: Path = ctx["inputs"]
    sim = inputs / "simulation.yaml"
    data = yaml.safe_load(sim.read_text())
    # walk dotted path
    keys = dotted.split(".")
    cur = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    # coerce value to int/float if possible
    v: object
    try:
        v = int(value)
    except ValueError:
        try:
            v = float(value)
        except ValueError:
            if value.lower() in ("true", "false"):
                v = value.lower() == "true"
            else:
                v = value
    cur[keys[-1]] = v
    sim.write_text(yaml.safe_dump(data))


@given(parsers.parse('I set operator "{tap}" key "{dotted}" to {value}'))
def given_set_operator_yaml_key(ctx, tap: str, dotted: str, value: str):
    inputs: Path = ctx["inputs"]
    fname = tap if tap.endswith('.yaml') else f"{tap}.yaml"
    op = inputs / "operator" / fname
    assert op.exists(), f"Operator file not found: {op}"
    data = yaml.safe_load(op.read_text())
    keys = dotted.split(".")
    cur = data
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    # coerce value
    v: object
    try:
        v = int(value)
    except ValueError:
        try:
            v = float(value)
        except ValueError:
            if isinstance(value, str) and value.lower() in ("true", "false"):
                v = value.lower() == "true"
            else:
                v = value
    cur[keys[-1]] = v
    op.write_text(yaml.safe_dump(data))


@when(parsers.parse('I run the engine CLI with pipelines "{pipelines}" and seed {seed:d}'))
def when_run_engine_cli(ctx, pipelines: str, seed: int):
    inputs: Path = ctx["inputs"]
    out: Path = ctx["out"]
    cmd = [
        sys.executable,
        "-m",
        "d3_engine.cli",
        "--inputs",
        str(inputs),
        "--out",
        str(out),
        "--pipelines",
        pipelines,
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ctx["proc"] = proc
    ctx["stdout_lines"] = proc.stdout.splitlines()


@when(parsers.parse('I run the engine CLI with pipelines "{pipelines}" and seed {seed:d} into the second outputs directory'))
def when_run_engine_cli_second(ctx, pipelines: str, seed: int):
    inputs: Path = ctx["inputs"]
    out: Path = ctx["out2"]
    cmd = [
        sys.executable,
        "-m",
        "d3_engine.cli",
        "--inputs",
        str(inputs),
        "--out",
        str(out),
        "--pipelines",
        pipelines,
        "--seed",
        str(seed),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ctx["proc2"] = proc
    ctx["stdout_lines2"] = proc.stdout.splitlines()


@when(parsers.parse('I run the engine CLI with args "{args}"'))
def when_run_engine_cli_with_args(ctx, args: str):
    tokens = shlex.split(args)
    cmd = [sys.executable, "-m", "d3_engine.cli"] + tokens
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ctx["proc"] = proc
    ctx["stdout_lines"] = proc.stdout.splitlines()


@then(parsers.parse("the command should exit with code {code:d}"))
def then_exit_code(ctx, code: int):
    proc: subprocess.CompletedProcess = ctx["proc"]
    assert proc.returncode == code, f"Expected {code}, got {proc.returncode}, stderr=\n{proc.stderr}"


@then(parsers.parse('the outputs directory should contain file "{name}"'))
def then_out_contains(ctx, name: str):
    out: Path = ctx["out"]
    target = out / name
    assert target.exists(), f"Missing output file: {target}"


@then(parsers.parse('the outputs directory should contain CSV "{name}" with headers {headers_csv}'))
def then_out_csv_headers(ctx, name: str, headers_csv: str):
    out: Path = ctx["out"]
    target = out / name
    assert target.exists(), f"Missing CSV: {target}"
    with target.open() as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
    expected = [h.strip().strip('"') for h in headers_csv.split(',') if h.strip()]
    assert hdr[: len(expected)] == expected, f"Headers mismatch: {hdr} vs {expected}"


@then(parsers.parse('CSV "{name}" column "{col}" should be monotonic nondecreasing'))
def then_csv_column_monotonic(ctx, name: str, col: str):
    out: Path = ctx["out"]
    target = out / name
    assert target.exists(), f"Missing CSV: {target}"
    with target.open() as f:
        rdr = csv.DictReader(f)
        prev = None
        for row in rdr:
            try:
                val = float(row[col])
            except Exception:
                pytest.fail(f"Cannot parse column {col} value: {row.get(col)}")
            if prev is not None and val < prev - 1e-9:
                pytest.fail(f"Column {col} not monotonic: {prev} -> {val}")
            prev = val


@then(parsers.parse('stdout should contain substring "{text}"'))
def then_stdout_contains(ctx, text: str):
    data = "\n".join(ctx.get("stdout_lines", []))
    assert text in data, f"Substring not found in stdout: {text!r}"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@then(parsers.parse('file "{name}" hashes should match between outputs'))
def then_file_hashes_match_between_outputs(ctx, name: str):
    out1: Path = ctx["out"]
    out2: Path = ctx["out2"]
    f1 = out1 / name
    f2 = out2 / name
    assert f1.exists() and f2.exists(), f"Missing files to compare: {f1}, {f2}"
    assert _sha256(f1) == _sha256(f2), f"Hash mismatch: {f1} vs {f2}"


@then(parsers.parse('the run_summary key "{key}" should match between outputs'))
def then_run_summary_key_match_between_outputs(ctx, key: str):
    out1: Path = ctx["out"]
    out2: Path = ctx["out2"]
    p1 = out1 / "run_summary.json"
    p2 = out2 / "run_summary.json"
    assert p1.exists() and p2.exists(), "run_summary.json missing in one of outputs"
    d1 = json.loads(p1.read_text())
    d2 = json.loads(p2.read_text())
    assert key in d1 and key in d2, f"Key {key!r} missing in summaries"
    assert d1[key] == d2[key], f"run_summary[{key!r}] differ: {d1[key]} vs {d2[key]}"


@then(parsers.parse('CSV "{name}" column "{col}" should be all "{value}"'))
def then_csv_column_all_equal(ctx, name: str, col: str, value: str):
    out: Path = ctx["out"]
    target = out / name
    assert target.exists(), f"Missing CSV: {target}"
    with target.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get(col) != value:
                pytest.fail(f"Column {col} expected {value} but saw {row.get(col)} in row {row}")


@then(parsers.parse('CSV "{name}" column "{col}" should be all >= {min_val:f}'))
def then_csv_column_all_gte(ctx, name: str, col: str, min_val: float):
    out: Path = ctx["out"]
    target = out / name
    assert target.exists(), f"Missing CSV: {target}"
    with target.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                val = float(row[col])
            except Exception:
                pytest.fail(f"Cannot parse column {col} value: {row.get(col)}")
            if val < min_val - 1e-9:
                pytest.fail(f"Column {col} value {val} < {min_val}")


@then("stdout should contain JSONL records with keys \"ts\",\"level\",\"event\"")
def then_stdout_jsonl_keys(ctx):
    lines = ctx.get("stdout_lines", [])
    assert lines, "No stdout lines captured"
    saw = False
    for ln in lines:
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        keys = set(rec.keys())
        if {"ts", "level", "event"} <= keys:
            saw = True
            break
    assert saw, "No JSONL record with required keys found on stdout"


@then(parsers.parse('stdout should contain event "{event}"'))
def then_stdout_contains_event(ctx, event: str):
    lines = ctx.get("stdout_lines", [])
    for ln in lines:
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == event:
            return
    assert False, f"No JSONL record with event={event!r} found"


@then(parsers.parse('the run_summary should contain keys {keys_csv}'))
def then_run_summary_has_keys(ctx, keys_csv: str):
    out: Path = ctx["out"]
    p = out / "run_summary.json"
    assert p.exists(), "run_summary.json is missing"
    data = json.loads(p.read_text())
    needed = {k.strip().strip('"') for k in keys_csv.split(',') if k.strip()}
    assert needed <= set(data.keys()), f"Missing keys: {needed - set(data.keys())}"
