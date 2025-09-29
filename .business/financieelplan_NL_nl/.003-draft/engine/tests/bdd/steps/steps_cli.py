from __future__ import annotations
import io
import json
import sys
from pathlib import Path
from contextlib import redirect_stdout
from pytest_bdd import given, when, then, parsers
import cli as engine_cli
from ..util_inputs import make_minimal_inputs


@given(parsers.parse('the inputs directory ../inputs'))
def given_inputs_dir(tmp_path, ctx):
    # Create minimal inputs under a temp folder
    ctx['inputs'] = make_minimal_inputs(tmp_path)


@given('a fresh outputs directory')
def given_outputs(tmp_path, ctx):
    out = tmp_path / 'outputs'
    out.mkdir(parents=True, exist_ok=True)
    ctx['outputs'] = out


@given('another fresh outputs directory')
def given_outputs2(tmp_path, ctx):
    out2 = tmp_path / 'outputs2'
    out2.mkdir(parents=True, exist_ok=True)
    ctx['outputs2'] = out2


@when(parsers.parse('I run the engine CLI with pipelines "{pipelines}" and seed {seed:d}'))
def when_run_engine(pipelines: str, seed: int, ctx, monkeypatch):
    inputs = ctx['inputs']
    out = ctx.get('outputs') or (inputs.parent / 'outputs')
    # Prepare argv
    argv = [
        'cli',
        '--inputs', str(inputs),
        '--out', str(out),
        '--pipelines', pipelines,
        '--seed', str(seed),
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        monkeypatch.setattr(sys, 'argv', argv)
        code = engine_cli.main()
    ctx['exit_code'] = code
    ctx['stdout'] = buf.getvalue()
    ctx['outputs'] = out


@when(parsers.parse('I run the engine CLI with pipelines "{pipelines}" and seed {seed:d} into the second outputs directory'))
def when_run_engine_second(pipelines: str, seed: int, ctx, monkeypatch):
    inputs = ctx['inputs']
    out = ctx['outputs2']
    argv = [
        'cli',
        '--inputs', str(inputs),
        '--out', str(out),
        '--pipelines', pipelines,
        '--seed', str(seed),
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        monkeypatch.setattr(sys, 'argv', argv)
        code = engine_cli.main()
    ctx['exit_code2'] = code
    ctx['stdout2'] = buf.getvalue()
    ctx['outputs2'] = out


@then(parsers.parse('the command should exit with code {code:d}'))
def then_exit_code(ctx, code: int):
    assert ctx.get('exit_code', ctx.get('exit_code2')) == code


@then(parsers.parse('the outputs directory should contain file "{name}"'))
def then_outputs_contains(ctx, name: str):
    out = ctx['outputs']
    assert (out / name).exists()


@then(parsers.parse('stdout should contain JSONL records with keys "{keys}"'))
def then_stdout_contains(ctx, keys: str):
    ks = [k.strip() for k in keys.split(',')]
    lines = [l for l in ctx['stdout'].splitlines() if l.strip().startswith('{')]
    assert lines, 'no JSONL lines captured'
    rec = json.loads(lines[0])
    for k in ks:
        assert k in rec


@then(parsers.parse('the run_summary key "{key}" should match between outputs'))
def then_run_summary_match(ctx, key: str):
    import yaml
    out1 = ctx['outputs'] / 'run_summary.json'
    out2 = ctx['outputs2'] / 'run_summary.json'
    d1 = yaml.safe_load(out1.read_text())
    d2 = yaml.safe_load(out2.read_text())
    assert d1.get(key) == d2.get(key)
