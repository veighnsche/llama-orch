from __future__ import annotations
import shutil
from pathlib import Path
import yaml
from pytest_bdd import given, when, then, parsers


def _set_dotted(d: dict, dotted: str, value):
    cur = d
    parts = dotted.split('.')
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


@given('I copy the inputs to a temporary workspace')
def copy_inputs_to_ws(tmp_path, ctx):
    src = ctx['inputs']
    dst = tmp_path / 'ws_inputs'
    shutil.copytree(src, dst, dirs_exist_ok=True)
    ctx['inputs'] = dst


@given(parsers.parse('I set simulation.yaml key "{key}" to {raw}'))
def set_sim_yaml_key(ctx, key: str, raw: str):
    # Interpret raw as JSON-like for booleans/numbers; fallback to string
    value: object
    if raw.lower() in ('true', 'false'):
        value = (raw.lower() == 'true')
    else:
        try:
            if '.' in raw:
                value = float(raw)
            else:
                value = int(raw)
        except Exception:
            value = raw
    sim_p = ctx['inputs'] / 'simulation.yaml'
    data = yaml.safe_load(sim_p.read_text()) or {}
    _set_dotted(data, key, value)
    sim_p.write_text(yaml.safe_dump(data))


@then(parsers.parse('stdout should contain event "{event}"'))
def then_stdout_has_event(ctx, event: str):
    found = any((f'"event": "{event}"' in line) for line in ctx['stdout'].splitlines())
    assert found, f"event {event} not found in stdout"
