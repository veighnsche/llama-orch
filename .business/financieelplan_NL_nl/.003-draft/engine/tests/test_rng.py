from __future__ import annotations
from d3_engine.core import rng


def test_seed_resolution_precedence():
    state = {
        "simulation": {
            "run": {"random_seed": 222},
            "stochastic": {"random_seed": 111},
        },
        "operator": {
            "public_tap": {"meta": {"seed": 333}},
        },
    }
    assert rng.resolve_seed_from_state(state) == 111

    state2 = {
        "simulation": {"run": {"random_seed": 222}},
        "operator": {"public_tap": {"meta": {"seed": 333}}},
    }
    assert rng.resolve_seed_from_state(state2) == 222

    state3 = {
        "simulation": {},
        "operator": {"public_tap": {"meta": {"seed": 333}}},
    }
    assert rng.resolve_seed_from_state(state3) == 333


def test_substream_determinism():
    master = 424242
    a1 = rng.substream(master, "ns", 0, 0, 0).random()
    a2 = rng.substream(master, "ns", 0, 0, 0).random()
    b1 = rng.substream(master, "ns", 0, 0, 1).random()
    assert a1 == a2
    assert a1 != b1
