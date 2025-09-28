"""Validator (scaffold).
Schema/type/domain/reference checks per .specs/10_inputs.md and friends.
"""
from typing import Dict, Any


class ValidationError(Exception):
    pass


def validate(state: Dict[str, Any]) -> None:
    # TODO: implement schema checks
    _ = state
