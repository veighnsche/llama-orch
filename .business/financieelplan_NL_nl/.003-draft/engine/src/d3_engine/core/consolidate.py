"""Consolidation (scaffold)."""
from typing import Dict, Any


def consolidate(public: Dict[str, Any], private: Dict[str, Any]) -> Dict[str, Any]:
    return {"public": public, "private": private}
