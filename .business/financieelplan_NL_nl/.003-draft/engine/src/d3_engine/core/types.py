"""Shared types for D3 engine runner orchestration."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RunConfig:
    inputs_dir: Path
    out_dir: Path
    pipelines: List[str]
    seed: Optional[int]
    fail_on_warning: bool
    max_concurrency: int


@dataclass(frozen=True)
class RunContext:
    state: Dict[str, Any]
    targets: Dict[str, Any]
    policy: Any  # ASGPolicy-like object


@dataclass(frozen=True)
class Job:
    grid_index: int
    replicate_index: int
    mc_index: int
    combo: Dict[str, float]


@dataclass(frozen=True)
class JobResult:
    grid_index: int
    replicate_index: int
    mc_index: int
    public: Dict[str, tuple[list[str], list[dict]]]
    private: Dict[str, tuple[list[str], list[dict]]]


@dataclass(frozen=True)
class RunResult:
    artifacts: List[str]
    analysis: Dict[str, Any]
    accepted: bool
    input_hashes: List[Dict[str, str]]
