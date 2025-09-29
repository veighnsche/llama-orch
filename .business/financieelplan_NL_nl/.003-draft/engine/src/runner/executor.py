"""Job execution helpers for runner orchestration."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, List, Tuple

from core import logging as elog
import sys
import shutil
from core import variables as vargrid
from pipelines import REGISTRY as PIPELINES_REGISTRY

JobT = Tuple[int, int, int, dict]  # (grid_index, replicate_index, mc_index, combo)


class _SimpleProgress:
    """Minimal terminal progress bar using a single updating line on stderr."""

    def __init__(self, total: int) -> None:
        self.total = int(total)
        self.current = 0
        self._render()

    def _render(self) -> None:
        total = max(1, self.total)
        cols = shutil.get_terminal_size((80, 20)).columns
        # Reserve space for counter and percent
        prefix = f" {self.current}/{total} "
        pct = int(self.current * 100 / total)
        suffix = f" {pct:3d}%"
        bar_width = max(10, cols - len(prefix) - len(suffix) - 6)
        filled = int(bar_width * self.current / total)
        bar = "#" * filled + "-" * (bar_width - filled)
        line = f"[{bar}]{prefix}{suffix}"
        print("\r" + line[:cols - 1], end="", file=sys.stderr, flush=True)

    def tick(self, n: int = 1) -> None:
        self.current = min(self.total, self.current + int(n))
        self._render()

    def finalize(self) -> None:
        self.current = self.total
        self._render()
        print(file=sys.stderr)


def _set_path(d: dict, dotted: str, value) -> None:
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _with_overrides(base: dict, overrides: dict) -> dict:
    from copy import deepcopy
    if not overrides:
        return base
    new_state = deepcopy(base)
    for path, val in overrides.items():
        _set_path(new_state, path, val)
    return new_state


def compute_job_entry(job: JobT, state: dict, pipelines: List[str], random_specs, master_seed: int) -> Tuple[int, int, int, dict, dict]:
    gi, ri, mi, combo = job
    random_overrides = vargrid.draw_randoms(random_specs, master_seed, gi, ri, mi)
    overrides = dict(combo)
    overrides.update(random_overrides)
    state_job = _with_overrides(state, overrides) if overrides else state
    pub_tables = REGISTRY_SAFE["public"](state_job) if "public" in pipelines else {}
    prv_tables = REGISTRY_SAFE["private"](state_job) if "private" in pipelines else {}
    return (gi, ri, mi, pub_tables, prv_tables)


# Create a module-level safe alias for pipelines registry to improve picklability
REGISTRY_SAFE = PIPELINES_REGISTRY


def execute_jobs(
    jobs: Iterable[JobT],
    compute_job: Callable[..., Tuple[int, int, int, dict, dict]],
    max_workers: int,
    total_jobs: int | None = None,
    compute_args: tuple = (),
    mode: str = "threads",
) -> List[Tuple[int, int, int, dict, dict]]:
    """Execute jobs sequentially or with a thread pool.

    Emits JSONL job_started/job_done logs and returns results preserving determinism
    by logging on completion and collecting results for later deterministic writing.
    """
    jobs_list: List[JobT] = list(jobs)
    results: List[Tuple[int, int, int, dict, dict]] = []
    workers = int(max_workers) if (max_workers and max_workers > 0) else 1
    progress = _SimpleProgress(total=(len(jobs_list) if total_jobs is None else int(total_jobs)))

    if workers == 1 and mode != "processes":
        for gi, ri, mi, combo in jobs_list:
            res = compute_job((gi, ri, mi, combo), *compute_args)
            results.append(res)
            progress.tick(1)
        progress.finalize()
        return results

    if mode == "processes":
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(compute_job, job, *compute_args): job for job in jobs_list}
            for fut in as_completed(future_map):
                _ = future_map[fut]
                res = fut.result()
                results.append(res)
                progress.tick(1)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(compute_job, job, *compute_args): job for job in jobs_list}
            for fut in as_completed(future_map):
                _ = future_map[fut]
                res = fut.result()
                results.append(res)
                progress.tick(1)
    progress.finalize()
    return results
