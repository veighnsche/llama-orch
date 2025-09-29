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
    """Copy-on-write overlay applying dotted-path overrides without deep-copying the whole state.

    High-risk speed path: assumes pipelines do not mutate shared substructures under untouched paths.
    """
    if not overrides:
        return base
    # Shallow copy the root only; then clone along each override path.
    root = dict(base)
    for dotted, value in overrides.items():
        cur_base = base
        cur_new = root
        parts = dotted.split(".")
        for p in parts[:-1]:
            base_child = cur_base.get(p, {}) if isinstance(cur_base, dict) else {}
            new_child = cur_new.get(p)
            # If absent or aliasing the base child, create a shallow copy
            if not isinstance(new_child, dict) or new_child is base_child:
                new_child = dict(base_child) if isinstance(base_child, dict) else {}
                cur_new[p] = new_child
            cur_base = base_child if isinstance(base_child, dict) else {}
            cur_new = new_child
        cur_new[parts[-1]] = value
    return root


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
    """Execute jobs sequentially or with a pool. High-risk speed path for processes mode:

    - Avoid per-job pickling of large state via worker initializer and module-level globals.
    - Batch jobs to reduce IPC overhead and progress-tick per batch.
    - Preserve determinism at the artifact level (writers sort outputs by indices).
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

    # Optimized processes path with worker globals and chunked mapping
    if mode == "processes":
        # Fallback to legacy behavior if initializer args are missing
        if not compute_args or len(compute_args) != 4:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                future_map = {ex.submit(compute_job, job, *compute_args): job for job in jobs_list}
                for fut in as_completed(future_map):
                    _ = future_map[fut]
                    res = fut.result()
                    results.append(res)
                    progress.tick(1)
                progress.finalize()
                return results

        base_state, pipes, random_specs, master_seed = compute_args

        def _chunks(seq: List[JobT], size: int):
            for i in range(0, len(seq), size):
                yield seq[i : i + size]

        # Heuristic chunk size: balance IPC and latency
        chunk_size = max(32, (len(jobs_list) // max(1, workers * 8)) or 32)
        batches = list(_chunks(jobs_list, chunk_size))

        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(base_state, pipes, random_specs, master_seed)) as ex:
            # Map batches; each result is a list of job results
            for res_batch in ex.map(_worker_compute_batch, batches):
                results.extend(res_batch)
                progress.tick(len(res_batch))
        progress.finalize()
        return results

    # Threads path (legacy)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(compute_job, job, *compute_args): job for job in jobs_list}
        for fut in as_completed(future_map):
            _ = future_map[fut]
            res = fut.result()
            results.append(res)
            progress.tick(1)
    progress.finalize()
    return results


# --------------------
# Multiprocessing worker globals and functions (must be top-level for pickling)
# --------------------

_G_BASE = None
_G_PIPES: List[str] = []
_G_SPECS = None
_G_SEED = 0


def _worker_init(base, pipes, specs, seed):
    global _G_BASE, _G_PIPES, _G_SPECS, _G_SEED
    _G_BASE = base
    _G_PIPES = list(pipes) if pipes else []
    _G_SPECS = specs
    _G_SEED = int(seed)


def _worker_compute_job(job: JobT):
    gi, ri, mi, combo = job
    from core import variables as _vargrid
    overrides = dict(combo)
    rand = _vargrid.draw_randoms(_G_SPECS, _G_SEED, gi, ri, mi)
    overrides.update(rand)
    state_job = _with_overrides(_G_BASE, overrides) if overrides else _G_BASE
    pub = REGISTRY_SAFE["public"](state_job) if "public" in _G_PIPES else {}
    prv = REGISTRY_SAFE["private"](state_job) if "private" in _G_PIPES else {}
    return (gi, ri, mi, pub, prv)


def _worker_compute_batch(batch: List[JobT]):
    return [_worker_compute_job(j) for j in batch]
