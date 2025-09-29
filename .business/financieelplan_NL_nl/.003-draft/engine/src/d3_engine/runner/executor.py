"""Job execution helpers for runner orchestration."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Tuple

from ..core import logging as elog
import sys
import shutil

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


def execute_jobs(
    jobs: Iterable[JobT],
    compute_job: Callable[[JobT], Tuple[int, int, int, dict, dict]],
    max_workers: int,
    total_jobs: int | None = None,
) -> List[Tuple[int, int, int, dict, dict]]:
    """Execute jobs sequentially or with a thread pool.

    Emits JSONL job_started/job_done logs and returns results preserving determinism
    by logging on completion and collecting results for later deterministic writing.
    """
    jobs_list: List[JobT] = list(jobs)
    results: List[Tuple[int, int, int, dict, dict]] = []
    workers = int(max_workers) if (max_workers and max_workers > 0) else 1
    progress = _SimpleProgress(total=(len(jobs_list) if total_jobs is None else int(total_jobs)))

    if workers == 1:
        for gi, ri, mi, combo in jobs_list:
            res = compute_job((gi, ri, mi, combo))
            results.append(res)
            progress.tick(1)
        progress.finalize()
        return results

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(compute_job, job): job for job in jobs_list}
        for fut in as_completed(future_map):
            _ = future_map[fut]
            res = fut.result()
            results.append(res)
            progress.tick(1)
    progress.finalize()
    return results
