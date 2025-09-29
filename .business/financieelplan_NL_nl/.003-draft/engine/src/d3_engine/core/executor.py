"""Job execution helpers for runner orchestration."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Tuple

from . import logging as elog

JobT = Tuple[int, int, int, dict]  # (grid_index, replicate_index, mc_index, combo)


def execute_jobs(
    jobs: Iterable[JobT],
    compute_job: Callable[[JobT], Tuple[int, int, int, dict, dict]],
    max_workers: int,
) -> List[Tuple[int, int, int, dict, dict]]:
    """Execute jobs sequentially or with a thread pool.

    Emits JSONL job_started/job_done logs and returns results preserving determinism
    by logging on completion and collecting results for later deterministic writing.
    """
    jobs_list: List[JobT] = list(jobs)
    results: List[Tuple[int, int, int, dict, dict]] = []
    workers = int(max_workers) if (max_workers and max_workers > 0) else 1

    if workers == 1:
        for gi, ri, mi, combo in jobs_list:
            print(elog.jsonl("job_started", grid_index=gi, replicate_index=ri, mc_index=mi))
            res = compute_job((gi, ri, mi, combo))
            results.append(res)
            print(elog.jsonl("job_done", grid_index=gi, replicate_index=ri, mc_index=mi))
        return results

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {ex.submit(compute_job, job): job for job in jobs_list}
        for fut in as_completed(future_map):
            gi, ri, mi, _ = future_map[fut]
            print(elog.jsonl("job_started", grid_index=gi, replicate_index=ri, mc_index=mi))
            res = fut.result()
            results.append(res)
            print(elog.jsonl("job_done", grid_index=gi, replicate_index=ri, mc_index=mi))
    return results
