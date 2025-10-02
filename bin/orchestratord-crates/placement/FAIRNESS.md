# Placement Fairness Policy (orchestratord)

Status: Implementer Guide (aligns with `/.specs/35-worker-orcd-pool-managerd-contract.md` §2.6)
Scope: Inter-worker fairness and tie-breaking for task placement

---

## Goals

- Avoid hot-spotting a single worker when many are eligible.
- Prevent starvation: eligible workers should eventually receive work.
- Minimize latency: prefer workers that already have the requested model resident.
- Respect pin overrides and drain gating.

---

## Eligibility Filter

1. Exclude workers with `draining == true`.
2. For immediate execute:
   - Require `slots_free > 0`.
   - Prefer workers with the model `handle` already resident.
3. For staging (commit then execute):
   - Require `vram_free_bytes >= estimated_vram(model_ref)`.

Inputs come from cached snapshots of:
- `GET /worker/ready` (resident handles)
- `GET /worker/capacity` (slots, VRAM, draining)

---

## Tie-Breaking (Inter-Worker)

Apply in order:

1. Loaded preference: workers that already have the model resident.
2. Least-in-flight: minimal `executing_count = slots_total - slots_free`.
3. Stageable preference (when staging): maximum `vram_free_bytes`.
4. Stable tiebreak: pick the worker with the longest time since last assignment. If not tracked, apply a small random jitter.

Pin overrides bypass fairness but MUST still pass eligibility.

---

## Performance-Aware Scoring (After Fairness)

After applying eligibility and fairness (above), orchestrator MAY use performance telemetry to break ties and improve tail latency:

- **Per-model throughput**: maintain an EWMA of tokens/sec per `{worker_id, model_ref}` computed from worker metrics (`worker_tokens_out_total`, `worker_decode_time_ms`).
- **ETA hints**: use `est_slot_eta_ms`, `est_commit_eta_ms`, `est_eviction_eta_ms` from Plan decisions when provided.
- **Scoring example** (illustrative; tune weights):
  - `score = w_loaded*loaded_pref + w_iflight*(1.0 / (1 + executing_count)) + w_tps*norm(tps_model) + w_eta*norm(1/eta_ms) + w_vram*norm(vram_free_bytes)`
  - Choose max score; on ties, fall back to stable tiebreak above.
- **SLO classes**: optionally bias interactive jobs to higher-TPS workers.

Data sources:
- Worker Prometheus metrics (pull or snapshot via pool-managerd).
- Ready/Capacity and Plan decision hints for ETA fields.

Recommended orchestrator metrics:
- `orch_model_tps_smoothed{worker_id, model_ref}` (gauge)
- `orch_placement_score{worker_id, model_ref}` (gauge)
- `orch_placements_latency_ms` (histogram)

---

## Starvation Freedom

Maintain a `last_assigned_at` for each worker. When multiple workers tie at step 2 or 3, pick the one least recently assigned. This yields eventual selection for all eligible workers over time.

---

## Errors & Gating

- If a misrouted request hits a draining worker, the worker returns `503 ADMISSION_CLOSED` — route elsewhere.
- If staging cannot fit VRAM after eviction attempts, return fast with a non-2xx (e.g., `507 VRAM_OOM`) and retry another worker.

---

## Metrics (Recommended)

Expose per-worker placement metrics to validate policy:
- `orch_placements_assigned_total{worker_id}`
- `orch_placements_loaded_path_total{worker_id}`
- `orch_inflight_gauge{worker_id}`
- `orch_time_since_last_assignment_ms{worker_id}` (gauge)

---

## Notes

- Pinning & overrides: respect consumer pin decisions first.
- Residency awareness: if multiple handles exist per worker (multi-model), still apply the same rules using the requested `handle`.
- Snapshot cadence: refresh `Ready`/`Capacity` at ~200–500ms; make per-request decisions from cached state to reduce RPC chatter.
