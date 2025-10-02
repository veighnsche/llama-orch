# Pool-Managerd API SPEC — Planning & Assistance (POOL-3xxx)

Status: Draft
Applies to: `bin/pool-managerd-crates/api/`
Conformance: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

Defines planning intents/decisions and assistance hooks used between orchestrator ↔ pool-managerd and pool-managerd ↔ worker-orcd. Focus areas:
- Interpreting worker plans and augmenting with pool-level actions
- Host RAM prefetch assistance
- Eviction coordination

Parent specs:
- `/.specs/35-worker-orcd-pool-managerd-contract.md` (§2.2 Capacity & Planning)
- `bin/worker-orcd-crates/api/.specs/00_api.md` (§2.1 Planning Enums)

---

## 1. Planning Enums (non-breaking)

Pool-managerd MAY accept an optional `intent` and return a structured `decision`. Wire types use serde adjacently-tagged enums (`type` discriminator). All fields are optional unless stated.

### 1.1 PlanIntent (request, optional)

- `PreferResident { model_ref }`
- `StageIfFits { model_ref }`
- `EvictThenStage { model_ref, target_bytes?: u64 }`
- `PrefetchToRAM { model_ref, ttl_ms?: u32 }`
- `EstimateOnly {}`
- `DrainCheck {}`

### 1.2 PlanDecision (response, optional)

- `RouteToWorker { worker_id, handle_id?: String, slots_free?: u32, executing_count?: u32, queue_depth?: u32, est_slot_eta_ms?: u32 }`
- `CommitOnWorker { worker_id, est_commit_eta_ms?: u32, can_execute_now?: bool }`
- `EvictThenCommitOnWorker { worker_id, shard_ids: [String], bytes_to_free: u64, est_eviction_eta_ms?: u32, est_commit_eta_ms?: u32, can_execute_now_after?: bool }`
- `PrefetchRecommended { node_id, path?: String, est_bytes?: u64, trigger?: { "type": "AfterJobs", "count": u32 } }`
- `NoFit { required: u64, available: u64, queue_depth?: u32 }`
- `Draining { worker_id }`
 - `PrefetchAndCommitSequence { node_id, worker_id, steps: ["PrefetchToRAM", "CommitOnWorker"], est_total_eta_ms?: u32 }`
 - `RouteToSiblingWorker { worker_id, reason: String }`
 - `MigrateHandle { from_worker: String, to_worker: String, handle_id: String, est_eta_ms?: u32 }`
 - `ScheduleMaintenance { worker_id, window_ms: u32 }`

---

## 2. Assistance Hooks (Background)

Pool-managerd MAY assist workers by prefetching to host RAM or preparing on-node cache files.

- Triggers:
  - `AfterJobs { count }`: begin prefetch after the worker reports `count` remaining active jobs for the target handle.
  - `Immediate {}`: start prefetch immediately.
- Constraints: MUST honor RAM/disk caps; MUST yield to hot-path Commit/Execute; cancellation on contention is allowed.

---

## 3. Examples mapping (user semantics)

- "Already loaded in VRAM" → `RouteToWorker { worker_id, handle_id, slots_free }`
- "Loaded but busy (x seconds)" → `RouteToWorker { executing_count, est_slot_eta_ms }`
- "No room; can evict; queue x" → `EvictThenCommitOnWorker { shard_ids, bytes_to_free, ... }`
- "Busy (x seconds) but can load now" → `CommitOnWorker { est_commit_eta_ms, can_execute_now: false }`
- "Busy (x seconds) and can evict then load" → `EvictThenCommitOnWorker { ... }`
- "Load and infer immediately" → `CommitOnWorker { can_execute_now: true, est_commit_eta_ms: 0 }` or `RouteToWorker { slots_free > 0 }`
- "Prefetch after 2 jobs" → `PrefetchRecommended { trigger: { "type": "AfterJobs", "count": 2 } }`
 - "Prefetch then commit on that node" → `PrefetchAndCommitSequence { node_id, worker_id, steps: ["PrefetchToRAM", "CommitOnWorker"] }`
 - "Pick the sibling worker instead due to locality/availability" → `RouteToSiblingWorker { worker_id, reason }`
 - "Move the hot handle to a less busy worker" → `MigrateHandle { from_worker, to_worker, handle_id }`
 - "Avoid this worker for now (planned drain)" → `ScheduleMaintenance { worker_id, window_ms }`

---

## 4. Refinement Opportunities

- Add per-node RAM cache policy (TTL, LRU, size-aware eviction)
- Share utilization hints with orchestrator for placement heuristics
- Define backoff/circuit-breaking when prefetch repeatedly fails
