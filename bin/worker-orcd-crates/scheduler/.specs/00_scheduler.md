# Scheduler SPEC — Single-Slot Job Scheduling (M0) (WORKER-4xxx)

**Status**: Draft  
**Applies to**: `bin/worker-orcd-crates/scheduler/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

This crate implements single-slot job scheduling for M0 (one job at a time per worker). Post-M0 will add continuous batching and multi-request scheduling.

**Parent spec**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

## 1. M0 Scope: Single-Slot Scheduling

### 1.1 Requirements

- [SCHEDULER-M0-001] Scheduler MUST accept only one job at a time (single-slot constraint).
- [SCHEDULER-M0-002] Scheduler MUST track job state: `Pending`, `Executing`, `Completed`, `Failed`.
- [SCHEDULER-M0-003] Scheduler MUST reject new jobs while a job is `Pending` or `Executing`.
- [SCHEDULER-M0-004] Scheduler MUST allow new jobs after previous job reaches `Completed` or `Failed` state.
- [SCHEDULER-M0-005] Scheduler MUST provide `clear()` method to reset state after job completion.

### 1.2 Job State Machine

```
Pending → Executing → Completed
                   ↘ Failed
```

**Transitions**:
- `schedule(job_id)` → `Pending`
- `mark_executing(job_id)` → `Executing`
- `mark_completed(job_id)` → `Completed`
- `mark_failed(job_id)` → `Failed`
- `clear()` → Reset (if Completed or Failed)

---

## 2. API

### 2.1 Schedule Job

```rust
pub fn schedule(&mut self, job_id: String) -> Result<()>
```

- MUST check `can_accept()` before accepting
- MUST return `SchedulerError::WorkerBusy` if slot occupied
- MUST set state to `Pending`

### 2.2 State Transitions

```rust
pub fn mark_executing(&mut self, job_id: &str) -> Result<()>
pub fn mark_completed(&mut self, job_id: &str) -> Result<()>
pub fn mark_failed(&mut self, job_id: &str) -> Result<()>
```

- MUST validate job_id matches current job
- MUST return `SchedulerError::JobNotFound` if mismatch
- MUST update state atomically

### 2.3 Query State

```rust
pub fn can_accept(&self) -> bool
pub fn get_state(&self, job_id: &str) -> Result<JobState>
```

- `can_accept()` returns true if no job or job is Completed/Failed
- `get_state()` returns current state for job_id

### 2.4 Clear Slot

```rust
pub fn clear(&mut self)
```

- MUST only clear if state is `Completed` or `Failed`
- MUST reset `current_job` to `None`
- MUST reset state to `Pending`

---

## 3. Post-M0 Enhancements

**Deferred to post-M0**:
- Continuous batching support
- Dynamic batch size optimization
- KV cache allocation planning
- Prefill/decode phase scheduling
- Multi-request batch scheduling

See: `ARCHITECTURE_CHANGE_PLAN.md` Phase 5 (production hardening)

---

## 4. Error Types

```rust
pub enum SchedulerError {
    WorkerBusy,
    JobNotFound(String),
}
```

---

## 5. Security Properties

- **TIER 3 Clippy configuration** (medium-importance)
- Warn: unwrap, expect, panic
- Simple state machine (minimal attack surface)
- No resource exhaustion (single slot)

---

## 6. Dependencies

**Crates used**:
- `thiserror` — Error types

---

## 7. Traceability

**Code**: `bin/worker-orcd-crates/scheduler/src/lib.rs`  
**Tests**: `bin/worker-orcd-crates/scheduler/tests/`  
**Parent**: `bin/worker-orcd/.specs/00_worker-orcd.md` (lifecycle management)
