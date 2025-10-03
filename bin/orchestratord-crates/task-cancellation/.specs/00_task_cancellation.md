# Task Cancellation SPEC — Cancellation Propagation (CANCEL-21xxx)

**Status**: Draft  
**Applies to**: `bin/orchestratord-crates/task-cancellation/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `task-cancellation` crate handles cancellation requests from clients and propagates them to workers. It tracks cancellable jobs and ensures clean cancellation.

**Why it exists:**
- Clients need ability to cancel long-running jobs
- Need to propagate cancellation through orchestrator → worker chain
- Track cancelled jobs for cleanup

**What it does:**
- Accept cancellation requests from clients
- Find which worker is executing the job
- Send cancellation request to worker
- Update job state to cancelled
- Handle cancellation cleanup
- Propagate cancellation on client disconnects and on orchestrator-enforced timeouts

**What it does NOT do:**
- ❌ Execute cancellation on worker (worker does this)
- ❌ Stream events (streaming crate does this)
- ❌ Make retry decisions (scheduling does this)

---

## 1. Core Responsibilities

### [CANCEL-21001] Cancellation Request
The crate MUST accept cancellation requests from clients.

### [CANCEL-21002] Worker Lookup
The crate MUST find which worker is executing the job.

### [CANCEL-21003] Cancellation Propagation
The crate MUST send cancellation request to worker.

### [CANCEL-21004] State Update
The crate MUST update job state to cancelled.

### [CANCEL-21005] Idempotency
Cancellation MUST be idempotent. Repeated cancellations for the same `job_id` MUST be safe and yield the same terminal outcome.

---

## 2. Cancellation Flow

### [CANCEL-21010] Client Request
Client cancels job:

`POST /v2/tasks/{job_id}/cancel`

Response (idempotent):
```json
{
  "job_id": "job-xyz",
  "status": "cancelling"
}
```
Notes:
- Return 202 Accepted when cancellation is accepted and in progress; 200 OK if job already cancelled or completed (idempotent semantics).

### [CANCEL-21011] Orchestrator Handler
```rust
pub async fn cancel_job(&self, job_id: &str) -> Result<CancellationResult> {
    // 1. Find job
    let job = self.job_tracker.get_job(job_id)
        .ok_or(TaskError::JobNotFound)?;
    
    // 2. Check state
    match job.state {
        JobState::Queued => {
            // Remove from queue
            self.queue.remove(job_id);
            self.job_tracker.mark_cancelled(job_id);
            // Emit SSE error with code CANCELLED to any attached stream
            self.streaming.emit_error(job_id, "CANCELLED", "Client requested cancellation");
            return Ok(CancellationResult::Cancelled);
        }
        JobState::Executing => {
            // Propagate to worker
            self.cancel_on_worker(job_id, &job.worker_uri).await?;
            // Mark cancellation requested; stream layer will close with error=CANCELLED
            self.job_tracker.mark_cancel_requested(job_id);
            return Ok(CancellationResult::Accepted);
        }
        JobState::Completed | JobState::Failed => {
            return Err(TaskError::AlreadyCompleted);
        }
        JobState::Cancelled => {
            return Err(TaskError::AlreadyCancelled);
        }
    }
}
```

---

## 3. Worker Cancellation

### [CANCEL-21020] Cancel on Worker
Send cancel request to worker:
```rust
async fn cancel_on_worker(&self, job_id: &str, worker_uri: &str) -> Result<()> {
    let cancel_url = format!("{}/cancel", worker_uri);
    
    let response = reqwest::post(&cancel_url)
        .json(&json!({ "job_id": job_id }))
        .timeout(Duration::from_secs(5)) // cancellation deadline
        .send()
        .await?;
    
    if !response.status().is_success() {
        tracing::warn!(
            job_id = %job_id,
            worker_uri = %worker_uri,
            status = %response.status(),
            "Worker cancel request failed"
        );
    }
    
    Ok(())
}
```

### [CANCEL-21021] Worker Cancel Timeout
If worker doesn't respond within 5s (deadline):
- Log warning
- Treat cancellation as accepted and proceed to close client stream with SSE `error` (code `CANCELLED`)
- Best effort cleanup: worker may continue briefly; pool-managerd/worker will free resources upon detection

---

## 4. Cancellation States

### [CANCEL-21030] Job States
```rust
pub enum JobState {
    Queued,              // Can cancel immediately (remove from queue)
    Dispatched,          // Can cancel (worker hasn't started yet)
    Executing,           // Cancel propagates to worker
    CancelRequested,     // Cancel requested, awaiting worker/stream closure
    Cancelled,           // Cancellation complete
    Completed,      // Cannot cancel (already done)
    Failed,         // Cannot cancel (already failed)
}
```

### [CANCEL-21031] Cancellation Types
```rust
pub enum CancellationResult {
    Accepted,           // Propagation issued; stream will close with error=CANCELLED
    Cancelled,          // Successfully cancelled
    AlreadyCompleted,   // Job already done
    AlreadyCancelled,   // Already cancelled
    NotFound,           // Job doesn't exist
}
```

---

## 5. Queue Cancellation

### [CANCEL-21040] Remove from Queue
If job is queued but not dispatched:
```rust
fn cancel_queued_job(&mut self, job_id: &str) -> Result<()> {
    // Remove from queue
    self.queue.remove(job_id)?;
    
    // Update job state
    self.job_tracker.mark_cancelled(job_id);
    
    // Emit SSE error event to client (if connected)
    self.streaming.emit_error(job_id, "CANCELLED", "Client requested cancellation");
    
    Ok(())
}
```

---

## 6. Cleanup

### [CANCEL-21050] Post-Cancellation Cleanup
After cancellation:
```rust
async fn cleanup_cancelled_job(&self, job_id: &str) {
    // 1. Close SSE stream (if active)
    self.streams.close_stream(job_id);
    
    // 2. Mark worker as available again
    if let Some(worker_id) = self.job_tracker.get_worker(job_id) {
        self.worker_registry.mark_available(&worker_id);
    }
    
    // 3. Emit metrics
    self.metrics.tasks_cancelled_total.inc();
    
    // 4. Log cancellation
    tracing::info!(job_id = %job_id, "Job cancelled");
}
```

---

## 7. Timeout vs Cancellation

### [CANCEL-21060] Distinction
- **Cancellation**: Client explicitly requests cancel
**Timeout**: Orchestrator enforces timeout (job took too long)

Both use the same mechanism and MUST result in SSE `error` events with stable codes:
- `CANCELLED` (client request or disconnect)
- `TIMEOUT` (orchestrator-enforced deadline)

---

## 8. Error Handling
```rust
pub enum CancellationError {
    JobNotFound(String),
    AlreadyCompleted,
    AlreadyCancelled,
    WorkerUnreachable(String),
    PropagationFailed(String),
}
```

### [CANCEL-21071] Best Effort
Cancellation is best effort:
- If worker unreachable by deadline, treat as cancelled and close SSE with `error` code `CANCELLED`
- Worker/Pool-managerd will detect and free resources eventually
- Log failures for observability and emit metrics

---

## 9. Metrics

### [CANCEL-21080] Metrics
```rust
pub struct CancellationMetrics {
    pub cancel_requests_total: Counter,
    pub cancellations_successful_total: Counter,
    pub cancellations_failed_total: Counter,
    pub cancel_propagation_latency_ms: Histogram,
}
```

Additional labels and guidance:
- `cancel_requests_total{reason}` where reason ∈ { "client_request", "disconnect", "timeout", "admin" }
- `cancel_propagation_latency_ms` measures time from accept → SSE closed; target < 5s default

---

## 10. Dependencies

### [CANCEL-21090] Required Crates
```toml
[dependencies]
tokio = { workspace = true }
reqwest = { workspace = true }
serde = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
tracing = { workspace = true }
thiserror = { workspace = true }
```

---

## 11. Traceability

**Code**: `bin/orchestratord-crates/task-cancellation/src/`  
**Tests**: `bin/orchestratord-crates/task-cancellation/tests/`  
**Parent**: `bin/orchestratord/.specs/00_orchestratord.md`  
**Used by**: `orchestratord`, `agentic-api`  
**Spec IDs**: CANCEL-21001 to CANCEL-21090

---

**End of Specification**
