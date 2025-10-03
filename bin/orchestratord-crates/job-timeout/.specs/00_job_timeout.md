# Job Timeout SPEC — Timeout Enforcement (TIMEOUT-22xxx)

**Status**: Draft  
**Applies to**: `bin/orchestratord-crates/job-timeout/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

### Purpose

The `job-timeout` crate enforces timeout limits on jobs. It monitors job execution time and cancels jobs that exceed configured timeouts.

**Why it exists:**
- Prevent jobs from running forever (resource exhaustion)
- Enforce SLA/quality of service
- Detect hung workers

**What it does:**
- Track job execution time
- Detect jobs exceeding timeout
- Cancel timed-out jobs (via task-cancellation)
- Report timeout events

**What it does NOT do:**
- ❌ Execute cancellation (task-cancellation does this)
- ❌ Make timeout policy decisions (configured by orchestrator)
- ❌ Stream events (streaming does this)

---

## 1. Core Responsibilities

### [TIMEOUT-22001] Execution Time Tracking
The crate MUST track job execution time from dispatch to completion.

### [TIMEOUT-22002] Timeout Detection
The crate MUST detect jobs exceeding configured timeout.

### [TIMEOUT-22003] Timeout Enforcement
The crate MUST cancel jobs that exceed timeout.

### [TIMEOUT-22004] Timeout Reporting
The crate MUST report timeout events with metrics and logs.

---

## 2. Timeout Configuration

### [TIMEOUT-22010] Timeout Limits
```rust
pub struct TimeoutConfig {
    pub default_timeout_ms: u64,        // Default: 5 minutes
    pub max_timeout_ms: u64,            // Max allowed: 30 minutes
    pub per_model_timeouts: HashMap<String, u64>,  // Model-specific
}
```

### [TIMEOUT-22011] Timeout Sources
Timeout can come from:
1. **Request-level**: Client specifies timeout
2. **Model-level**: Model has default timeout
3. **Global default**: System default (5 minutes)

Priority: Request > Model > Global

---

## 3. Timeout Tracking

### [TIMEOUT-22020] Job Timer
Start timer when job dispatched:
```rust
pub fn start_timeout(&mut self, job_id: String, timeout_ms: u64) {
    let timeout_at = Utc::now() + Duration::from_millis(timeout_ms);
    
    self.timeouts.insert(job_id.clone(), TimeoutEntry {
        job_id,
        timeout_at,
        started_at: Utc::now(),
    });
    
    // Schedule timeout check
    let job_id_clone = job_id.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(timeout_ms)).await;
        self.check_timeout(job_id_clone).await;
    });
}
```

### [TIMEOUT-22021] Timeout Check Loop
Background task to check timeouts:
```rust
async fn timeout_check_loop(&mut self) {
    let mut interval = tokio::time::interval(Duration::from_secs(1));
    
    loop {
        interval.tick().await;
        
        let now = Utc::now();
        for entry in self.timeouts.values() {
            if now >= entry.timeout_at {
                self.handle_timeout(&entry.job_id).await;
            }
        }
    }
}
```

---

## 4. Timeout Enforcement

### [TIMEOUT-22030] Handle Timeout
When job times out:
```rust
async fn handle_timeout(&self, job_id: &str) -> Result<()> {
    // 1. Check if job still running
    let job = self.job_tracker.get_job(job_id)?;
    if job.state != JobState::Executing {
        return Ok(()); // Job already finished
    }
    
    // 2. Cancel via task-cancellation
    self.cancellation.cancel_job(job_id).await?;
    
    // 3. Update job state with timeout reason
    self.job_tracker.mark_failed(job_id, "TIMEOUT", "Job exceeded timeout");
    
    // 4. Emit timeout event
    if let Some(stream) = self.streams.get(job_id) {
        stream.send_event(SseEvent {
            event_type: "error",
            data: json!({
                "code": "TIMEOUT",
                "message": "Job exceeded timeout limit",
                "execution_time_ms": job.execution_time_ms(),
                "timeout_ms": entry.timeout_ms
            })
        });
    }
    
    // 5. Emit metrics
    self.metrics.timeouts_total.inc();
    
    // 6. Log timeout
    tracing::warn!(
        job_id = %job_id,
        execution_ms = job.execution_time_ms(),
        timeout_ms = entry.timeout_ms,
        "Job timed out"
    );
    
    Ok(())
}
```

---

## 5. Timeout Cancellation

### [TIMEOUT-22040] Stop Timer
When job completes before timeout:
```rust
pub fn cancel_timeout(&mut self, job_id: &str) {
    self.timeouts.remove(job_id);
}
```

### [TIMEOUT-22041] Cleanup
Remove timeout entry when:
- Job completes successfully
- Job fails
- Job cancelled by client

---

## 6. Timeout Stages

### [TIMEOUT-22050] Stage Timeouts
Different timeout stages:
```rust
pub struct StageTimeouts {
    pub queue_timeout_ms: Option<u64>,     // Max time in queue
    pub startup_timeout_ms: u64,           // Worker startup (60s)
    pub execution_timeout_ms: u64,         // Inference execution
    pub total_timeout_ms: u64,             // End-to-end
}
```

### [TIMEOUT-22051] Queue Timeout
If job sits in queue too long:
- Remove from queue
- Return error: `QUEUE_TIMEOUT`
- This is separate from execution timeout

### [TIMEOUT-22052] Startup Timeout
If worker doesn't start within timeout (60s):
- Cancel worker startup
- Return error: `STARTUP_TIMEOUT`
- Handled by worker-lifecycle

---

## 7. Timeout Metrics

### [TIMEOUT-22060] Metrics
```rust
pub struct TimeoutMetrics {
    pub timeouts_total: Counter,
    pub timeouts_by_stage: Counter,          // {stage: queue|startup|execution}
    pub execution_time_ms: Histogram,
    pub timeout_ratio: Gauge,                // timeouts / total_jobs
}
```

### [TIMEOUT-22061] Timeout Ratio
Track timeout ratio to detect issues:
- High timeout ratio → workers overloaded or config too strict
- Target: < 1% timeout ratio

---

## 8. Graceful Timeout

### [TIMEOUT-22070] Soft vs Hard Timeout
```rust
pub struct TimeoutPolicy {
    pub soft_timeout_ms: u64,    // Warn at 80% of timeout
    pub hard_timeout_ms: u64,    // Cancel at 100% of timeout
}
```

### [TIMEOUT-22071] Soft Timeout Warning
At 80% of timeout:
- Emit warning event (don't cancel yet)
- Log warning
- Give client heads-up

---

## 9. Error Handling

### [TIMEOUT-22080] Timeout Errors
```rust
pub enum TimeoutError {
    JobNotFound(String),
    AlreadyCompleted,
    CancellationFailed(String),
}
```

### [TIMEOUT-22081] Timeout Reporting
Include context in timeout errors:
- How long job ran
- Configured timeout
- Which stage timed out (queue, startup, execution)

---

## 10. Dependencies

### [TIMEOUT-22090] Required Crates
```toml
[dependencies]
task-cancellation = { path = "../task-cancellation" }
tokio = { workspace = true, features = ["time"] }
tracing = { workspace = true }
thiserror = { workspace = true }
serde = { workspace = true, features = ["derive"] }
chrono = { workspace = true }
```

---

## 11. Traceability

**Code**: `bin/orchestratord-crates/job-timeout/src/`  
**Tests**: `bin/orchestratord-crates/job-timeout/tests/`  
**Parent**: `bin/orchestratord/.specs/00_orchestratord.md`  
**Used by**: `orchestratord`, `scheduling`  
**Depends on**: `task-cancellation`  
**Spec IDs**: TIMEOUT-22001 to TIMEOUT-22090

---

**End of Specification**
