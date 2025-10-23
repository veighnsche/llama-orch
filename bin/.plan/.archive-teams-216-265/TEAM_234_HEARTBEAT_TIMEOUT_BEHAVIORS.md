# TEAM-234: Heartbeat + Timeout Enforcer Behavior Inventory

**Date:** Oct 22, 2025  
**Crates:** `rbee-heartbeat` + `timeout-enforcer`  
**Complexity:** Medium  
**Status:** ✅ COMPLETE

// TEAM-234: Investigated

---

## Executive Summary

Heartbeat protocol for health monitoring (Worker → Hive → Queen) + timeout enforcement with visual countdown. Heartbeat uses trait abstractions for flexibility, timeout enforcer prevents hanging operations.

**Key Behaviors:**
- Three-tier heartbeat (Worker → Hive → Queen)
- Trait-based abstractions for registries/catalogs
- Aggregated heartbeats (Hive collects all workers)
- Hard timeout enforcement with narration
- SSE routing for timeout events

---

## 1. rbee-heartbeat

### 1.1 Architecture

**Three-Tier Heartbeat:**
```text
Worker → Hive: POST /v1/heartbeat (30s interval)
  Payload: { worker_id, timestamp, health_status }

Hive → Queen: POST /v1/heartbeat (15s interval)
  Payload: { hive_id, timestamp, workers: [...] }
  (aggregates ALL worker states from registry)
```

**Why Three-Tier:**
- Queen doesn't need to know about individual workers
- Hive aggregates worker state
- Reduces network traffic to queen
- Scales better (N hives, M workers per hive)

### 1.2 Module Structure

**Modules:**
- `types` - Payload types and enums
- `worker` - Worker → Hive heartbeat logic
- `hive` - Hive → Queen heartbeat logic
- `queen` - Queen heartbeat handler
- `hive_receiver` - Hive worker heartbeat receiver
- `queen_receiver` - Queen hive heartbeat receiver
- `traits` - Trait abstractions for receivers

### 1.3 Types

**WorkerHeartbeatPayload:**
```rust
pub struct WorkerHeartbeatPayload {
    pub worker_id: String,
    pub timestamp: chrono::DateTime<Utc>,
    pub health_status: HealthStatus,
}

pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}
```

**HiveHeartbeatPayload:**
```rust
pub struct HiveHeartbeatPayload {
    pub hive_id: String,
    pub timestamp: chrono::DateTime<Utc>,
    pub workers: Vec<WorkerState>,
}

pub struct WorkerState {
    pub worker_id: String,
    pub health_status: HealthStatus,
    pub last_seen: chrono::DateTime<Utc>,
}
```

### 1.4 Worker Heartbeat

**Config:**
```rust
pub struct WorkerHeartbeatConfig {
    worker_id: String,
    hive_url: String,
    interval: Duration,  // Default: 30s
}
```

**Start Task:**
```rust
pub fn start_worker_heartbeat_task(
    config: WorkerHeartbeatConfig
) -> tokio::task::JoinHandle<()>
```

**Behavior:**
- Spawns background task
- Sends heartbeat every 30s
- POST to `{hive_url}/v1/heartbeat`
- Retries on failure (with backoff)
- Runs until task is aborted

### 1.5 Hive Heartbeat

**Config:**
```rust
pub struct HiveHeartbeatConfig {
    hive_id: String,
    queen_url: String,
    interval: Duration,  // Default: 15s
}
```

**Trait Requirement:**
```rust
pub trait WorkerStateProvider: Send + Sync {
    fn get_all_worker_states(&self) -> Vec<WorkerState>;
}
```

**Start Task:**
```rust
pub fn start_hive_heartbeat_task<P: WorkerStateProvider + 'static>(
    config: HiveHeartbeatConfig,
    provider: Arc<P>,
) -> tokio::task::JoinHandle<()>
```

**Behavior:**
- Spawns background task
- Collects worker states from provider
- Sends aggregated heartbeat every 15s
- POST to `{queen_url}/v1/heartbeat`
- Includes ALL worker states
- Retries on failure (with backoff)

### 1.6 Queen Receiver

**Handler:**
```rust
pub async fn handle_hive_heartbeat<C, D>(
    catalog: Arc<C>,
    payload: HiveHeartbeatPayload,
    device_detector: Arc<D>,
) -> Result<HeartbeatAcknowledgement, HeartbeatError>
where
    C: HiveCatalog,
    D: DeviceDetector,
```

**Behavior:**
1. Validate hive exists in catalog
2. Update hive last_seen timestamp
3. Update hive status (based on workers)
4. Detect devices if needed
5. Return acknowledgement

**HeartbeatAcknowledgement:**
```rust
pub struct HeartbeatAcknowledgement {
    pub received: bool,
    pub hive_status: HiveStatus,
}
```

### 1.7 Hive Receiver

**Handler:**
```rust
pub async fn handle_worker_heartbeat<R>(
    registry: Arc<R>,
    payload: WorkerHeartbeatPayload,
) -> Result<HeartbeatResponse, HeartbeatError>
where
    R: WorkerRegistry,
```

**Behavior:**
1. Validate worker exists in registry
2. Update worker last_seen timestamp
3. Update worker health status
4. Return response

**HeartbeatResponse:**
```rust
pub struct HeartbeatResponse {
    pub acknowledged: bool,
}
```

### 1.8 Trait Abstractions

**HiveCatalog:**
```rust
pub trait HiveCatalog: Send + Sync {
    fn get_hive(&self, hive_id: &str) -> Result<HiveRecord, CatalogError>;
    fn update_hive_status(&self, hive_id: &str, status: HiveStatus) -> Result<(), CatalogError>;
    fn update_last_seen(&self, hive_id: &str) -> Result<(), CatalogError>;
}
```

**WorkerRegistry:**
```rust
pub trait WorkerRegistry: Send + Sync {
    fn get_worker(&self, worker_id: &str) -> Result<WorkerState, RegistryError>;
    fn update_worker_health(&self, worker_id: &str, health: HealthStatus) -> Result<(), RegistryError>;
    fn update_last_seen(&self, worker_id: &str) -> Result<(), RegistryError>;
}
```

**DeviceDetector:**
```rust
pub trait DeviceDetector: Send + Sync {
    fn detect_devices(&self) -> Result<DeviceResponse, String>;
}

pub struct DeviceResponse {
    pub gpus: Vec<GpuDevice>,
    pub cpus: Vec<CpuDevice>,
}
```

**Why Traits:**
- Decouple heartbeat logic from storage implementation
- Each binary can provide its own implementation
- Testable with mock implementations
- Flexible for future changes

---

## 2. timeout-enforcer

### 2.1 Core Type

```rust
pub struct TimeoutEnforcer {
    duration: Duration,
    label: Option<String>,
    show_countdown: bool,
    job_id: Option<String>,  // TEAM-207: For SSE routing
}
```

### 2.2 Builder Methods

**`new(duration)`**
- Create enforcer with timeout duration
- Countdown disabled by default

**`with_label(label)`**
- Set operation label (shown in narration)
- Example: "Starting queen-rbee"

**`with_job_id(job_id)`**
- Set job_id for SSE routing (CRITICAL)
- Without this, timeout events go to stdout only
- Required for server-side operations

**`with_countdown()`**
- Enable visual progress bar
- Disabled by default (interferes with narration)

**`silent()`**
- Disable countdown (default)
- Still emits narration

### 2.3 Enforcement

**`enforce<F, T>(future) -> Result<T>`**
- Wraps future with timeout
- Returns Ok(T) if completes in time
- Returns Err if timeout occurs
- Auto-disables countdown if stderr is not TTY

**Behavior:**
1. Emit start narration
2. Run future with timeout
3. If completes: return result
4. If timeout: emit error narration, return error

### 2.4 Narration Integration

**Actor:** `"timeout"` (7 chars, ≤10 limit)

**Actions:**
- `start` - Operation started with timeout
- `timeout` - Operation timed out

**Pattern (Silent Mode):**
```rust
// Start
NARRATE.action("start")
    .job_id(&job_id)  // ← CRITICAL for SSE routing
    .context(label)
    .context(total_secs)
    .human("⏱️  {0} (timeout: {1}s)")
    .emit();

// Timeout
NARRATE.action("timeout")
    .job_id(&job_id)  // ← CRITICAL for SSE routing
    .context(label)
    .context(total_secs)
    .human("❌ {0} TIMED OUT after {1}s")
    .error_kind("operation_timeout")
    .emit_error();
```

### 2.5 Countdown Mode

**Visual Feedback:**
- Progress bar fills up over time
- Shows elapsed/total seconds
- Spinner animation
- Message label

**Format:**
```
⠋ [████████████████████████████████████████] 15/30s - Starting queen-rbee
```

**Auto-Disable:**
- Disabled if stderr is not a TTY
- Prevents hangs when running via Command::output()
- Uses `atty::is(atty::Stream::Stderr)`

**Why Disabled by Default:**
- Interferes with narration output
- Narration start/end messages provide sufficient feedback
- Countdown only useful for interactive CLI

### 2.6 TEAM-207 Fix

**Problem:**
- TimeoutEnforcer narration lacked job_id
- Events never reached SSE streams
- Only went to stdout

**Solution:**
- Added optional `job_id` field
- Added `.with_job_id()` builder method
- Narration now includes job_id for SSE routing

**Usage Pattern:**
```rust
// Client-side (rbee-keeper): No job_id needed
TimeoutEnforcer::new(timeout).enforce(future).await

// Server-side (queen-rbee): job_id required for SSE
TimeoutEnforcer::new(timeout)
    .with_job_id(&job_id)  // ← Critical for SSE routing!
    .enforce(future).await
```

---

## 3. Integration Points

### 3.1 Used By

**rbee-heartbeat:**
- Worker heartbeat sender (4 imports)
- Hive heartbeat sender
- Queen heartbeat receiver
- Hive heartbeat receiver

**timeout-enforcer:**
- queen-rbee (hive-lifecycle operations)
- Usage: 2 imports in product code

**Usage Pattern (Heartbeat):**
```rust
// Worker
let config = WorkerHeartbeatConfig::new(
    "worker-123".to_string(),
    "http://localhost:8600".to_string(),
);
let handle = start_worker_heartbeat_task(config);

// Hive
let config = HiveHeartbeatConfig::new(
    "localhost".to_string(),
    "http://localhost:8080".to_string(),
);
let handle = start_hive_heartbeat_task(config, registry);
```

**Usage Pattern (Timeout):**
```rust
// With SSE routing
TimeoutEnforcer::new(Duration::from_secs(15))
    .with_label("Fetching capabilities")
    .with_job_id(&job_id)  // ← CRITICAL
    .enforce(fetch_capabilities(url)).await?;
```

---

## 4. Error Handling

### 4.1 Heartbeat Errors

**HeartbeatError:**
```rust
pub enum HeartbeatError {
    HiveNotFound(String),
    WorkerNotFound(String),
    DeviceDetection(String),
    CatalogError(String),
    RegistryError(String),
}
```

**HTTP Mapping:**
- HiveNotFound → 404 NOT_FOUND
- DeviceDetection → 500 INTERNAL_SERVER_ERROR
- Other → 500 INTERNAL_SERVER_ERROR

### 4.2 Timeout Errors

**Error Message:**
```
{label} timed out after {seconds} seconds - operation was hanging
```

**Error Kind:**
```rust
.error_kind("operation_timeout")
```

---

## 5. Test Coverage

### 5.1 Existing Tests (timeout-enforcer)

**Unit Tests:**
- ✅ Successful operation (completes in time)
- ✅ Timeout occurs (operation too slow)
- ✅ Operation failure (not timeout)

**Async Tests:**
- ✅ All tests use tokio::test

### 5.2 Test Gaps

**rbee-heartbeat:**
- ❌ Worker heartbeat task (background task)
- ❌ Hive heartbeat task (background task)
- ❌ Heartbeat retry logic (on failure)
- ❌ Aggregation logic (hive collects workers)
- ❌ Trait implementations (mock registries/catalogs)
- ❌ Staleness detection (worker hasn't sent heartbeat)
- ❌ Health status transitions
- ❌ Concurrent heartbeat handling

**timeout-enforcer:**
- ❌ Countdown mode (progress bar)
- ❌ TTY detection (auto-disable countdown)
- ❌ Job_id propagation (SSE routing)
- ❌ Concurrent timeout enforcement
- ❌ Very short timeouts (<1s)
- ❌ Very long timeouts (>60s)

---

## 6. Performance Characteristics

**Heartbeat:**
- Interval: 30s (worker), 15s (hive)
- HTTP request: ~10-50ms
- Aggregation: O(N) where N = worker count
- Memory: ~100 bytes per worker state

**Timeout:**
- Overhead: <1ms (tokio::time::timeout)
- Countdown: ~1ms per second (progress bar update)
- Narration: <1ms per event

---

## 7. Dependencies

**rbee-heartbeat:**
- `tokio` - Async runtime, background tasks
- `reqwest` - HTTP client
- `serde` + `serde_json` - Serialization
- `chrono` - Timestamps
- `tracing` - Logging
- `thiserror` - Error types
- `async-trait` - Async trait methods
- `observability-narration-core` - Narration

**timeout-enforcer:**
- `tokio` - Async runtime, timeout
- `anyhow` - Error handling
- `indicatif` - Progress bar
- `atty` - TTY detection
- `observability-narration-core` - Narration

---

## 8. Critical Behaviors Summary

**rbee-heartbeat:**
1. **Three-tier architecture** - Worker → Hive → Queen
2. **Aggregated heartbeats** - Hive collects all workers
3. **Trait abstractions** - Flexible for different implementations
4. **Background tasks** - Spawned with tokio::spawn
5. **Retry logic** - Continues on failure
6. **Staleness detection** - Based on last_seen timestamp

**timeout-enforcer:**
1. **Hard timeout enforcement** - Operation WILL fail after timeout
2. **SSE routing** - job_id REQUIRED for server-side operations
3. **Visual countdown** - Optional, disabled by default
4. **TTY detection** - Auto-disable countdown if not TTY
5. **Narration integration** - Start/timeout events
6. **Zero tolerance** - No hanging operations allowed

---

## 9. Design Patterns

**rbee-heartbeat:**
- **Pattern:** Observer + Aggregator
- **Observer:** Workers/hives send periodic updates
- **Aggregator:** Hive collects worker states
- **Traits:** Dependency injection for flexibility

**timeout-enforcer:**
- **Pattern:** Decorator + Builder
- **Decorator:** Wraps future with timeout
- **Builder:** Fluent API for configuration
- **Narration:** Observability integration

---

## 10. Future Enhancements

**rbee-heartbeat:**
- ❌ Adaptive intervals (based on health)
- ❌ Heartbeat history (for trends)
- ❌ Alert thresholds (missed heartbeats)
- ❌ Graceful degradation (partial failures)

**timeout-enforcer:**
- ❌ Adaptive timeouts (based on history)
- ❌ Timeout policies (retry, backoff)
- ❌ Timeout metrics (for monitoring)

---

**Handoff:** Ready for Phase 5 integration analysis  
**All Phase 4 Teams Complete:** ✅ TEAM-230, ✅ TEAM-231, ✅ TEAM-232, ✅ TEAM-233, ✅ TEAM-234
