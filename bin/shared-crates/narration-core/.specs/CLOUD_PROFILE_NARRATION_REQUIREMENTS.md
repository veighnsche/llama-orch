# Narration-Core Requirements for CLOUD_PROFILE

**Date**: 2025-09-30  
**Status**: URGENT - Required for v0.2.0  
**Related**: `.specs/01_cloud_profile.md`, `CLOUD_PROFILE_MIGRATION_PLAN.md`

---

## Executive Summary

The **CLOUD_PROFILE migration** (v0.2.0) introduces **distributed, multi-machine deployments** where services run on separate nodes. This **fundamentally changes observability requirements** for narration-core.

**Key Changes**:
1. **Distributed tracing is mandatory** (not optional)
2. **Correlation IDs must propagate across HTTP boundaries**
3. **Service identity (provenance) is critical** for debugging multi-node flows
4. **Structured logging must work across log aggregators**
5. **No filesystem access for log capture** (test adapter must be in-process)

---

## Cloud Profile Architecture Impact

### Service Topology

```
Control Plane Node (rbees-orcd)
    ↓ HTTP
GPU Worker Node 1 (pool-managerd + engine-provisioner + engines)
    ↓ HTTP
GPU Worker Node 2 (pool-managerd + engine-provisioner + engines)
    ↓ HTTP
GPU Worker Node N (pool-managerd + engine-provisioner + engines)
```

**Observability Challenge**: A single user request now touches **3+ services across 2+ machines**. Without proper narration, debugging is impossible.

### Example Flow

**User Request**: "Generate text with model X"

**Service Hops**:
1. Client → rbees-orcd (control plane)
2. rbees-orcd → pool-managerd (GPU worker 1) - health check
3. rbees-orcd → adapter → engine (GPU worker 1) - dispatch
4. engine → rbees-orcd → client - SSE stream

**Without Narration-Core**:
```
# rbees-orcd.log (machine 1)
{"level":"info","msg":"task created"}

# pool-managerd.log (machine 2)
println!("Spawning engine...")

# engine-provisioner.log (machine 2)
println!("Building llama.cpp")
```
**Problem**: No correlation. No way to trace request across machines.

**With Narration-Core** (as specified):
```
# rbees-orcd.log (machine 1)
{"level":"info","actor":"rbees-orcd","action":"admission","target":"session-abc123",
 "human":"Accepted request; queued at position 3 on pool 'default'",
 "correlation_id":"req-xyz","trace_id":"otel-trace-123","emitted_by":"rbees-orcd@0.2.0",
 "session_id":"session-abc123","pool_id":"default"}

# pool-managerd.log (machine 2)
{"level":"info","actor":"pool-managerd","action":"spawn","target":"GPU0",
 "human":"Spawning engine llamacpp-v1 for pool 'default' on GPU0",
 "correlation_id":"req-xyz","trace_id":"otel-trace-123","emitted_by":"pool-managerd@0.2.0",
 "pool_id":"default","replica_id":"r0"}

# engine-provisioner.log (machine 2)
{"level":"info","actor":"engine-provisioner","action":"build","target":"llamacpp-v1",
 "human":"Building llama.cpp with CUDA support for GPU0",
 "correlation_id":"req-xyz","trace_id":"otel-trace-123","emitted_by":"engine-provisioner@0.2.0",
 "engine":"llamacpp","device":"GPU0"}
```
**Solution**: Grep `correlation_id=req-xyz` across all machines → see complete flow.

---

## New Requirements for CLOUD_PROFILE

### 1. Distributed Tracing Integration (CRITICAL)

**Requirement**: Narration MUST integrate with OpenTelemetry traces.

**From `.specs/01_cloud_profile.md` (lines 67-75)**:
> **Rule**: All inter-service communication MUST be observable.
> 
> **Requirements**:
> - Distributed tracing (OpenTelemetry)
> - Structured logging with correlation IDs
> - Prometheus metrics for all HTTP calls
> - Health checks on all services

**Implementation**:
```rust
pub struct NarrationFields {
    // ... existing fields ...
    
    // NEW: OpenTelemetry integration
    pub trace_id: Option<String>,      // Already added ✅
    pub span_id: Option<String>,       // Already added ✅
    pub parent_span_id: Option<String>, // NEW - for span hierarchy
}

// Auto-extract from current OTEL context
pub fn narrate_with_otel_context(fields: NarrationFields) {
    let ctx = opentelemetry::Context::current();
    let span_ctx = ctx.span().span_context();
    
    let mut fields = fields;
    fields.trace_id = Some(span_ctx.trace_id().to_string());
    fields.span_id = Some(span_ctx.span_id().to_string());
    
    narrate(fields);
}
```

**Why Critical**: Without trace_id propagation, you cannot correlate logs across machines in distributed tracing UI (Tempo/Jaeger).

### 2. Service Identity (Provenance) is Mandatory

**Requirement**: Every narration MUST include `emitted_by` field.

**Rationale**: In distributed deployments, you need to know **which machine/service** emitted each log entry.

**From `.specs/01_cloud_profile.md` (lines 723-733)**:
```rust
tracing::info!(
    target: "rbees-orcd::dispatch",
    task_id = %task_id,
    pool_id = %pool_id,
    correlation_id = %corr_id,  // ← Already in spec!
    latency_ms = dispatch_latency,
    "task dispatched successfully"
);
```

**Narration-Core Must Provide**:
```rust
// Auto-inject service identity
pub fn narrate_auto(fields: NarrationFields) {
    let mut fields = fields;
    
    // Auto-detect service name and version
    fields.emitted_by = Some(format!(
        "{}@{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION")
    ));
    
    // Auto-inject timestamp
    fields.emitted_at_ms = Some(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    );
    
    narrate(fields);
}
```

**Why Mandatory**: When debugging across 10 GPU workers, you need to know "which pool-managerd on which machine" emitted the log.

### 3. HTTP Header Propagation

**Requirement**: Correlation IDs and trace context MUST propagate via HTTP headers.

**Implementation**:
```rust
// In rbees-orcd (client)
async fn poll_pool_status(pool_id: &str, correlation_id: &str) {
    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://gpu-worker-1:9200/v2/pools/{}/status", pool_id))
        .header("X-Correlation-Id", correlation_id)
        .header("X-Trace-Id", current_trace_id())
        .send()
        .await?;
    
    // Narrate the call
    narrate(NarrationFields {
        actor: "rbees-orcd",
        action: "pool_health_check",
        target: pool_id.to_string(),
        human: format!("Polling pool '{}' status", pool_id),
        correlation_id: Some(correlation_id.into()),
        trace_id: Some(current_trace_id()),
        ..Default::default()
    });
}

// In pool-managerd (server)
async fn get_pool_status(
    headers: HeaderMap,
    Path(pool_id): Path<String>,
) -> Result<Json<PoolStatus>> {
    // Extract correlation ID from header
    let correlation_id = headers
        .get("X-Correlation-Id")
        .and_then(|v| v.to_str().ok())
        .map(String::from);
    
    let trace_id = headers
        .get("X-Trace-Id")
        .and_then(|v| v.to_str().ok())
        .map(String::from);
    
    // Narrate with propagated IDs
    narrate(NarrationFields {
        actor: "pool-managerd",
        action: "get_status",
        target: pool_id.clone(),
        human: format!("Responding with status for pool '{}'", pool_id),
        correlation_id,
        trace_id,
        pool_id: Some(pool_id.clone()),
        ..Default::default()
    });
    
    // ... return status ...
}
```

**Why Critical**: This is how correlation works in distributed systems. Without header propagation, each service starts a new correlation context.

### 4. Log Aggregation Compatibility

**Requirement**: Narration output MUST be compatible with Loki/Elasticsearch/CloudWatch.

**From `.specs/01_cloud_profile.md` (line 736)**:
> **Log Aggregation**: Loki, Elasticsearch, CloudWatch

**Implementation Requirements**:
- ✅ JSON output (already supported via tracing)
- ✅ Consistent field names across services
- ✅ Timestamp in ISO 8601 or Unix millis
- ✅ Service labels for filtering
- ⚠️ **NEW**: Log level must be explicit field

**Example Output**:
```json
{
  "timestamp": "2025-09-30T22:46:33.123Z",
  "level": "info",
  "service": "rbees-orcd",
  "version": "0.2.0",
  "actor": "rbees-orcd",
  "action": "admission",
  "target": "session-abc123",
  "human": "Accepted request; queued at position 3",
  "correlation_id": "req-xyz",
  "trace_id": "otel-trace-123",
  "span_id": "span-456",
  "session_id": "session-abc123",
  "pool_id": "default",
  "queue_position": 3,
  "predicted_start_ms": 420
}
```

**Loki Query Example**:
```logql
{service="rbees-orcd"} |= "req-xyz" | json
```

**Elasticsearch Query Example**:
```json
{
  "query": {
    "bool": {
      "must": [
        {"term": {"correlation_id": "req-xyz"}},
        {"range": {"timestamp": {"gte": "now-1h"}}}
      ]
    }
  }
}
```

### 5. Test Capture Adapter Must Be In-Process

**Requirement**: BDD tests CANNOT rely on filesystem for log capture in CLOUD_PROFILE.

**Problem**: In Kubernetes/Docker, each service runs in a separate container. Tests cannot read logs from other containers' filesystems.

**Solution**: Test capture adapter MUST be in-process (already implemented ✅).

**Usage in BDD**:
```rust
#[given(regex = "^rbees-orcd and pool-managerd are running$")]
async fn given_services_running(world: &mut World) {
    // Install capture adapters in both services
    world.rbees-orcd_capture = CaptureAdapter::install();
    world.pool_managerd_capture = CaptureAdapter::install();
}

#[then(regex = "^narration shows request flow across services$")]
async fn then_narration_shows_flow(world: &mut World) {
    // Assert on rbees-orcd narration
    world.rbees-orcd_capture.assert_includes("Accepted request");
    world.rbees-orcd_capture.assert_field("correlation_id", "req-xyz");
    
    // Assert on pool-managerd narration
    world.pool_managerd_capture.assert_includes("Spawning engine");
    world.pool_managerd_capture.assert_field("correlation_id", "req-xyz");
    
    // Verify same correlation ID across services
    let orch_corr = world.rbees-orcd_capture.captured()[0].correlation_id.clone();
    let pool_corr = world.pool_managerd_capture.captured()[0].correlation_id.clone();
    assert_eq!(orch_corr, pool_corr, "Correlation ID must propagate across services");
}
```

---

## Updated Spec Requirements

### Add to `.specs/00_narration_core.md`

#### Section: Cloud Profile Requirements

```markdown
## Cloud Profile Requirements (v0.2.0+)

For distributed deployments (CLOUD_PROFILE), narration-core MUST:

1. **Integrate with OpenTelemetry**:
   - Extract `trace_id` and `span_id` from current context
   - Provide `narrate_with_otel_context()` helper
   - Support W3C Trace Context propagation

2. **Mandate Service Identity**:
   - `emitted_by` field is REQUIRED (not optional)
   - Auto-inject via `narrate_auto()` helper
   - Format: `{service_name}@{version}`

3. **Support HTTP Header Propagation**:
   - Provide helpers to extract correlation/trace IDs from HTTP headers
   - Provide helpers to inject correlation/trace IDs into HTTP headers
   - Compatible with `axum`, `reqwest`, `hyper`

4. **Log Aggregation Compatibility**:
   - JSON output with consistent field names
   - Explicit `level` field
   - ISO 8601 or Unix millis timestamps
   - Service labels for filtering

5. **Cross-Service Correlation**:
   - Correlation IDs MUST propagate across all HTTP calls
   - Trace IDs MUST link to OpenTelemetry spans
   - BDD tests MUST verify correlation across services
```

#### Section: Integration Points (Update)

```markdown
## Integration Points

### HOME_PROFILE (v0.1.x)
- Orchestratord: admission, placement, stream start/end/cancel hooks
- Adapter Host: submit/cancel wrappers (optional)
- Provisioners: preflight/build/spawn/readiness narration (optional)

### CLOUD_PROFILE (v0.2.0+)
- **Orchestratord** (control plane): admission, placement, pool polling, adapter binding, dispatch
- **pool-managerd** (GPU worker): handoff watcher, pool status, health checks
- **engine-provisioner** (GPU worker): preflight, build, spawn, CUDA checks
- **Adapter Host**: submit/cancel wrappers with correlation ID propagation
- **Worker Adapters**: streaming, errors, retries with trace context

**CRITICAL**: All services MUST use narration-core for cross-service correlation.
```

---

## Implementation Priority

### Phase 1: Core Features (Week 1) - BLOCKING v0.2.0

1. **OpenTelemetry Integration** (2 days)
   - Add `parent_span_id` field
   - Implement `narrate_with_otel_context()`
   - Unit tests

2. **Auto-Injection Helpers** (1 day)
   - Implement `narrate_auto()` (service identity + timestamp)
   - Macro for ergonomics: `narrate_auto!(...)`
   - Unit tests

3. **HTTP Header Helpers** (2 days)
   - `extract_correlation_from_headers()`
   - `inject_correlation_into_headers()`
   - Integration with `axum` and `reqwest`
   - Unit tests

### Phase 2: Cross-Service Adoption (Week 2) - BLOCKING v0.2.0

1. **rbees-orcd** (2 days)
   - Replace all `tracing::info!` with `narrate()`
   - Add correlation ID propagation to pool-managerd calls
   - Update BDD tests

2. **pool-managerd** (2 days)
   - Replace `println!` with `narrate()`
   - Extract correlation IDs from HTTP headers
   - Add narration to handoff watcher
   - Update BDD tests

3. **engine-provisioner** (1 day)
   - Replace `println!` with `narrate()`
   - Add narration to build/spawn flows

### Phase 3: Testing & Validation (Week 3) - BLOCKING v0.2.0

1. **BDD Cross-Service Tests** (2 days)
   - Test correlation ID propagation
   - Test trace ID propagation
   - Test service identity in logs

2. **E2E Distributed Tests** (2 days)
   - Deploy to 2-machine test environment
   - Verify log aggregation (Loki)
   - Verify distributed tracing (Tempo)

3. **Performance Testing** (1 day)
   - Measure narration overhead
   - Ensure <1ms per narration call
   - Load test with 1000 tasks/sec

---

## Risk Assessment

### HIGH RISK: Narration-Core Not Ready for v0.2.0

**Current State**: 95% unimplemented (only `human()` function exists)

**Required for v0.2.0**:
- ✅ Provenance fields (emitted_by, trace_id, etc.) - DONE
- ❌ OpenTelemetry integration - NOT STARTED
- ❌ HTTP header helpers - NOT STARTED
- ❌ Auto-injection helpers - NOT STARTED
- ❌ Cross-service adoption - NOT STARTED
- ❌ BDD cross-service tests - NOT STARTED

**Impact if Not Ready**:
- ❌ Cannot debug distributed deployments
- ❌ Cannot trace requests across services
- ❌ Cannot correlate logs in Loki/Elasticsearch
- ❌ Cannot link logs to OpenTelemetry traces
- ❌ v0.2.0 release blocked or shipped with poor observability

**Mitigation**:
1. **Prioritize narration-core completion** (3 weeks, as per URGENT_MEMO)
2. **Assign dedicated owner** for narration-core
3. **Make cross-service adoption mandatory** (not optional)
4. **Block v0.2.0 release** until narration coverage ≥80%

---

## Conclusion

**Narration-core is NOT optional for CLOUD_PROFILE**. It is **critical infrastructure** for debugging distributed deployments.

**Without it**:
- Debugging multi-service flows is impossible
- Incident response time increases 10x
- Production issues cannot be diagnosed
- Users lose trust in the platform

**With it**:
- Grep `correlation_id=X` → see complete request flow
- Click trace ID in logs → see OpenTelemetry trace
- Filter by `service=pool-managerd` → see all GPU worker activity
- Assert in BDD → verify observability coverage

**Action Required**:
1. Update `.specs/00_narration_core.md` with Cloud Profile requirements
2. Prioritize narration-core completion (3 weeks)
3. Make cross-service adoption mandatory
4. Add BDD tests for cross-service correlation
5. Block v0.2.0 release until complete

---

**Status**: URGENT - Required for v0.2.0  
**Owner**: TBD  
**Target Completion**: Before v0.2.0 release (5-6 weeks from now)
