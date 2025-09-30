# orchestratord Implementation Status Analysis

**Generated**: 2025-09-30  
**Purpose**: Comprehensive audit of what exists vs what TODO.md claims needs implementation

---

## Key Findings

### ✅ Already Implemented (Not in TODO)

#### 1. **Error Taxonomy - COMPLETE** 
- ✅ All required error codes exist in `contracts/api-types/src/generated.rs`:
  - `AdmissionReject`, `QueueFullDropLru`, `InvalidParams`, `PoolUnready`, `PoolUnavailable`
  - `ReplicaExhausted`, `DecodeTimeout`, `WorkerReset`, `Internal`, `DeadlineUnmet`
  - Plus: `ModelDeprecated`, `UntrustedArtifact`
- ✅ `ErrorEnvelope` struct has all advisory fields:
  - `code: ErrorKind`
  - `message: Option<String>`
  - `engine: Option<Engine>`
  - `retriable: Option<bool>`
  - `retry_after_ms: Option<i64>`
  - `policy_label: Option<String>`
- ✅ Domain error mapping in `src/domain/error.rs` implements `IntoResponse`
- ✅ Retry headers (`Retry-After`, `X-Backoff-Ms`) already emitted for queue full errors

**Status**: OC-CTRL-2030, OC-CTRL-2031, OC-CTRL-2032 are **DONE**

#### 2. **Correlation ID - COMPLETE**
- ✅ Middleware in `src/app/middleware.rs::correlation_id_layer()`:
  - Extracts from `X-Correlation-Id` header or generates UUIDv4
  - Attaches to request extensions
  - Adds to all responses
  - Works even on early auth failures
- ✅ Applied in router before auth layers

**Status**: OC-CTRL-2052 is **DONE** (just needs verification in SSE responses and logs)

#### 3. **HTTP/2 Support - PARTIALLY DONE**
- ✅ Environment variable `ORCHD_PREFER_H2` exists in `src/app/bootstrap.rs`
- ✅ Narration emitted when H2 preference set
- ⚠️ Actual HTTP/2 enablement needs Axum configuration
- ⚠️ Compression settings not configured

**Status**: OC-CTRL-2025 is **50% DONE** (env var exists, needs Axum config)

#### 4. **Metrics Infrastructure - COMPLETE**
- ✅ Full metrics system in `src/metrics.rs`:
  - Counters, gauges, histograms
  - Label support
  - Throttling for high-cardinality metrics
  - Pre-registration of common metrics
  - Prometheus text format export
- ✅ Pre-registered metrics include:
  - `tasks_enqueued_total`, `tasks_started_total`, `tasks_canceled_total`, `tasks_rejected_total`
  - `tokens_in_total`, `tokens_out_total`
  - `admission_backpressure_events_total`, `catalog_verifications_total`
  - `queue_depth`, `kv_cache_usage_ratio`, `gpu_utilization`, `vram_used_bytes`, `model_state`
  - `latency_first_token_ms`, `latency_decode_ms`
- ✅ Metrics endpoint at `/metrics`

**Status**: Metrics infrastructure is **DONE**, just needs more call sites

#### 5. **Catalog Integration - READY**
- ✅ `catalog-core` crate fully implemented:
  - `ModelRef::parse()` for hf:/file:/url: schemes
  - `FsCatalog` with JSON index
  - `CatalogStore` trait (get/put/set_state/list/delete)
  - `FileFetcher` for local files
  - `verify_digest()` function
  - Lifecycle states (Active/Retired)
- ✅ Catalog HTTP endpoints exist in `src/api/catalog.rs`:
  - `create_model`, `get_model`, `delete_model`, `verify_model`, `set_model_state`
- ⚠️ Not yet called from admission flow

**Status**: Infrastructure **DONE**, integration into admission **PENDING**

#### 6. **Orchestrator-Core Queue - EXISTS**
- ✅ `libs/orchestrator-core/src/queue.rs`:
  - `InMemoryQueue` with capacity and policy
  - Priority support (Interactive/Batch)
  - Enqueue/cancel operations
  - Policy: Reject or DropLru
- ✅ Already used in `src/api/data.rs` via `state.admission`

**Status**: Queue infrastructure **DONE**

#### 7. **PlacementOverrides in Contracts - EXISTS**
- ✅ `TaskRequest.placement: Option<PlacementOverrides>` in generated types
- ⚠️ Not yet enforced in admission logic

**Status**: Contract **DONE**, enforcement **PENDING**

---

## What Actually Needs Implementation

### High Priority 🔴

#### Control Plane
1. **Drain/Reload Logic** - Stubs exist, need real implementation
   - Files: `src/api/control.rs::drain_pool()`, `reload_pool()`
   
2. **Capabilities Enhancement** - Basic version exists, needs enrichment
   - File: `src/services/capabilities.rs`
   - Need: Per-engine metadata (version, sampler_profile, ctx_max, etc.)

#### Data Plane
3. **Replace Sentinel Validations** - Easy win
   - File: `src/api/data.rs::create_task()`
   - Replace: `if body.model_ref == "pool-unavailable"` with real checks

4. **Catalog Integration in Admission** - Infrastructure ready
   - File: `src/api/data.rs::create_task()`
   - Add: Call `catalog-core` to check model presence
   - Add: Provision policy decision

5. **Pin Override Enforcement** - Contract ready
   - File: `src/api/data.rs::create_task()`
   - Add: Check `body.placement.pin_pool_id`, route accordingly

6. **Real ETA Calculation** - Currently heuristic
   - File: `src/api/data.rs::create_task()`
   - Replace: `pos * 100` with pool throughput calculation

#### Streaming
7. **HTTP/2 Axum Configuration** - Env var exists
   - File: `src/app/bootstrap.rs::start_server()`
   - Add: Axum HTTP/2 config when `ORCHD_PREFER_H2=1`

8. **SSE Error Event Emission** - Structure exists
   - File: `src/services/streaming.rs`
   - Add: Emit `event: error` with ErrorEnvelope JSON on failures

9. **Structured Cancellation Token** - Currently polling
   - File: `src/services/streaming.rs`
   - Replace: Shared state polling with `CancellationToken`

#### Observability
10. **Add More Metrics Call Sites** - Infrastructure ready
    - Files: `src/api/data.rs`, `src/services/streaming.rs`
    - Add: Queue depth updates, reject/drop counters, error class counters

11. **Correlation ID in SSE** - Middleware done, SSE needs it
    - File: `src/api/data.rs::stream_task()`
    - Add: Include correlation ID in SSE response headers

12. **Correlation ID in Logs** - Middleware done, logging needs it
    - Files: All logging sites
    - Add: Extract from request extensions, include in structured logs

#### Budgets
13. **Budget Enforcement** - Session tracking exists
    - File: `src/api/data.rs::create_task()`
    - Add: Reject when budget exceeded (not just headers)

---

## Revised Priority Assessment

### Can Complete Quickly (< 1 hour each)

1. **Replace Sentinel Validations** (15 min)
   - Remove string sentinels, use real error types
   
2. **Add Correlation ID to SSE Headers** (10 min)
   - Extract from extensions, add to headers
   
3. **Emit More Metrics** (30 min)
   - Add queue_depth gauge updates
   - Add rejection/drop counters
   - Add error class counters

4. **HTTP/2 Configuration** (20 min)
   - Add Axum HTTP/2 config when env var set

5. **Pin Override Enforcement** (30 min)
   - Check placement field, route to specified pool

### Medium Effort (1-2 hours each)

6. **Catalog Integration in Admission** (90 min)
   - Call catalog to check model presence
   - Add provision policy logic
   
7. **SSE Error Event Emission** (60 min)
   - Detect failures, emit error events
   - Ensure stream termination after error

8. **Real ETA Calculation** (60 min)
   - Query pool metrics
   - Calculate based on throughput

### Larger Effort (2+ hours each)

9. **Drain/Reload Implementation** (2-3 hours)
   - Coordinate with pool-managerd
   - Handle in-flight tasks
   
10. **Structured Cancellation** (2 hours)
    - Replace polling with CancellationToken
    - Propagate to adapters

11. **Budget Enforcement** (2 hours)
    - Add admission-time checks
    - Surface in SSE metrics frames

---

## Recommendations

### Phase 1: Quick Wins (2-3 hours total)
1. Replace sentinel validations
2. Add correlation ID to SSE
3. Emit more metrics
4. HTTP/2 configuration
5. Pin override enforcement

### Phase 2: Core Functionality (4-5 hours total)
6. Catalog integration in admission
7. SSE error event emission
8. Real ETA calculation

### Phase 3: Advanced Features (6-8 hours total)
9. Drain/reload implementation
10. Structured cancellation
11. Budget enforcement

---

## Files That Need Updates

### Immediate (Phase 1)
- `src/api/data.rs` - Remove sentinels, add pin override, add metrics
- `src/app/bootstrap.rs` - HTTP/2 config
- `src/services/streaming.rs` - Add metrics

### Near-term (Phase 2)
- `src/api/data.rs` - Catalog integration, ETA calculation
- `src/services/streaming.rs` - Error event emission

### Later (Phase 3)
- `src/api/control.rs` - Drain/reload logic
- `src/services/streaming.rs` - Cancellation token
- `src/api/data.rs` - Budget enforcement

---

## Conclusion

**Good News**: 
- Error taxonomy is complete
- Correlation ID middleware is complete
- Metrics infrastructure is complete
- Catalog-core is ready to use
- Queue system is working

**Reality Check**:
- TODO.md overestimated what needs to be built
- Most "high priority" items are actually integration work, not new systems
- Can achieve significant progress in 2-3 hours with Phase 1
- Full completion is ~12-16 hours, not 4-6 hours

**Next Steps**:
1. Update TODO.md to mark completed items
2. Execute Phase 1 (quick wins)
3. Re-assess after Phase 1 completion
