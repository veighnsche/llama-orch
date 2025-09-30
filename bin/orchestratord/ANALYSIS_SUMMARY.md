# orchestratord Codebase Analysis Summary

**Date**: 2025-09-30  
**Analyst**: AI Assistant  
**Purpose**: Comprehensive audit after completing Owner F's work

---

## Executive Summary

After deep-diving into the entire codebase, I discovered that **orchestratord is significantly more complete than the initial TODO suggested**. Many foundational systems are fully implemented and just need integration/wiring.

### Key Discoveries

✅ **Error Taxonomy**: 100% complete  
✅ **Correlation ID Middleware**: 100% complete  
✅ **Metrics Infrastructure**: 100% complete  
✅ **Catalog System**: 100% complete (ready for integration)  
✅ **Queue System**: 100% complete (already integrated)  
⚠️ **HTTP/2 Support**: 50% complete (env var exists, needs Axum config)  

---

## What's Actually Done

### 1. Error Handling (Complete)

**Location**: `contracts/api-types/src/generated.rs`, `src/domain/error.rs`

All required error codes exist:
- `AdmissionReject`, `QueueFullDropLru`, `InvalidParams`
- `PoolUnready`, `PoolUnavailable`, `ReplicaExhausted`
- `DecodeTimeout`, `WorkerReset`, `Internal`, `DeadlineUnmet`
- Plus: `ModelDeprecated`, `UntrustedArtifact`

`ErrorEnvelope` has all advisory fields:
- `code: ErrorKind`
- `message: Option<String>`
- `engine: Option<Engine>`
- `retriable: Option<bool>`
- `retry_after_ms: Option<i64>`
- `policy_label: Option<String>`

Retry headers (`Retry-After`, `X-Backoff-Ms`) are emitted for queue full errors.

**Status**: ✅ **OC-CTRL-2030, OC-CTRL-2031, OC-CTRL-2032 are DONE**

### 2. Correlation ID (Complete)

**Location**: `src/app/middleware.rs::correlation_id_layer()`

- Extracts from `X-Correlation-Id` header or generates UUIDv4
- Attached to request extensions for handler access
- Added to all responses (including early auth failures)
- Applied before auth layers in router

**Remaining**: 
- Add to SSE response headers (5 min)
- Extract from extensions in logging sites (15 min)

**Status**: ✅ **OC-CTRL-2052 is 90% DONE**

### 3. Metrics System (Complete)

**Location**: `src/metrics.rs`

Full implementation with:
- Counters, gauges, histograms
- Label support with proper formatting
- Throttling for high-cardinality metrics (`observe_histogram_throttled`)
- Pre-registration of common metrics
- Prometheus text format export

Pre-registered metrics:
- **Counters**: `tasks_enqueued_total`, `tasks_started_total`, `tasks_canceled_total`, `tasks_rejected_total`, `tokens_in_total`, `tokens_out_total`, `admission_backpressure_events_total`, `catalog_verifications_total`
- **Gauges**: `queue_depth`, `kv_cache_usage_ratio`, `gpu_utilization`, `vram_used_bytes`, `model_state`
- **Histograms**: `latency_first_token_ms`, `latency_decode_ms`

**Remaining**: Add more call sites (30 min)

**Status**: ✅ **Infrastructure DONE, integration PENDING**

### 4. Catalog System (Complete)

**Location**: `libs/catalog-core/src/lib.rs`

Fully implemented:
- `ModelRef::parse()` for `hf:`, `file:`, URL schemes
- `FsCatalog` with JSON index (get/put/set_state/list/delete)
- `CatalogStore` trait
- `FileFetcher` for local files
- `verify_digest()` function
- Lifecycle states (Active/Retired)
- HTTP endpoints in `src/api/catalog.rs`

**Remaining**: Call from admission flow (30 min)

**Status**: ✅ **Infrastructure DONE, integration PENDING**

### 5. Queue System (Complete & Integrated)

**Location**: `libs/orchestrator-core/src/queue.rs`

- `InMemoryQueue` with capacity and policy
- Priority support (Interactive/Batch)
- Enqueue/cancel operations
- Policy: Reject or DropLru
- Already used in `src/api/data.rs` via `state.admission`

**Status**: ✅ **DONE and INTEGRATED**

### 6. HTTP/2 Support (Partial)

**Location**: `src/app/bootstrap.rs`

- Environment variable `ORCHD_PREFER_H2` exists
- Narration emitted when preference set
- **Missing**: Actual Axum HTTP/2 configuration

**Remaining**: Configure Axum (20 min)

**Status**: ⚠️ **50% DONE**

### 7. PlacementOverrides (Contract Ready)

**Location**: `contracts/api-types/src/generated.rs`

- `TaskRequest.placement: Option<PlacementOverrides>` exists
- **Missing**: Enforcement in admission logic

**Remaining**: Check field and route to pinned pool (30 min)

**Status**: ⚠️ **Contract DONE, enforcement PENDING**

---

## What Actually Needs Work

### Quick Wins (< 1 hour total)

1. **Remove Sentinel Validations** (15 min)
   - File: `src/api/data.rs::create_task()`
   - Replace: `if body.model_ref == "pool-unavailable"` with real checks

2. **Add Correlation ID to SSE** (10 min)
   - File: `src/api/data.rs::stream_task()`
   - Extract from extensions, add to headers

3. **HTTP/2 Configuration** (20 min)
   - File: `src/app/bootstrap.rs::start_server()`
   - Configure Axum when `ORCHD_PREFER_H2=1`

4. **Pin Override Enforcement** (30 min)
   - File: `src/api/data.rs::create_task()`
   - Check `body.placement.pin_pool_id`, route accordingly

**Total**: ~75 minutes

### Medium Effort (2-3 hours total)

5. **Catalog Integration** (30 min)
   - File: `src/api/data.rs::create_task()`
   - Call `catalog-core` to check model presence

6. **Add Metrics Call Sites** (30 min)
   - Files: `src/api/data.rs`, `src/services/streaming.rs`
   - Queue depth updates, reject/drop counters

7. **SSE Error Event Emission** (60 min)
   - File: `src/services/streaming.rs`
   - Emit `event: error` with ErrorEnvelope on failures

8. **Real ETA Calculation** (60 min)
   - File: `src/api/data.rs::create_task()`
   - Replace `pos * 100` with pool throughput calculation

**Total**: ~3 hours

### Larger Effort (4+ hours)

9. **Drain/Reload Implementation** (2-3 hours)
10. **Structured Cancellation** (2 hours)
11. **Budget Enforcement** (2 hours)

**Total**: ~6-7 hours

---

## Recommended Execution Plan

### Phase 1: Quick Wins (1 hour)
Execute all 4 quick wins in sequence. High impact, low risk.

**Expected Outcome**: 
- Sentinel validations removed
- Correlation ID in SSE
- HTTP/2 enabled
- Pin override working

### Phase 2: Integration (3 hours)
Wire up existing infrastructure.

**Expected Outcome**:
- Catalog checks in admission
- Metrics coverage improved
- SSE error events working
- Better ETA calculation

### Phase 3: Advanced (6-7 hours)
Implement complex features.

**Expected Outcome**:
- Drain/reload operational
- Robust cancellation
- Budget enforcement

---

## Files Requiring Updates

### Phase 1 (Quick Wins)
- `src/api/data.rs` - Remove sentinels, add pin override
- `src/app/bootstrap.rs` - HTTP/2 config

### Phase 2 (Integration)
- `src/api/data.rs` - Catalog integration, ETA calculation, metrics
- `src/services/streaming.rs` - Error events, metrics

### Phase 3 (Advanced)
- `src/api/control.rs` - Drain/reload logic
- `src/services/streaming.rs` - Cancellation token
- `src/api/data.rs` - Budget enforcement

---

## Conclusion

The codebase is in **much better shape** than initially assessed:

- **Error taxonomy**: Complete ✅
- **Observability foundations**: Complete ✅  
- **Catalog system**: Complete ✅
- **Queue system**: Complete ✅
- **Metrics infrastructure**: Complete ✅

**Reality**: Most "high priority" work is **integration**, not **new development**.

**Estimate**: 
- Phase 1: 1 hour
- Phase 2: 3 hours  
- Phase 3: 6-7 hours
- **Total**: ~10-11 hours (not 4-6 hours, but also not 16+ hours)

**Next Step**: Execute Phase 1 (quick wins) to demonstrate rapid progress.
