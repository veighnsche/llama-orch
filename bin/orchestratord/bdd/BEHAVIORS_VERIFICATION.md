# BEHAVIORS.md Verification Report

**Date**: 2025-10-01  
**Verified Against**: `bin/orchestratord/src/` codebase  
**Status**: ⚠️ **PARTIALLY ACCURATE** - Multiple inaccuracies found

---

## Executive Summary

The `BEHAVIORS.md` document contains **200+ cataloged behaviors** but has **significant inaccuracies** in several sections. The document appears to be outdated and does not reflect the current implementation.

### Critical Findings

- ❌ **API Key Middleware (B-MW-010 to B-MW-014)**: Completely inaccurate - no X-API-Key middleware exists
- ❌ **Worker Registration Auth (B-CP-050 to B-CP-058)**: Partially inaccurate - uses different token env var
- ❌ **Background Services (B-BG-001 to B-BG-012)**: Not found in codebase
- ❌ **Configuration (B-CFG-003 to B-CFG-010)**: Several env vars not found or incorrect
- ✅ **Most other sections**: Generally accurate with minor discrepancies

---

## Detailed Findings by Section

### 1. Middleware Behaviors

#### ✅ Correlation ID Middleware (B-MW-001 to B-MW-004)
**Status**: ACCURATE

- Implementation: `src/app/middleware.rs`
- All behaviors verified:
  - Echoes `X-Correlation-Id` when present
  - Generates UUIDv4 when missing
  - Attaches to request extensions
  - Includes in all responses

#### ❌ API Key Middleware (B-MW-010 to B-MW-014)
**Status**: COMPLETELY INACCURATE

**Documented behaviors**:
- B-MW-010: Skip `/metrics` endpoint
- B-MW-011: No `X-API-Key` → 401
- B-MW-012: Invalid `X-API-Key` → 403
- B-MW-013: Valid `X-API-Key` == "valid" → allow
- B-MW-014: Early auth failures include correlation ID

**Reality**: 
- ❌ No `X-API-Key` middleware exists in the codebase
- ✅ Bearer token authentication exists instead (`src/app/auth_min.rs`)
- Uses `LLORCH_API_TOKEN` environment variable
- Uses `Authorization: Bearer <token>` header
- Timing-safe comparison with token fingerprinting
- `/metrics` endpoint is exempted from auth

**Recommendation**: Replace B-MW-010 to B-MW-014 with Bearer auth behaviors.

#### ✅ Bearer Identity Middleware (B-MW-020 to B-MW-022)
**Status**: ACCURATE

- Implementation: `src/app/auth_min.rs`
- Extracts Bearer token from Authorization header
- Attaches identity to request extensions
- Optional auth (allows requests without token when `LLORCH_API_TOKEN` not set)

---

### 2. Control Plane Behaviors

#### ✅ Capabilities Discovery (B-CP-001 to B-CP-004)
**Status**: ACCURATE

- Implementation: `src/api/control.rs::get_capabilities()`
- Caches capabilities snapshot
- Returns API version and engine metadata

#### ✅ Pool Health (B-CP-010 to B-CP-015)
**Status**: ACCURATE

- Implementation: `src/api/control.rs::get_pool_health()`
- Queries pool-managerd registry
- Returns health status with metrics
- Checks draining state

#### ✅ Pool Drain (B-CP-020 to B-CP-024)
**Status**: ACCURATE

- Implementation: `src/api/control.rs::drain_pool()`
- Marks pool as draining
- Returns 202 Accepted
- Stub implementation (doesn't wait or prevent admissions)

#### ✅ Pool Reload (B-CP-030 to B-CP-035)
**Status**: ACCURATE

- Implementation: `src/api/control.rs::reload_pool()`
- Test sentinel for "bad" model_ref → 409
- Updates model_state gauge metric
- Stub implementation

#### ✅ Pool Purge (B-CP-040 to B-CP-041)
**Status**: ACCURATE

- Implementation: `src/api/control.rs::purge_pool_v2()`
- Accepts untyped JSON body
- Returns 202 Accepted (stub)

#### ⚠️ Worker Registration (B-CP-050 to B-CP-058)
**Status**: PARTIALLY INACCURATE

**Documented**:
- B-CP-051: No Bearer token → 401 with `{code: 40101, message: "MISSING_TOKEN"}`
- B-CP-052: Invalid token → 401 with `{code: 40102, message: "BAD_TOKEN"}`
- B-CP-054: Uses `AUTH_TOKEN` env var

**Reality**:
- ✅ Requires `Authorization: Bearer <token>` header
- ❌ Uses `LLORCH_API_TOKEN` env var (not `AUTH_TOKEN`)
- ❌ Returns standard 401 Unauthorized (no custom error codes 40101/40102)
- ✅ Uses timing-safe comparison
- ✅ Logs identity breadcrumb
- ✅ Extracts pool_id and replica_id with defaults
- ✅ Binds MockAdapter when feature enabled
- ✅ Returns 200 OK with response body

**Recommendation**: Update B-CP-054 to reference `LLORCH_API_TOKEN` and B-CP-051/052 to reflect actual error responses.

---

### 3. Data Plane Behaviors

#### ⚠️ Task Admission (B-DP-001 to B-DP-035)
**Status**: MOSTLY ACCURATE with sentinel discrepancy

**Documented**:
- B-DP-005: Test sentinels removed
- B-DP-006: Test sentinels removed

**Reality**:
- ❌ Test sentinels are **still present** in `src/api/data.rs`:
  - Line 59: `if body.model_ref == "pool-unavailable"` → returns `ErrO::PoolUnavailable`
  - Line 62: `if body.prompt.as_deref() == Some("cause-internal")` → returns `ErrO::Internal`

**Other behaviors**: All other admission behaviors (B-DP-001 to B-DP-035) are accurate:
- Validation logic matches
- Queue admission logic matches
- Response building matches
- Error handling matches
- Metrics emission matches

**Recommendation**: Update B-DP-005 and B-DP-006 to reflect that sentinels are still present.

#### ✅ Task Streaming (B-DP-100 to B-DP-153)
**Status**: ACCURATE

- Implementation: `src/api/data.rs::stream_task()` and `src/services/streaming.rs`
- All documented behaviors verified
- Health-gated dispatch logic present
- SSE event generation matches
- Cancellation behaviors match
- Persistence behaviors match
- Deterministic fallback present

#### ✅ Task Cancellation (B-DP-200 to B-DP-206)
**Status**: ACCURATE

- Implementation: `src/api/data.rs::cancel_task()`
- All behaviors verified

---

### 4. Session Behaviors (B-SS-001 to B-SS-023)

**Status**: ✅ ACCURATE

- Implementation: `src/api/data.rs` and `src/services/session.rs`
- All behaviors verified
- Default TTL: 600,000ms
- In-memory storage
- Get/create/delete operations match

---

### 5. Catalog Behaviors (B-CAT-001 to B-CAT-043)

**Status**: ✅ ACCURATE

- Implementation: `src/api/catalog.rs`
- All CRUD operations match documented behaviors
- Uses `catalog_core::FsCatalog`
- Lifecycle state management matches
- Digest parsing matches

---

### 6. Artifact Behaviors (B-ART-001 to B-ART-023)

**Status**: ✅ ACCURATE

- Implementation: `src/api/artifacts.rs` and `src/services/artifacts.rs`
- Content-addressed storage with SHA-256
- In-memory and filesystem stores
- All behaviors verified

---

### 7. Streaming Behaviors (B-STR-001 to B-STR-033)

**Status**: ✅ ACCURATE

- Implementation: `src/services/streaming.rs`
- Health-gated dispatch logic present
- Adapter integration matches
- SSE encoding matches
- Transcript persistence matches

---

### 8. Observability Behaviors (B-OBS-001 to B-OBS-024)

**Status**: ✅ ACCURATE

- Implementation: `src/api/observability.rs` and `src/metrics.rs`
- Metrics endpoint returns Prometheus format
- All documented metrics pre-registered
- Logging and narration behaviors match

---

### 9. Background Service Behaviors (B-BG-001 to B-BG-024)

#### ❌ Handoff Autobind Watcher (B-BG-001 to B-BG-012)
**Status**: NOT FOUND IN CODEBASE

**Documented behaviors**:
- Watch `ORCHD_RUNTIME_DIR` directory
- Poll interval `ORCHD_HANDOFF_WATCH_INTERVAL_MS`
- Parse handoff JSON files
- Bind adapters automatically

**Reality**:
- ❌ No handoff watcher service found in codebase
- ❌ `ORCHD_RUNTIME_DIR` not referenced anywhere
- ❌ `ORCHD_HANDOFF_WATCH_INTERVAL_MS` not referenced anywhere
- ✅ Manual adapter binding exists in `src/app/bootstrap.rs` (lines 32-47) using `ORCHD_LLAMACPP_URL`

**Recommendation**: Either remove B-BG-001 to B-BG-012 or mark as "planned but not implemented".

#### ✅ Placement Service (B-BG-020 to B-BG-024)
**Status**: ACCURATE

- Implementation: `src/services/placement.rs` and `src/services/placement_v2.rs`
- Placement cache with TTL
- Stub implementation returns "default" pool

---

### 10. Error Handling Behaviors (B-ERR-001 to B-ERR-017)

**Status**: ✅ ACCURATE

- Error status code mapping matches
- Error envelope construction matches
- Retry-After headers present

---

### 11. Configuration Behaviors (B-CFG-001 to B-CFG-022)

#### ⚠️ Environment Variables (B-CFG-001 to B-CFG-010)
**Status**: PARTIALLY INACCURATE

**Verified accurate**:
- ✅ B-CFG-001: `ORCHD_ADMISSION_CAPACITY` (default: 8, not 16)
- ✅ B-CFG-002: `ORCHD_ADMISSION_POLICY`
- ✅ B-CFG-009: `ORCHD_ADDR` (default: "0.0.0.0:8080", not "127.0.0.1:8080")
- ✅ B-CFG-010: `ORCHD_PREFER_H2` (found in `src/app/bootstrap.rs`)

**Not found or inaccurate**:
- ❌ B-CFG-003: `ORCHD_LLAMACPP_URL` - exists but only when `llamacpp-adapter` feature enabled
- ❌ B-CFG-004: `ORCHD_LLAMACPP_POOL` - exists but only when `llamacpp-adapter` feature enabled
- ❌ B-CFG-005: `ORCHD_LLAMACPP_REPLICA` - exists but only when `llamacpp-adapter` feature enabled
- ❌ B-CFG-006: `ORCHD_RUNTIME_DIR` - NOT FOUND
- ❌ B-CFG-007: `ORCHD_HANDOFF_WATCH_INTERVAL_MS` - NOT FOUND
- ❌ B-CFG-008: `AUTH_TOKEN` - INCORRECT (should be `LLORCH_API_TOKEN`)

**Additional env vars not documented**:
- `ORCHESTRATORD_CLOUD_PROFILE` (enables cloud profile mode)
- `ORCHESTRATORD_PLACEMENT_STRATEGY` (round-robin, least-loaded, random)
- `ORCHESTRATORD_NODE_TIMEOUT_MS` (cloud profile)
- `ORCHESTRATORD_STALE_CHECK_INTERVAL_SECS` (cloud profile)
- `ORCHD_POOLS_CONFIG` (optional pools config file path)
- `ORCHD_SSE_MICROBATCH` (enables token batching in SSE)

**Recommendation**: Update configuration section to reflect actual implementation.

#### ✅ Feature Flags (B-CFG-020 to B-CFG-022)
**Status**: ACCURATE

- Verified in `Cargo.toml`:
  - `llamacpp-adapter`
  - `mock-adapters`
  - `metrics`

---

## Summary of Inaccuracies

### Critical Issues (Must Fix)

1. **API Key Middleware (B-MW-010 to B-MW-014)**: Completely wrong - Bearer auth is used instead
2. **Background Services (B-BG-001 to B-BG-012)**: Handoff watcher not implemented
3. **Configuration (B-CFG-006, B-CFG-007, B-CFG-008)**: Environment variables don't exist or are incorrect

### Minor Issues (Should Fix)

4. **Test Sentinels (B-DP-005, B-DP-006)**: Documented as removed but still present
5. **Worker Registration (B-CP-054)**: Wrong env var name
6. **Default Values**: Some defaults incorrect (e.g., admission capacity default is 8, not 16)

### Missing Documentation

7. **Cloud Profile Configuration**: Not documented but implemented
8. **Placement Strategy Configuration**: Not documented but implemented
9. **SSE Microbatch Configuration**: Not documented but implemented

---

## Recommendations

### Immediate Actions

1. **Replace API Key Middleware section** with Bearer Authentication behaviors
2. **Remove or mark as "not implemented"** the Handoff Autobind Watcher section
3. **Update configuration section** with correct environment variable names
4. **Fix test sentinel documentation** to reflect current state
5. **Add cloud profile behaviors** to the catalog

### Future Actions

6. **Add verification tests** that programmatically verify behaviors against code
7. **Establish process** to keep BEHAVIORS.md in sync with code changes
8. **Create traceability matrix** linking behaviors to test scenarios

---

## Verification Methodology

This report was generated by:
1. Reading the entire `BEHAVIORS.md` document
2. Examining all source files in `bin/orchestratord/src/`
3. Searching for referenced environment variables
4. Verifying middleware, API handlers, and service implementations
5. Cross-referencing with `Cargo.toml` for feature flags

**Files examined**: 30+ source files including all API handlers, middleware, services, and configuration modules.
