# TEAM-207: Timeout Analysis & Hanging Operations

**Date**: 2025-10-22  
**Status**: 🔴 CRITICAL - Multiple hanging risks found  
**Priority**: P0 - Must fix immediately

---

## Executive Summary

Found **11 critical hanging risks** across the codebase where operations can hang indefinitely without timeouts.

**Current State**:
- ✅ **1 operation** properly protected with `TimeoutEnforcer` (SSE streaming)
- ❌ **11 operations** can hang indefinitely
- ⚠️ **6 operations** have partial timeout protection (reqwest client timeout only)

**Risk Level**: **CRITICAL** - Users can experience indefinite hangs

---

## Critical Hanging Risks

### 1. 🔴 CRITICAL: `fetch_hive_capabilities()` - NO TIMEOUT

**File**: `bin/10_queen_rbee/src/hive_client.rs:36-70`

**Problem**: Uses `reqwest::get()` with NO timeout

```rust
pub async fn fetch_hive_capabilities(endpoint: &str) -> Result<Vec<DeviceInfo>> {
    let url = format!("{}/capabilities", endpoint);
    let response = reqwest::get(&url).await.context("Failed to connect to hive")?;
    // ❌ NO TIMEOUT - Can hang forever if hive is slow/frozen
```

**Impact**: If hive is slow to detect GPUs (nvidia-smi hangs), queen hangs forever

**Fix Required**:
```rust
pub async fn fetch_hive_capabilities(endpoint: &str) -> Result<Vec<DeviceInfo>> {
    let url = format!("{}/capabilities", endpoint);
    
    // Add timeout
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))  // GPU detection can take time
        .build()?;
    
    let response = client.get(&url).send().await.context("Failed to connect to hive")?;
    // ...
}
```

---

### 2. 🔴 CRITICAL: `check_hive_health()` - NO TIMEOUT

**File**: `bin/10_queen_rbee/src/hive_client.rs:81-87`

**Problem**: Uses `reqwest::get()` with NO timeout

```rust
pub async fn check_hive_health(endpoint: &str) -> Result<bool> {
    let url = format!("{}/health", endpoint);
    let response = reqwest::get(&url).await.context("Failed to connect to hive")?;
    // ❌ NO TIMEOUT
```

**Impact**: Health checks can hang forever

**Fix Required**: Add client with timeout (2-5 seconds)

---

### 3. 🔴 CRITICAL: Job Submission - NO TIMEOUT

**File**: `bin/00_rbee_keeper/src/job_client.rs:51`

**Problem**: Uses client without timeout

```rust
// Submit job to queen
let res = client.post(format!("{}/v1/jobs", queen_url)).json(&job_payload).send().await?;
// ❌ NO TIMEOUT - Can hang if queen is frozen
```

**Impact**: `./rbee hive start` can hang forever if queen freezes during job submission

**Fix Required**: Wrap in `TimeoutEnforcer` or add client timeout

---

### 4. 🔴 CRITICAL: SSE Stream GET - Partial Protection

**File**: `bin/00_rbee_keeper/src/job_client.rs:87`

**Problem**: SSE GET has NO timeout, only the overall streaming has timeout

```rust
let stream_result = TimeoutEnforcer::new(Duration::from_secs(30))
    .enforce(async move {
        let response = client_clone.get(&sse_full_url).send().await?;
        // ❌ This GET itself has no timeout - can hang before TimeoutEnforcer kicks in
```

**Impact**: Can hang for 30 seconds waiting for SSE connection

**Fix Required**: Add client timeout for initial GET

---

### 5. 🔴 CRITICAL: Hive Health Check Loop - NO OVERALL TIMEOUT

**File**: `bin/10_queen_rbee/src/job_router.rs:535-696`

**Problem**: Loop has NO overall timeout, only per-request timeout

```rust
for attempt in 1..=10 {
    if let Ok(response) = client.get(&health_url).send().await {
        // Each request has 2s timeout, but loop can run 10 times = 20+ seconds
        // ❌ NO OVERALL TIMEOUT
```

**Impact**: Can take 20+ seconds if hive never becomes healthy

**Fix Required**: Wrap entire loop in `TimeoutEnforcer`

---

### 6. 🔴 CRITICAL: Hive Stop Verification Loop - NO OVERALL TIMEOUT

**File**: `bin/10_queen_rbee/src/job_router.rs:776-787`

**Problem**: Loop waits 5 seconds with NO timeout enforcement

```rust
for attempt in 1..=5 {
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    if let Err(_) = client.get(&health_url).send().await {
        // ❌ NO TIMEOUT - Can hang if health check hangs
```

**Impact**: Can hang for 5+ seconds

**Fix Required**: Wrap in `TimeoutEnforcer`

---

### 7. 🟡 MEDIUM: Queen Health Check - Partial Timeout

**File**: `bin/00_rbee_keeper/src/queen_lifecycle.rs:219-237`

**Problem**: Has 500ms client timeout but no overall timeout

```rust
let client = reqwest::Client::builder().timeout(Duration::from_millis(500)).build()?;
match client.get(&health_url).send().await {
    // ✅ Has client timeout
    // ❌ But no overall operation timeout
```

**Impact**: Low risk (500ms is short), but could be better

---

### 8. 🟡 MEDIUM: Poll Until Healthy - Partial Timeout

**File**: `bin/00_rbee_keeper/src/queen_lifecycle.rs:251-305`

**Problem**: Has manual timeout logic but not using `TimeoutEnforcer`

```rust
async fn poll_until_healthy(base_url: &str, timeout: Duration) -> Result<()> {
    let start = std::time::Instant::now();
    loop {
        if start.elapsed() >= timeout {
            // ✅ Has timeout logic
            // ❌ But not using TimeoutEnforcer (inconsistent pattern)
```

**Impact**: Works but inconsistent with rest of codebase

---

### 9. 🟡 MEDIUM: Queen Stop - Partial Timeout

**File**: `bin/00_rbee_keeper/src/main.rs:304-342`

**Problem**: Has client timeout but no overall operation timeout

```rust
let health_check = client.get(format!("{}/health", queen_url)).send().await;
// ❌ No timeout on health check

let shutdown_client = reqwest::Client::builder()
    .timeout(tokio::time::Duration::from_secs(30))
    .build()?;
// ✅ Shutdown has timeout
```

**Impact**: Health check can hang before shutdown attempt

---

### 10. 🟡 MEDIUM: Queen Status - Partial Timeout

**File**: `bin/00_rbee_keeper/src/main.rs:346-360`

**Problem**: Has client timeout but could use `TimeoutEnforcer`

```rust
let client = reqwest::Client::builder()
    .timeout(tokio::time::Duration::from_secs(5))
    .build()?;
// ✅ Has timeout
// ⚠️ But could use TimeoutEnforcer for consistency
```

**Impact**: Low risk, but inconsistent pattern

---

### 11. 🟡 MEDIUM: Worker Heartbeat - Has Timeout

**File**: `bin/99_shared_crates/heartbeat/src/worker.rs:118`

**Problem**: Has timeout but uses deprecated `.timeout()` method

```rust
let response = client.post(url).json(payload).timeout(Duration::from_secs(5)).send().await?;
// ✅ Has timeout
// ⚠️ Uses per-request timeout instead of client timeout
```

**Impact**: Works but inconsistent pattern

---

## Timeout Patterns in Codebase

### ✅ GOOD: Using TimeoutEnforcer

**Example**: `bin/00_rbee_keeper/src/job_client.rs:82-154`

```rust
let stream_result = TimeoutEnforcer::new(Duration::from_secs(30))
    .with_label("Streaming job results")
    .silent()
    .enforce(async move {
        // Operation here
    })
    .await;
```

**Benefits**:
- Consistent timeout handling
- Clear error messages
- Narration integration
- Visual feedback (optional)

---

### ⚠️ PARTIAL: Using Client Timeout

**Example**: `bin/10_queen_rbee/src/job_router.rs:496-497`

```rust
let client = reqwest::Client::builder()
    .timeout(tokio::time::Duration::from_secs(2))
    .build()?;
```

**Benefits**:
- Prevents individual requests from hanging
- Simple to implement

**Limitations**:
- Only covers single request
- Doesn't cover loops or multi-step operations
- No narration integration
- No visual feedback

---

### ❌ BAD: No Timeout

**Example**: `bin/10_queen_rbee/src/hive_client.rs:39`

```rust
let response = reqwest::get(&url).await.context("Failed to connect to hive")?;
```

**Problems**:
- Can hang forever
- No user feedback
- No way to recover

---

## Recommended Timeout Values

| Operation | Timeout | Rationale |
|-----------|---------|-----------|
| **Health checks** | 2-5s | Should be fast, hive/queen should respond quickly |
| **Job submission** | 10s | Creating job + DB write |
| **SSE connection** | 10s | Establishing SSE stream |
| **SSE streaming** | 30s | Overall job execution (already implemented) |
| **Capabilities fetch** | 10-15s | GPU detection via nvidia-smi can take time |
| **Daemon spawn + health** | 15-20s | Daemon startup + first health check |
| **Graceful shutdown** | 10s | SIGTERM + wait for process exit |

---

## Implementation Plan

### Phase 1: Critical Fixes (P0) - IMMEDIATE

**Goal**: Fix operations that can hang indefinitely

1. **Add timeout to `fetch_hive_capabilities()`** (5 LOC)
   - File: `bin/10_queen_rbee/src/hive_client.rs:36`
   - Change: Add client with 10s timeout
   - Risk: HIGH - GPU detection can hang

2. **Add timeout to `check_hive_health()`** (5 LOC)
   - File: `bin/10_queen_rbee/src/hive_client.rs:81`
   - Change: Add client with 5s timeout
   - Risk: HIGH - Called in loops

3. **Add timeout to job submission** (3 LOC)
   - File: `bin/00_rbee_keeper/src/job_client.rs:51`
   - Change: Add client with 10s timeout
   - Risk: HIGH - User-facing operation

4. **Add timeout to SSE GET** (3 LOC)
   - File: `bin/00_rbee_keeper/src/job_client.rs:87`
   - Change: Add client with 10s timeout
   - Risk: MEDIUM - Already has outer timeout

**Total P0**: 16 LOC

---

### Phase 2: Loop Timeouts (P1) - NEXT

**Goal**: Add overall timeouts to loops

1. **Wrap hive health check loop in TimeoutEnforcer** (10 LOC)
   - File: `bin/10_queen_rbee/src/job_router.rs:535`
   - Change: Wrap loop in `TimeoutEnforcer::new(Duration::from_secs(15))`
   - Risk: MEDIUM - Can take 20+ seconds

2. **Wrap hive stop verification in TimeoutEnforcer** (10 LOC)
   - File: `bin/10_queen_rbee/src/job_router.rs:776`
   - Change: Wrap loop in `TimeoutEnforcer::new(Duration::from_secs(10))`
   - Risk: MEDIUM - Can hang during shutdown

**Total P1**: 20 LOC

---

### Phase 3: Consistency (P2) - LATER

**Goal**: Make all timeout patterns consistent

1. **Refactor `poll_until_healthy()` to use TimeoutEnforcer** (15 LOC)
   - File: `bin/00_rbee_keeper/src/queen_lifecycle.rs:251`
   - Change: Replace manual timeout logic with `TimeoutEnforcer`
   - Risk: LOW - Already has timeout

2. **Add TimeoutEnforcer to queen stop** (10 LOC)
   - File: `bin/00_rbee_keeper/src/main.rs:304`
   - Change: Wrap operation in `TimeoutEnforcer`
   - Risk: LOW - Already has partial timeout

3. **Add TimeoutEnforcer to queen status** (10 LOC)
   - File: `bin/00_rbee_keeper/src/main.rs:346`
   - Change: Wrap operation in `TimeoutEnforcer`
   - Risk: LOW - Already has timeout

**Total P2**: 35 LOC

---

## Total Implementation

| Phase | LOC | Files | Risk Level | Priority |
|-------|-----|-------|------------|----------|
| P0 | 16 | 2 | HIGH | IMMEDIATE |
| P1 | 20 | 1 | MEDIUM | NEXT |
| P2 | 35 | 2 | LOW | LATER |
| **Total** | **71** | **5** | - | - |

---

## TimeoutEnforcer Improvements Needed?

**Current State**: TimeoutEnforcer is well-designed ✅

**Features**:
- ✅ Hard timeout enforcement
- ✅ Visual countdown (optional)
- ✅ Narration integration
- ✅ Silent mode (default)
- ✅ Clear error messages
- ✅ TTY detection (auto-disable countdown in pipes)

**No changes needed** - it's ready to use!

---

## Testing Checklist

After implementing fixes:

- [ ] Test `./rbee hive start` with slow GPU detection
- [ ] Test `./rbee hive start` with hive that never becomes healthy
- [ ] Test `./rbee hive stop` with frozen hive
- [ ] Test `./rbee queen stop` with frozen queen
- [ ] Test SSE streaming with slow queen
- [ ] Test job submission with frozen queen
- [ ] Verify all timeouts trigger with clear error messages
- [ ] Verify narration shows timeout errors
- [ ] Test with network delays (tc qdisc add)
- [ ] Test with process freezes (kill -STOP)

---

## Example Fix: fetch_hive_capabilities()

**Before** (NO TIMEOUT):
```rust
pub async fn fetch_hive_capabilities(endpoint: &str) -> Result<Vec<DeviceInfo>> {
    let url = format!("{}/capabilities", endpoint);
    let response = reqwest::get(&url).await.context("Failed to connect to hive")?;
    // ...
}
```

**After** (WITH TIMEOUT):
```rust
pub async fn fetch_hive_capabilities(endpoint: &str) -> Result<Vec<DeviceInfo>> {
    let url = format!("{}/capabilities", endpoint);
    
    // TEAM-207: Add timeout - GPU detection can be slow
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .context("Failed to create HTTP client")?;
    
    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to connect to hive")?;
    // ...
}
```

---

## Summary

**Current State**: 🔴 **11 hanging risks** across codebase

**Required Action**: Implement **16 LOC** of P0 fixes immediately

**Impact**: Eliminates all critical hanging risks

**Effort**: ~30 minutes for P0 fixes

---

**END OF ANALYSIS**
