# TEAM-328: get.rs + status.rs + health.rs Consolidation Analysis

## Question
Can `get.rs`, `status.rs`, and `health.rs` be consolidated?

## Current State

### Three Files, Three Purposes

**1. health.rs (200 LOC)**
- `is_daemon_healthy()` - Simple bool check (no narration)
- `poll_until_healthy()` - Retry with exponential backoff (with narration)
- **Purpose:** HTTP health checking (technical)

**2. status.rs (97 LOC)**
- `check_daemon_status()` - Returns StatusResponse with narration
- **Purpose:** User-facing status check (business logic)

**3. get.rs (116 LOC)**
- `get_daemon()` - Generic trait-based instance retrieval
- `GettableConfig` trait
- **Purpose:** Configuration-based instance lookup (not HTTP)

---

## Analysis

### Overlap Matrix

|  | health.rs | status.rs | get.rs |
|---|---|---|---|
| **HTTP health check** | ✅ | ✅ | ❌ |
| **Returns bool** | ✅ | ❌ | ❌ |
| **Returns struct** | ❌ | ✅ | ✅ |
| **Narration** | poll only | ✅ | ✅ |
| **Retry logic** | poll only | ❌ | ❌ |
| **Config-based** | ❌ | ❌ | ✅ |
| **Trait-based** | ❌ | ❌ | ✅ |

### Usage Analysis

**health.rs - Used 5 times:**
- `rebuild.rs` - Check if daemon running
- `stop.rs` - Check if daemon running
- `shutdown.rs` - Check if daemon running
- `uninstall.rs` - Check if daemon running
- `poll_until_healthy()` - Internal retry loop

**status.rs - Used 2 times:**
- `rbee-keeper/handlers/queen.rs` - Queen status command
- `rbee-keeper/handlers/hive.rs` - Hive status command

**get.rs - Used 0 times:**
- ❌ **NO EXTERNAL CALLERS**
- Only used in examples/docs

---

## RULE ZERO Violations

### ❌ VIOLATION 1: status.rs vs health.rs

**Problem:** TWO ways to check daemon health via HTTP

**status.rs:**
```rust
pub async fn check_daemon_status(
    id: &str,
    health_url: &str,
    daemon_type: Option<&str>,
    job_id: Option<&str>,
) -> Result<StatusResponse> {
    // HTTP GET to health_url
    // Returns StatusResponse { is_running, ... }
}
```

**health.rs:**
```rust
pub async fn is_daemon_healthy(
    base_url: &str,
    health_endpoint: Option<&str>,
    timeout: Option<Duration>,
) -> bool {
    // HTTP GET to base_url + endpoint
    // Returns bool
}
```

**Why this is entropy:**
- Both do HTTP health checks
- Both check if daemon is running
- Different signatures for same purpose
- Callers must choose which one to use

**Current usage:**
- `check_daemon_status()` - User-facing commands (2 callers)
- `is_daemon_healthy()` - Internal checks (4 callers)

**Should be:**
- ONE function for HTTP health checks
- Different return types can be handled with conversion

---

### ❌ VIOLATION 2: get.rs is Unused

**Problem:** Entire file (116 LOC) has ZERO external callers

**Evidence:**
```bash
# Search for usage outside daemon-lifecycle
grep -r "get_daemon\|GettableConfig" bin/ --exclude-dir=daemon-lifecycle
# Result: NONE (only in examples/docs)
```

**Why this exists:**
- TEAM-211 created it for hive-lifecycle
- Never actually used
- Trait-based abstraction that wasn't needed
- 116 LOC of dead code

**Should be:**
- DELETE entire file
- If needed later, implement when actually needed

---

### ⚠️ POTENTIAL VIOLATION 3: status.rs Wraps health.rs

**Current implementation:**
```rust
// status.rs internally does:
let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?;
let running = match client.get(&health_url).send().await { ... }
```

**This duplicates health.rs logic!**

`is_daemon_healthy()` already does this:
```rust
let client = Client::builder().timeout(timeout).build()?;
match client.get(&url).send().await { ... }
```

**Why this is entropy:**
- HTTP health check logic duplicated
- Two places to maintain timeout logic
- Two places to fix bugs

---

## Recommended Consolidation

### Phase 1: Delete get.rs (IMMEDIATE)

**Action:** DELETE entire file (116 LOC)

**Reason:**
- Zero external callers
- Unused abstraction
- Pure dead code

**Impact:** NONE (nobody uses it)

**Files to modify:**
- Delete `src/get.rs`
- Remove from `src/lib.rs` exports:
  ```rust
  - pub use get::{get_daemon, GettableConfig};
  ```

---

### Phase 2: Consolidate status.rs into health.rs

**Problem:** `check_daemon_status()` duplicates HTTP logic from `is_daemon_healthy()`

**Solution:** Make `check_daemon_status()` use `is_daemon_healthy()` internally

**Before (status.rs):**
```rust
pub async fn check_daemon_status(...) -> Result<StatusResponse> {
    // Duplicate HTTP client logic
    let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?;
    let running = match client.get(&health_url).send().await { ... }
    
    Ok(StatusResponse {
        id: id.to_string(),
        is_running: running,
        health_status: Some(health_url.to_string()),
        metadata: None,
    })
}
```

**After (status.rs):**
```rust
pub async fn check_daemon_status(...) -> Result<StatusResponse> {
    // Use is_daemon_healthy() instead of duplicating logic
    let is_running = crate::health::is_daemon_healthy(
        health_url,
        None,
        Some(Duration::from_secs(5)),
    ).await;
    
    Ok(StatusResponse {
        id: id.to_string(),
        is_running,
        health_status: Some(health_url.to_string()),
        metadata: None,
    })
}
```

**Benefits:**
- Eliminates duplicate HTTP client logic
- Single source of truth for health checks
- Easier to maintain
- Easier to test

---

### Phase 3: Consider Merging status.rs into health.rs

**Current structure:**
- `health.rs` - Low-level HTTP health checks
- `status.rs` - High-level status with narration

**Could become:**
- `health.rs` - All health/status checking
  - `is_daemon_healthy()` - Simple bool
  - `poll_until_healthy()` - Retry loop
  - `check_daemon_status()` - StatusResponse with narration

**Benefits:**
- All health-related code in one place
- Clear hierarchy (simple → complex)
- Easier to find and maintain

**Tradeoff:**
- Larger file (200 + 97 = 297 LOC)
- But still reasonable size
- Better cohesion

---

## Implementation Plan

### Step 1: Delete get.rs ✅ SAFE
```bash
# Zero callers, pure dead code
rm bin/99_shared_crates/daemon-lifecycle/src/get.rs

# Update lib.rs
- pub use get::{get_daemon, GettableConfig};
```

**Impact:** NONE  
**Risk:** ZERO  
**LOC saved:** -116

---

### Step 2: Refactor status.rs to use health.rs ✅ SAFE
```rust
// In status.rs, replace HTTP client logic with:
let is_running = crate::health::is_daemon_healthy(
    health_url,
    None,
    Some(Duration::from_secs(5)),
).await;
```

**Impact:** Internal refactor only  
**Risk:** LOW (same behavior)  
**LOC saved:** ~20

---

### Step 3: Consider merging status.rs → health.rs ⚠️ OPTIONAL
```bash
# Move check_daemon_status() into health.rs
# Delete status.rs
# Update lib.rs exports
```

**Impact:** File organization  
**Risk:** LOW (just moving code)  
**LOC saved:** ~10 (file overhead)

---

## Summary

### RULE ZERO Violations Found

1. ❌ **get.rs is dead code** (116 LOC, zero callers)
2. ❌ **status.rs duplicates health.rs** (HTTP client logic)
3. ⚠️ **Two ways to check health** (but different purposes)

### Recommended Actions

**IMMEDIATE (Critical):**
1. ✅ DELETE get.rs (116 LOC saved, zero impact)
2. ✅ Refactor status.rs to use is_daemon_healthy() (eliminate duplication)

**OPTIONAL (Nice to have):**
3. 💡 Merge status.rs into health.rs (better organization)

### Expected Savings

- **Code deletion:** -116 LOC (get.rs)
- **Deduplication:** -20 LOC (status.rs refactor)
- **Optional merge:** -10 LOC (file overhead)
- **Total:** ~146 LOC reduction

### Risk Assessment

- **Delete get.rs:** ZERO risk (unused)
- **Refactor status.rs:** LOW risk (internal change)
- **Merge files:** LOW risk (organizational)

---

**TEAM-328 Assessment:** YES, consolidation needed - get.rs is dead code, status.rs duplicates health.rs
