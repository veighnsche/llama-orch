# TEAM-206: Hive Start Flow Narration Analysis

**Date**: 2025-10-22  
**Author**: TEAM-206  
**Scope**: Complete analysis of `./rbee hive start` flow with narration gaps, bugs, and improvements

---

## Executive Summary

Analyzed the entire `./rbee hive start` flow from rbee-keeper → queen-rbee → rbee-hive. Found **7 critical gaps** in narration, **3 bugs**, and **4 edge cases** that need immediate attention.

**Key Issues**:
1. ❌ **NO narration from hive during device detection** (CRITICAL GAP)
2. ❌ **Missing cache check narration in queen** (user can't tell if cached or fresh)
3. ❌ **No job_id propagation to hive** (breaks SSE routing)
4. 🐛 **Default port inconsistency** (9000 vs 8600 vs 8500)
5. 🐛 **No actual cache checking logic** (always fetches fresh)
6. 🐛 **Wrong crate name** (should use `rbee-hive-device-detection`, currently uses it but could be clearer)

---

## Flow Diagram: Current State

```
┌──────────────────────────────────────────────────────────────────────┐
│ 1. rbee-keeper (bin/00_rbee_keeper)                                  │
│    - submit_and_stream_job()                                         │
│    - ✅ NARRATION: job_submit, job_stream                           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                │ POST /v1/jobs {"operation": "HiveStart"}
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 2. queen-rbee (bin/10_queen_rbee)                                    │
│    - create_job() → job_id                                           │
│    - ✅ NARRATION: job_create                                       │
│    - execute_job() → route_operation()                               │
│    - ✅ NARRATION: route_job, hive_start                           │
│    - Check if hive already running                                   │
│    - ✅ NARRATION: hive_check                                       │
│    - Spawn hive daemon                                               │
│    - ✅ NARRATION: hive_spawn                                       │
│    - Wait for health check                                           │
│    - ✅ NARRATION: hive_health, hive_success                        │
│    - ❌ NO NARRATION: Check if caps cached (MISSING!)              │
│    - ✅ NARRATION: hive_caps (Fetching...)                         │
│    - fetch_hive_capabilities(&endpoint)                              │
│    - ❌ NO NARRATION: HTTP request being made (MISSING!)           │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                │ GET http://127.0.0.1:9000/capabilities
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 3. rbee-hive (bin/20_rbee_hive)                                      │
│    - ❌ NO NARRATION: Received capabilities request (MISSING!)      │
│    - get_capabilities() handler                                      │
│    - ❌ NO NARRATION: Starting device detection (MISSING!)          │
│    - rbee_hive_device_detection::detect_gpus()                       │
│    - ❌ NO NARRATION: Trying nvidia-smi (MISSING!)                  │
│    - ❌ NO NARRATION: Found/parsed X devices (MISSING!)             │
│    - ❌ NO NARRATION: Adding CPU-0 fallback (MISSING!)              │
│    - Returns JSON response                                           │
│    - ❌ NO NARRATION: Response sent (MISSING!)                      │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
                                │ JSON: {"devices": [...]}
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ 4. queen-rbee (continued)                                            │
│    - ✅ NARRATION: hive_caps_ok, hive_device (for each device)     │
│    - ✅ NARRATION: hive_cache (Updating cache...)                  │
│    - config.capabilities.update_hive()                               │
│    - ✅ NARRATION: hive_cache_saved or hive_cache_error           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Findings

### 1. CRITICAL: No Narration from Hive During Device Detection

**Location**: `bin/20_rbee_hive/src/main.rs:137-159`

**Problem**: When the hive receives `/capabilities` request, it performs device detection with ZERO narration visible to the user.

**Current Code**:
```rust
async fn get_capabilities() -> Json<CapabilitiesResponse> {
    // Detect GPUs
    let gpu_info = rbee_hive_device_detection::detect_gpus();
    
    let mut devices: Vec<HiveDevice> = gpu_info.devices.iter().map(|gpu| HiveDevice {
        // ... mapping ...
    }).collect();
    
    // Add CPU device (always available)
    devices.push(HiveDevice {
        id: "CPU-0".to_string(),
        // ...
    });
    
    Json(CapabilitiesResponse { devices })
}
```

**Missing Narration**:
- 📡 "Received capabilities request from queen"
- 🔍 "Detecting GPUs via nvidia-smi..."
- ✅ "Found 1 GPU(s)" or ℹ️ "No GPUs detected, using CPU only"
- 🖥️ "Adding CPU-0 as fallback device"
- 📤 "Sending capabilities response (2 devices)"

**Root Cause**: 
1. The hive doesn't have access to `job_id` (it's not passed in HTTP headers)
2. Device detection uses `tracing::debug!()` instead of narration
3. No awareness of the job context

**Impact**: User sees this output:
```
[qn-router ] hive_caps      : 📊 Fetching device capabilities...
[qn-router ] hive_caps_ok   : ✅ Discovered 1 device(s)
```

But has NO IDEA that the hive:
- Received the request
- Executed nvidia-smi
- Parsed the output
- Added CPU fallback
- Sent the response

---

### 2. CRITICAL: Missing Cache Check Narration in Queen

**Location**: `bin/10_queen_rbee/src/job_router.rs:547-556`

**Problem**: The queen ALWAYS fetches fresh capabilities but never:
1. Checks if capabilities are already cached
2. Narrates the cache check decision

**Current Code**:
```rust
// TEAM-196: Fetch and cache capabilities
let endpoint = format!("http://{}:{}", hive_config.hostname, hive_config.hive_port);

NARRATE
    .action("hive_caps").job_id(&job_id)
    .human("📊 Fetching device capabilities...")
    .emit();

match fetch_hive_capabilities(&endpoint).await {
    // ...
}
```

**Missing Logic & Narration**:
```rust
// SHOULD BE:
let endpoint = format!("http://{}:{}", hive_config.hostname, hive_config.hive_port);

// Check cache first
if state.config.capabilities.contains(&alias) {
    NARRATE
        .action("hive_cache_check")
        .job_id(&job_id)
        .human("💾 Checking capabilities cache...")
        .emit();
    
    let cached = state.config.capabilities.get(&alias);
    
    NARRATE
        .action("hive_cache_hit")
        .job_id(&job_id)
        .context(&cached.last_updated.to_string())
        .human("✅ Using cached capabilities (last updated: {})")
        .emit();
    
    // Display cached devices
    // ...
    return Ok(());
} else {
    NARRATE
        .action("hive_cache_miss")
        .job_id(&job_id)
        .human("ℹ️  No cached capabilities, fetching fresh...")
        .emit();
}

// Fetch fresh capabilities
NARRATE
    .action("hive_caps_fetch")
    .job_id(&job_id)
    .human("📊 Fetching device capabilities from hive...")
    .emit();
```

**Impact**: 
- User doesn't know if capabilities are cached or fresh
- No way to tell cache age
- Always makes unnecessary HTTP request

---

### 3. CRITICAL: No job_id Propagation to Hive

**Location**: `bin/10_queen_rbee/src/hive_client.rs:36-70`

**Problem**: When queen calls `fetch_hive_capabilities()`, it doesn't pass `job_id` in HTTP headers. The hive can't emit job-scoped narration.

**Current Code**:
```rust
pub async fn fetch_hive_capabilities(endpoint: &str) -> Result<Vec<DeviceInfo>> {
    let url = format!("{}/capabilities", endpoint);
    let response = reqwest::get(&url).await.context("Failed to connect to hive")?;
    // ...
}
```

**Should Be**:
```rust
pub async fn fetch_hive_capabilities(
    endpoint: &str,
    job_id: &str,  // ← ADD THIS
) -> Result<Vec<DeviceInfo>> {
    let url = format!("{}/capabilities", endpoint);
    
    let response = reqwest::Client::new()
        .get(&url)
        .header("X-Rbee-Job-Id", job_id)  // ← PASS job_id
        .send()
        .await
        .context("Failed to connect to hive")?;
    // ...
}
```

**Hive Handler Update**:
```rust
use axum::http::HeaderMap;

async fn get_capabilities(headers: HeaderMap) -> Json<CapabilitiesResponse> {
    // Extract job_id from headers
    let job_id = headers.get("x-rbee-job-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    
    if let Some(ref jid) = job_id {
        NARRATE
            .action("caps_request")
            .job_id(jid)
            .human("📡 Received capabilities request from queen")
            .emit();
    }
    
    // ... rest of detection ...
}
```

**Impact**: Without job_id, hive narration goes to wrong channel or is dropped entirely.

---

### 4. BUG: Default Port Inconsistency

**Locations**:
1. `bin/10_queen_rbee/src/job_router.rs:113` - `hive_port: 9000`
2. `bin/20_rbee_hive/src/main.rs:30` - `default_value = "8600"`
3. `bin/10_queen_rbee/src/main.rs` - Queen defaults to `8500`

**Problem**: Three different default ports!

**Fix**: Standardize to **9000** everywhere:
- Queen: 8500 ✅ (correct)
- Hive: 9000 ✅ (should be)
- Localhost entry: 9000 ✅ (already correct)

**Change Required**:
```rust
// bin/20_rbee_hive/src/main.rs:30
#[arg(short, long, default_value = "9000")]  // ← Change from 8600
port: u16,
```

---

### 5. BUG: No Actual Cache Checking Logic

**Location**: `bin/10_queen_rbee/src/job_router.rs:547-556`

**Problem**: Code always fetches fresh capabilities, never uses cache.

**Evidence**:
- Line 413: `state.config.capabilities.contains(&alias)` is only used in HiveUninstall
- Line 547-556: HiveStart directly calls `fetch_hive_capabilities()` without checking cache

**Fix**: Add cache checking logic (see Finding #2 above)

---

### 6. EDGE CASE: Stale Cache Handling

**Problem**: No mechanism to refresh stale cache. What if:
- GPU added/removed since last check?
- VRAM capacity changed?
- Cache is 1 week old?

**Solution**: Add cache expiry and refresh logic:
```rust
const CACHE_MAX_AGE_SECS: u64 = 3600; // 1 hour

if let Some(cached) = state.config.capabilities.get(&alias) {
    let age = SystemTime::now()
        .duration_since(cached.last_updated)
        .unwrap_or(Duration::from_secs(0));
    
    if age.as_secs() < CACHE_MAX_AGE_SECS {
        // Use cache
    } else {
        NARRATE
            .action("hive_cache_stale")
            .job_id(&job_id)
            .context(&format!("{}s old", age.as_secs()))
            .human("⚠️  Cache is stale ({}), refreshing...")
            .emit();
        // Fetch fresh
    }
}
```

---

### 7. EDGE CASE: HTTP Request Failure Handling

**Location**: `bin/10_queen_rbee/src/hive_client.rs:36-70`

**Problem**: Limited narration on HTTP failures.

**Current**:
```rust
let response = reqwest::get(&url).await.context("Failed to connect to hive")?;
```

**Better**:
```rust
let response = match reqwest::get(&url).await {
    Ok(r) => r,
    Err(e) => {
        NARRATE
            .action("hive_caps_conn_fail")
            .job_id(&job_id)
            .context(&e.to_string())
            .human("❌ Failed to connect to hive: {}")
            .emit();
        return Err(e.into());
    }
};
```

---

### 8. Device Detection Crate Verification

**Location**: `bin/25_rbee_hive_crates/device-detection/`

**Status**: ✅ CORRECT crate is being used

**Evidence**:
- `bin/20_rbee_hive/Cargo.toml:34` - `rbee-hive-device-detection = { path = "../25_rbee_hive_crates/device-detection" }`
- `bin/20_rbee_hive/src/main.rs:139` - `rbee_hive_device_detection::detect_gpus()`

**Note**: Crate uses `tracing::debug!()` and `tracing::info!()` instead of narration system. These logs don't appear in SSE stream.

---

## Proposed Narration Flow (Fixed)

```
[keeper    ] job_submit     : 📋 Job job-xxx submitted
[keeper    ] job_stream     : 📡 Streaming results...
[qn-router ] job_create     : Job job-xxx created
[job-exec  ] execute        : Executing job job-xxx
[qn-router ] route_job      : Executing operation: hive_start
[qn-router ] hive_start     : 🚀 Starting hive 'localhost'
[qn-router ] hive_check     : 📋 Checking if hive is already running...
[qn-router ] hive_spawn     : 🔧 Spawning hive daemon: target/debug/rbee-hive
[qn-router ] hive_health    : ⏳ Waiting for hive to be healthy...
[qn-router ] hive_success   : ✅ Hive 'localhost' started successfully on http://127.0.0.1:9000/health

# ← NEW: Cache checking
[qn-router ] hive_cache_check: 💾 Checking capabilities cache...
[qn-router ] hive_cache_miss: ℹ️  No cached capabilities, fetching fresh...

# ← NEW: HTTP request narration
[qn-router ] hive_caps_fetch: 📊 Fetching device capabilities from hive...
[qn-router ] hive_caps_http : 🌐 GET http://127.0.0.1:9000/capabilities

# ← NEW: Hive-side narration (WITH job_id!)
[hive      ] caps_request   : 📡 Received capabilities request from queen
[hive      ] caps_gpu_check : 🔍 Detecting GPUs via nvidia-smi...
[hive      ] caps_gpu_found : ✅ Found 0 GPU(s)
[hive      ] caps_cpu_add   : 🖥️  Adding CPU-0 as fallback device
[hive      ] caps_response  : 📤 Sending capabilities response (1 device)

# Existing (good)
[qn-router ] hive_caps_ok   : ✅ Discovered 1 device(s)
[qn-router ] hive_device    :   🖥️  CPU-0 - CPU
[qn-router ] hive_cache     : 💾 Updating capabilities cache...
[qn-router ] hive_cache_saved: ✅ Capabilities cached
[DONE]
[keeper    ] job_complete   : ✅ Complete
```

---

## Implementation Plan

### Phase 1: Quick Wins (<40 LOC) - IMMEDIATE

1. **Fix port mismatch** (1 line)
   - File: `bin/20_rbee_hive/src/main.rs:30`
   - Change: `default_value = "9000"`

2. **Add cache check narration** (15 lines)
   - File: `bin/10_queen_rbee/src/job_router.rs:547`
   - Add: Cache check before fetch

3. **Add HTTP request narration** (8 lines)
   - File: `bin/10_queen_rbee/src/job_router.rs:556`
   - Add: Narration before `fetch_hive_capabilities()`

4. **Add hive action constants** (6 lines)
   - File: `bin/20_rbee_hive/src/narration.rs:22`
   - Add: `ACTION_CAPS_*` constants

**Total: 30 lines** ✅

### Phase 2: job_id Propagation (Medium) - NEXT

1. Update `fetch_hive_capabilities()` signature to accept `job_id`
2. Add `X-Rbee-Job-Id` header to request
3. Update hive handler to extract header
4. Add narration in hive with job_id

**Complexity**: Medium (40-60 LOC)

### Phase 3: Hive Device Detection Narration (Large) - LATER

1. Add narration to `get_capabilities()` handler
2. Wrap device detection with narration
3. Add per-device narration
4. Handle GPU vs CPU detection paths

**Complexity**: High (80-100 LOC)

### Phase 4: Cache Logic (Medium) - LATER

1. Implement cache checking before fetch
2. Add cache age tracking
3. Add cache refresh on stale
4. Add cache invalidation API

**Complexity**: Medium (60-80 LOC)

---

## Additional Bugs Found

### BUG-1: Race Condition in Hive Health Check

**Location**: `bin/10_queen_rbee/src/job_router.rs:535-540`

**Problem**: Exponential backoff in health check can take too long if hive starts slowly.

**Current**:
```rust
for attempt in 1..=10 {
    tokio::time::sleep(tokio::time::Duration::from_millis(200 * attempt)).await;
    // ...
}
```

**Issue**: Sleeps BEFORE first check!
- Attempt 1: Sleep 200ms → check
- Attempt 2: Sleep 400ms → check
- Total: 11 seconds of waiting even if hive is ready after 100ms

**Fix**: Check first, THEN sleep:
```rust
for attempt in 1..=10 {
    if let Ok(response) = client.get(&health_url).send().await {
        if response.status().is_success() {
            // Success!
        }
    }
    
    if attempt < 10 {
        tokio::time::sleep(tokio::time::Duration::from_millis(200 * attempt)).await;
    }
}
```

### BUG-2: Silent nvidia-smi Failure

**Location**: `bin/25_rbee_hive_crates/device-detection/src/detection.rs:10-27`

**Problem**: If nvidia-smi fails, fallback is silent. User doesn't know WHY GPU detection failed.

**Fix**: Add detection method tracking and return in response:
```rust
pub struct GpuInfo {
    pub devices: Vec<GpuDevice>,
    pub available: bool,
    pub count: usize,
    pub detection_method: DetectionMethod,  // ← NEW
}

pub enum DetectionMethod {
    NvidiaSmi,
    CudaRuntime,
    None,
}
```

---

## Summary: Changes Required

| Priority | Item | LOC | File(s) | Complexity |
|----------|------|-----|---------|------------|
| 🔴 P0 | Fix port mismatch | 1 | `rbee-hive/src/main.rs` | Trivial |
| 🔴 P0 | Add cache check narration | 15 | `job_router.rs` | Low |
| 🔴 P0 | Add HTTP request narration | 8 | `job_router.rs` | Low |
| 🟡 P1 | job_id propagation | 50 | `hive_client.rs`, `main.rs` | Medium |
| 🟡 P1 | Hive device detection narration | 80 | `rbee-hive/src/main.rs` | High |
| 🟢 P2 | Cache logic implementation | 60 | `job_router.rs` | Medium |
| 🟢 P2 | Fix health check race | 5 | `job_router.rs` | Low |
| 🟢 P2 | Detection method tracking | 20 | `device-detection/` | Low |

**Total P0 (Quick Wins)**: 24 LOC ✅  
**Total P0+P1**: 154 LOC  
**Total All**: 239 LOC

---

## Recommendations

1. **Immediate**: Implement Phase 1 (Quick Wins) - 24 LOC, <30 mins
2. **This Session**: Implement job_id propagation (Phase 2) - Critical for proper narration routing
3. **Next Session**: Implement hive device detection narration (Phase 3)
4. **Future**: Add comprehensive cache logic (Phase 4)

---

## Testing Checklist

After implementing fixes, verify:

- [ ] Port 9000 is used consistently
- [ ] Cache check narration appears before fetch
- [ ] HTTP request narration shows URL being called
- [ ] Hive narration appears with correct job_id
- [ ] Device detection steps are visible in output
- [ ] GPU and CPU detection paths both narrate correctly
- [ ] Cache hit/miss logic works
- [ ] Stale cache is refreshed
- [ ] Health check doesn't sleep unnecessarily
- [ ] nvidia-smi failure reason is visible

---

**END OF ANALYSIS**
