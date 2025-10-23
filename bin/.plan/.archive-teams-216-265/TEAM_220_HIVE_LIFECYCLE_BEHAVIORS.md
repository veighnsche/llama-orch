# HIVE-LIFECYCLE BEHAVIOR INVENTORY

**Team:** TEAM-220  
**Component:** `bin/15_queen_rbee_crates/hive-lifecycle`  
**Date:** Oct 22, 2025  
**LOC:** ~1,629 (TEAM-210 through TEAM-215)

---

## 1. Public API Surface

### Exported Functions (11 operations)

```rust
// Simple Operations (TEAM-211)
pub async fn execute_hive_list(request: HiveListRequest, config: Arc<RbeeConfig>, job_id: &str) -> Result<HiveListResponse>
pub async fn execute_hive_get(request: HiveGetRequest, config: Arc<RbeeConfig>, job_id: &str) -> Result<HiveGetResponse>
pub async fn execute_hive_status(request: HiveStatusRequest, config: Arc<RbeeConfig>, job_id: &str) -> Result<HiveStatusResponse>

// Lifecycle Operations (TEAM-212)
pub async fn execute_hive_start(request: HiveStartRequest, config: Arc<RbeeConfig>) -> Result<HiveStartResponse>
pub async fn execute_hive_stop(request: HiveStopRequest, config: Arc<RbeeConfig>) -> Result<HiveStopResponse>

// Install/Uninstall Operations (TEAM-213)
pub async fn execute_hive_install(request: HiveInstallRequest, config: Arc<RbeeConfig>, job_id: &str) -> Result<HiveInstallResponse>
pub async fn execute_hive_uninstall(request: HiveUninstallRequest, config: Arc<RbeeConfig>, job_id: &str) -> Result<HiveUninstallResponse>

// Capabilities Operation (TEAM-214)
pub async fn execute_hive_refresh_capabilities(request: HiveRefreshCapabilitiesRequest, config: Arc<RbeeConfig>) -> Result<HiveRefreshCapabilitiesResponse>

// Validation Helper (TEAM-210)
pub fn validate_hive_exists<'a>(config: &'a RbeeConfig, alias: &str) -> Result<&'a HiveEntry>

// SSH Testing (TEAM-210)
pub async fn execute_ssh_test(request: SshTestRequest) -> Result<SshTestResponse>

// HTTP Client (TEAM-212)
pub async fn fetch_hive_capabilities(endpoint: &str) -> Result<Vec<DeviceInfo>>
pub async fn check_hive_health(endpoint: &str) -> Result<bool>
```

### Request/Response Types (9 pairs)

**All types in `src/types.rs` (188 LOC):**
- `HiveInstallRequest/Response` - Binary path resolution
- `HiveUninstallRequest/Response` - Cleanup confirmation
- `HiveStartRequest/Response` - Includes `job_id` (CRITICAL for SSE)
- `HiveStopRequest/Response` - Includes `job_id` (CRITICAL for SSE)
- `HiveListRequest/Response` - Returns `Vec<HiveInfo>`
- `HiveGetRequest/Response` - Returns single `HiveInfo`
- `HiveStatusRequest/Response` - Includes `job_id`, returns `running: bool`
- `HiveRefreshCapabilitiesRequest/Response` - Includes `job_id`, returns device count
- `HiveInfo` - Shared metadata struct (alias, hostname, port, binary_path)

---

## 2. State Machine Behaviors

### Hive Lifecycle States

```
NOT_INSTALLED → INSTALLED → STOPPED → STARTING → RUNNING → STOPPING → STOPPED
                    ↓                                          ↓
                UNINSTALLED ←──────────────────────────────────┘
```

**State Transitions:**

1. **NOT_INSTALLED → INSTALLED** (`execute_hive_install`)
   - Validates hive config exists
   - Resolves binary path (config → debug → release)
   - Localhost only (remote SSH not implemented)

2. **INSTALLED → STARTING** (`execute_hive_start`)
   - Checks if already running (health check)
   - Spawns daemon via `DaemonManager`
   - Polls health with exponential backoff (10 attempts, 200ms * attempt)

3. **STARTING → RUNNING** (health poll success)
   - Fetches capabilities with 15s timeout
   - Caches device information
   - Returns endpoint URL

4. **RUNNING → STOPPING** (`execute_hive_stop`)
   - Sends SIGTERM (graceful shutdown)
   - Waits 5 seconds for graceful exit
   - Falls back to SIGKILL if timeout

5. **STOPPING → STOPPED** (process exit)
   - Health check fails (connection refused)
   - Process no longer in process table

6. **INSTALLED → UNINSTALLED** (`execute_hive_uninstall`)
   - Removes from capabilities cache
   - User must manually edit hives.conf

### Capabilities Cache States

```
EMPTY → FETCHING → CACHED → STALE → REFRESHING → CACHED
         ↓                                ↓
       ERROR ←──────────────────────────ERROR
```

**Cache Behaviors:**
- **Cache Hit** (start.rs:238): Returns cached data, suggests refresh
- **Cache Miss** (start.rs:258): Fetches fresh, updates cache
- **Refresh** (capabilities.rs:41): Forces fresh fetch, updates cache
- **Cleanup** (uninstall.rs:49): Removes from cache on uninstall

---

## 3. Data Flows

### HiveStart Data Flow (Most Complex)

```
Request(alias, job_id) 
  → validate_hive_exists() 
  → health_check(2s timeout)
  → [if running] → check_cache → [hit] → return cached
                                → [miss] → fetch_capabilities(15s)
  → [if not running] → resolve_binary_path()
                    → DaemonManager::spawn()
                    → health_poll(10 attempts, exponential backoff)
                    → fetch_capabilities(15s timeout)
                    → update_cache()
                    → return Response(endpoint)
```

**Critical Timeouts:**
- Health check: 2s (start.rs:64)
- Health poll: 200ms * attempt, max 10 attempts = 11s total (start.rs:126)
- Capabilities fetch: 15s with TimeoutEnforcer (start.rs:284)
- Status check: 5s (status.rs:47)

### Binary Resolution Logic (3-tier fallback)

```
1. Check config.binary_path → if exists, use it
2. Check target/debug/rbee-hive → if exists, use it
3. Check target/release/rbee-hive → if exists, use it
4. Error: "Build it with: cargo build --bin rbee-hive"
```

**Files:** `start.rs:145-223`, `install.rs:84-158`

### Graceful Shutdown Flow

```
pkill -TERM rbee-hive
  → wait 1s → health_check → [failed] → SUCCESS
  → wait 1s → health_check → [failed] → SUCCESS
  → wait 1s → health_check → [failed] → SUCCESS
  → wait 1s → health_check → [failed] → SUCCESS
  → wait 1s → health_check → [still running] → pkill -KILL rbee-hive
                                              → wait 500ms
                                              → SUCCESS (forced)
```

**File:** `stop.rs:133-172`

---

## 4. Error Handling

### Error Types & Propagation

**All operations return `anyhow::Result<T>`**

**Error Categories:**

1. **Configuration Errors** (validation.rs:31-70)
   - Hive alias not found → Lists available hives
   - hives.conf missing → Auto-generates template with example
   - Localhost special case → Returns default entry (no config needed)

2. **Binary Resolution Errors** (start.rs:218, install.rs:154)
   - Binary not found → Suggests `cargo build --bin rbee-hive`
   - Provided path invalid → Shows exact path that failed

3. **Connection Errors** (hive_client.rs:40-47)
   - Health check timeout → "Hive not running"
   - HTTP error status → Shows status code + body
   - JSON parse error → "Failed to parse capabilities response"

4. **Lifecycle Errors**
   - Start timeout (start.rs:141): "Hive health check timed out"
   - Stop not running (stop.rs:71): Returns success (idempotent)
   - Capabilities fetch timeout (start.rs:343): Partial success, warns user

5. **SSH Errors** (ssh_test.rs:79-84)
   - Connection failed → Returns `SshTestResponse { success: false, error: Some(...) }`
   - Timeout → Handled by ssh-client crate (5s default)

### Error Message Patterns

**All errors include actionable advice:**

```rust
// Example: Binary not found (install.rs:139-156)
"❌ rbee-hive binary not found.\n\
 \n\
 Please build it first:\n\
 \n\
   cargo build --bin rbee-hive\n\
 \n\
 Or provide a binary path:\n\
 \n\
   ./rbee hive install --binary-path /path/to/rbee-hive"
```

---

## 5. Integration Points

### Dependencies (Cargo.toml:14-28)

**External:**
- `anyhow` - Error handling
- `tokio` - Async runtime
- `reqwest` - HTTP client
- `serde/serde_json` - Serialization
- `once_cell` - Lazy static (localhost entry)

**Internal:**
- `daemon-lifecycle` - Process spawning/management
- `observability-narration-core` - SSE event emission
- `timeout-enforcer` - Visible timeouts with countdown
- `rbee-config` - Configuration + capabilities cache
- `queen-rbee-ssh-client` - SSH connection testing

### Dependents

**Primary consumer:** `bin/10_queen_rbee/src/job_router.rs`

**Integration pattern (job_router.rs):**
```rust
Operation::HiveStart { alias } => {
    let request = HiveStartRequest { alias, job_id: job_id.clone() };
    execute_hive_start(request, state.config.clone()).await?;
}
```

**All 9 operations follow this thin wrapper pattern (TEAM-215)**

### Narration Integration (SSE Routing)

**CRITICAL:** All narration includes `.job_id(job_id)` for SSE channel routing

**Pattern (used in all operations):**
```rust
NARRATE
    .action("hive_start")
    .job_id(job_id)           // ← REQUIRED for SSE routing
    .context(alias)
    .human("🚀 Starting hive '{}'")
    .emit();
```

**Without job_id:** Events go to stdout only, never reach SSE streams (fail-fast security)

**Files using narration:** All 9 operation files + ssh_test.rs

---

## 6. Critical Invariants

### Must Always Be True

1. **Localhost Special Case** (validation.rs:18-28)
   - `alias == "localhost"` → Always returns default entry
   - No hives.conf required for localhost operations
   - Binary path defaults to `target/debug/rbee-hive`

2. **Job ID Propagation** (types.rs:56, 80, 150, 174)
   - All operations requiring SSE routing include `job_id` field
   - Start, Stop, Status, RefreshCapabilities have `job_id`
   - List, Get, Install, Uninstall receive `job_id` as parameter

3. **Binary Resolution Order** (start.rs:145-223, install.rs:84-158)
   - Always: config → debug → release → error
   - Never skip steps in resolution chain
   - Always validate path exists before using

4. **Health Check Idempotency** (start.rs:67-79, stop.rs:63-87)
   - Start: If already running → check cache → return success
   - Stop: If not running → return success (no error)

5. **Capabilities Cache Consistency** (start.rs:318, capabilities.rs:150)
   - Always update cache after successful fetch
   - Always save to disk (warn on error, don't fail)
   - Cache key = hive alias (string)

6. **Graceful Shutdown Pattern** (stop.rs:108-172)
   - Always try SIGTERM first (5 second grace period)
   - Always fall back to SIGKILL if timeout
   - Never leave zombie processes

7. **Error Messages Include Context** (all files)
   - Always show what failed (alias, path, URL)
   - Always suggest next action
   - Always use emoji prefixes for visual parsing

### Must Never Happen

1. **Never spawn without health poll** (start.rs:98-142)
   - Spawning without health check → orphan process
   - Always poll after spawn, timeout after 10 attempts

2. **Never cache without validation** (start.rs:318-335)
   - Always validate devices exist before caching
   - Always check save() result (warn on error)

3. **Never use job_id for routing without SSE** (all operations)
   - Operations without job_id → stdout only
   - Operations with job_id → SSE routing enabled

4. **Never skip binary validation** (start.rs:157-166, install.rs:93-102)
   - Always check path.exists() before using
   - Never assume binary is present

---

## 7. Existing Test Coverage

### Unit Tests

**Status:** ❌ NO UNIT TESTS FOUND

**Expected locations:**
- `src/lib.rs` - No `#[cfg(test)]` modules
- Individual operation files - No test modules
- `tests/` directory - Does not exist

### BDD Tests

**Status:** ⚠️ PLACEHOLDER ONLY

**File:** `bdd/tests/features/placeholder.feature` (13 lines)
- Generic placeholder scenario
- No actual hive lifecycle tests
- BDD harness exists but not used

### Integration Tests

**Status:** ❌ NONE

**Expected:** Integration tests with real hive instances

### Test Coverage Gaps (IMPLEMENTED CODE)

**Critical gaps (no tests for implemented features):**

1. **Binary Resolution Logic** (start.rs:145-223, install.rs:84-158)
   - No tests for 3-tier fallback (config → debug → release)
   - No tests for path validation
   - No tests for error messages

2. **Health Polling** (start.rs:108-142)
   - No tests for exponential backoff
   - No tests for timeout behavior
   - No tests for early success

3. **Graceful Shutdown** (stop.rs:108-172)
   - No tests for SIGTERM → SIGKILL fallback
   - No tests for 5-second grace period
   - No tests for idempotency (already stopped)

4. **Capabilities Caching** (start.rs:226-358)
   - No tests for cache hit/miss logic
   - No tests for cache update/save
   - No tests for cache cleanup on uninstall

5. **Validation Logic** (validation.rs:14-71)
   - No tests for localhost special case
   - No tests for hives.conf auto-generation
   - No tests for error message formatting

6. **HTTP Client** (hive_client.rs:36-89)
   - No tests for capabilities fetch
   - No tests for health check
   - No tests for timeout handling

7. **SSH Testing** (ssh_test.rs:49-88)
   - No tests for connection success/failure
   - No tests for timeout behavior

**Future Features (TODO markers, not gaps):**
- Remote SSH installation (install.rs:64-74) - Intentionally not implemented
- Full cleanup options (uninstall.rs:96-122) - Documented for future

---

## 8. Behavior Checklist

- [x] All public APIs documented
- [x] All state transitions documented
- [x] All error paths documented
- [x] All integration points documented
- [x] All edge cases documented
- [x] Existing test coverage assessed
- [x] Coverage gaps identified (for IMPLEMENTED features only)

---

## Summary

**Complexity:** HIGH (1,629 LOC, 11 operations, complex state machine)

**Key Strengths:**
- Clean module structure (9 operation files)
- Consistent error handling with actionable messages
- Proper SSE integration (job_id propagation)
- Idempotent operations (start/stop)
- Graceful shutdown with fallback

**Key Weaknesses:**
- **ZERO test coverage** (critical gap)
- No unit tests for binary resolution
- No integration tests for lifecycle
- BDD harness exists but unused

**Critical Behaviors:**
1. Binary resolution: config → debug → release
2. Health polling: exponential backoff, 10 attempts
3. Graceful shutdown: SIGTERM → 5s wait → SIGKILL
4. Capabilities caching: fetch → cache → save
5. Localhost special case: no config required

**Next Steps for Testing (Phase 3):**
1. Unit tests for binary resolution logic
2. Unit tests for validation helpers
3. Integration tests for start/stop lifecycle
4. BDD scenarios for end-to-end flows
5. Mock HTTP client for capabilities tests

---

**TEAM-220: Investigation complete**  
**Status:** ✅ READY FOR PHASE 3
