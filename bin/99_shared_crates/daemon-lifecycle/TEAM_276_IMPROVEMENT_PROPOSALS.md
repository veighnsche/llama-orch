# TEAM-276: daemon-lifecycle Improvement Proposals

**Date:** Oct 23, 2025  
**Status:** PROPOSED  
**Based on**: Lessons learned from queen-lifecycle, hive-lifecycle, worker-lifecycle

## Executive Summary

After implementing three lifecycle crates, we've identified **8 key improvements** that could be added to `daemon-lifecycle` to make it more powerful and consistent with patterns we've discovered.

## Current State

`daemon-lifecycle` currently provides:
- ✅ DaemonManager for spawning processes
- ✅ Health checking (`is_daemon_healthy`)
- ✅ Ensure daemon running pattern
- ✅ Install/uninstall with binary resolution
- ✅ List/get/status operations
- ✅ Basic narration support

## Proposed Improvements

### 1. **Add Health Polling with Exponential Backoff** ⭐⭐⭐

**Pattern from**: hive-lifecycle/start.rs

**Current:**
```rust
pub async fn is_daemon_healthy(url: &str, endpoint: Option<&str>, timeout: Option<Duration>) -> bool
```

**Problem**: Single health check, no retry logic

**Proposed:**
```rust
pub struct HealthPollConfig {
    pub url: String,
    pub endpoint: Option<String>,
    pub max_attempts: usize,        // Default: 10
    pub initial_delay_ms: u64,      // Default: 200ms
    pub backoff_multiplier: f64,    // Default: 1.5
    pub job_id: Option<String>,
}

pub async fn poll_until_healthy(config: HealthPollConfig) -> Result<()> {
    for attempt in 1..=config.max_attempts {
        let delay = Duration::from_millis(config.initial_delay_ms * (config.backoff_multiplier.powi(attempt as i32)) as u64);
        
        if is_daemon_healthy(&config.url, config.endpoint.as_deref(), Some(delay)).await {
            return Ok(());
        }
        
        // Emit progress narration
        sleep(delay).await;
    }
    anyhow::bail!("Daemon failed to become healthy after {} attempts", config.max_attempts)
}
```

**Benefits:**
- Reusable by all three lifecycle crates
- Consistent retry logic
- Progress narration
- Configurable backoff

**Impact**: HIGH - Used by hive-lifecycle, could be used by queen/worker

---

### 2. **Add Graceful Shutdown Pattern** ⭐⭐⭐

**Pattern from**: hive-lifecycle/stop.rs, queen-lifecycle/stop.rs

**Current**: No stop/shutdown utilities

**Proposed:**
```rust
pub struct ShutdownConfig {
    pub daemon_name: String,
    pub health_url: String,
    pub shutdown_endpoint: String,  // e.g., "/v1/shutdown"
    pub sigterm_timeout_secs: u64,  // Default: 5
    pub job_id: Option<String>,
}

pub async fn graceful_shutdown(config: ShutdownConfig) -> Result<()> {
    // Step 1: Check if running
    if !is_daemon_healthy(&config.health_url, None, Some(Duration::from_secs(2))).await {
        // Not running, return Ok
        return Ok(());
    }
    
    // Step 2: Send shutdown request
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()?;
    
    match client.post(&config.shutdown_endpoint).send().await {
        Ok(_) => Ok(()),
        Err(e) if e.is_connect() || e.to_string().contains("connection closed") => {
            // Expected - daemon shut down before responding
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

pub async fn force_shutdown(pid: u32, job_id: Option<&str>) -> Result<()> {
    // SIGTERM, wait, then SIGKILL if needed
    // (Implementation from hive-lifecycle/stop.rs)
}
```

**Benefits:**
- Reusable shutdown logic
- Handles expected connection errors
- Fallback to SIGKILL if needed
- Consistent behavior

**Impact**: MEDIUM - Used by queen/hive, less by worker

---

### 3. **Add Process Management Utilities** ⭐⭐

**Pattern from**: worker-lifecycle (process_list.rs, process_get.rs)

**Current**: No process inspection utilities

**Proposed:**
```rust
pub mod process;

pub struct ProcessInfo {
    pub pid: u32,
    pub command: String,
    pub args: Vec<String>,
    pub uptime_secs: u64,
    pub memory_mb: u64,
}

/// List all processes matching a pattern
pub fn list_processes(pattern: &str) -> Result<Vec<ProcessInfo>> {
    // Use `ps aux | grep pattern`
}

/// Get process info by PID
pub fn get_process(pid: u32) -> Result<ProcessInfo> {
    // Use `ps -p PID -o ...`
}

/// Check if process is running
pub fn is_process_running(pid: u32) -> bool {
    // Use `kill -0 PID`
}
```

**Benefits:**
- Reusable process inspection
- Cross-platform (with #[cfg])
- Consistent output format

**Impact**: MEDIUM - Used by worker-lifecycle, useful for debugging

---

### 4. **Add SSH Support** ⭐⭐

**Pattern from**: hive-lifecycle (ssh_helper.rs, ssh_test.rs)

**Current**: No SSH utilities

**Proposed:**
```rust
pub mod ssh;

pub struct SshConfig {
    pub host: String,
    pub user: String,
    pub port: u16,
    pub identity_file: Option<String>,
}

/// Execute command over SSH
pub async fn ssh_exec(config: &SshConfig, command: &str) -> Result<String> {
    // Implementation from hive-lifecycle/ssh_helper.rs
}

/// Test SSH connection
pub async fn ssh_test(config: &SshConfig) -> Result<()> {
    // Implementation from hive-lifecycle/ssh_test.rs
}

/// Get remote binary path
pub async fn get_remote_binary_path(config: &SshConfig, binary_name: &str) -> Result<String> {
    // Find binary on remote system
}
```

**Benefits:**
- Reusable SSH operations
- Consistent error handling
- Support for remote daemons

**Impact**: LOW - Only used by hive-lifecycle currently, but enables remote worker management

---

### 5. **Add Configuration Patterns** ⭐⭐⭐

**Pattern from**: All three lifecycle crates use config structs

**Current**: Mix of config structs (InstallConfig, UninstallConfig) and loose parameters

**Proposed**: Standardize all operations to use config structs

```rust
// Common pattern for all operations
pub struct OperationConfig {
    pub daemon_name: String,
    pub job_id: Option<String>,
    // ... operation-specific fields
}

// Implement From<T> for easy conversion
impl From<SimpleParams> for OperationConfig {
    fn from(params: SimpleParams) -> Self {
        // ...
    }
}
```

**Benefits:**
- Consistent API surface
- Easy to extend without breaking changes
- Better documentation
- Optional parameters more obvious

**Impact**: MEDIUM - Improves API consistency

---

### 6. **Add Narration Helpers** ⭐⭐

**Pattern from**: All lifecycle crates repeat narration patterns

**Current**: Each function constructs narration manually

**Proposed:**
```rust
pub mod narration;

/// Emit progress narration with job_id
pub fn emit_progress(
    action: &str,
    message: &str,
    job_id: Option<&str>,
) {
    let mut n = NARRATE.action(action).human(message);
    if let Some(jid) = job_id {
        n = n.job_id(jid);
    }
    n.emit();
}

/// Emit error narration with job_id and error_kind
pub fn emit_error(
    action: &str,
    message: &str,
    error_kind: &str,
    job_id: Option<&str>,
) {
    let mut n = NARRATE.action(action)
        .human(message)
        .error_kind(error_kind);
    if let Some(jid) = job_id {
        n = n.job_id(jid);
    }
    n.emit_error();
}
```

**Benefits:**
- DRY - Don't repeat narration patterns
- Consistent formatting
- Always includes job_id when available

**Impact**: LOW - Quality of life improvement

---

### 7. **Add Timeout Enforcement** ⭐⭐⭐

**Pattern from**: hive-lifecycle uses TimeoutEnforcer

**Current**: Manual timeout handling

**Proposed:**
```rust
use timeout_enforcer::TimeoutEnforcer;

pub struct TimeoutConfig {
    pub operation_name: String,
    pub timeout_secs: u64,
    pub job_id: Option<String>,
}

/// Wrap an async operation with timeout enforcement
pub async fn with_timeout<F, T>(
    config: TimeoutConfig,
    operation: F,
) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    let mut enforcer = TimeoutEnforcer::new(
        Duration::from_secs(config.timeout_secs),
        &config.operation_name,
    );
    
    if let Some(ref job_id) = config.job_id {
        enforcer = enforcer.with_job_id(job_id);
    }
    
    enforcer.enforce(operation).await
}
```

**Benefits:**
- Consistent timeout behavior
- Automatic timeout narration
- Job ID propagation

**Impact**: HIGH - Prevents hangs across all operations

---

### 8. **Add Stdio Configuration** ⭐

**Pattern from**: All lifecycle crates need stdio control

**Current**: DaemonManager uses Stdio::null() hardcoded

**Proposed:**
```rust
pub enum StdioMode {
    Null,           // Stdio::null() - for daemons
    Inherit,        // Stdio::inherit() - for debugging
    Piped,          // Stdio::piped() - for capturing output
    File(PathBuf),  // Redirect to file
}

pub struct SpawnConfig {
    pub binary_path: PathBuf,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub stdout: StdioMode,
    pub stderr: StdioMode,
    pub stdin: StdioMode,
}
```

**Benefits:**
- Flexible stdio handling
- Easy debugging (switch to Inherit)
- Log capture support

**Impact**: LOW - Nice to have for debugging

---

## Priority Matrix

| Priority | Improvement | Impact | Effort | Status |
|----------|-------------|--------|--------|--------|
| **P0** | Health Polling with Backoff | HIGH | Medium | PROPOSED |
| **P0** | Timeout Enforcement | HIGH | Low | PROPOSED |
| **P1** | Graceful Shutdown Pattern | MEDIUM | Medium | PROPOSED |
| **P1** | Configuration Patterns | MEDIUM | High | PROPOSED |
| **P2** | Process Management | MEDIUM | Medium | PROPOSED |
| **P2** | Narration Helpers | LOW | Low | PROPOSED |
| **P3** | SSH Support | LOW | High | PROPOSED |
| **P3** | Stdio Configuration | LOW | Low | PROPOSED |

## Implementation Plan

### Phase 1: Core Improvements (P0)
1. Add `health::poll_until_healthy()` with exponential backoff
2. Add `timeout` module with `with_timeout()` wrapper
3. Update existing functions to use new utilities

### Phase 2: Lifecycle Patterns (P1)
4. Add `shutdown` module with graceful/force shutdown
5. Standardize all operations to use config structs
6. Update documentation with new patterns

### Phase 3: Advanced Features (P2-P3)
7. Add `process` module for process inspection
8. Add `narration` module with helpers
9. Add `ssh` module (optional, only if remote worker management needed)
10. Add flexible stdio configuration

## Migration Path

### Backward Compatibility

All improvements should be **additive** - don't break existing APIs:

```rust
// OLD API (keep working)
pub async fn is_daemon_healthy(url: &str, endpoint: Option<&str>, timeout: Option<Duration>) -> bool

// NEW API (add alongside)
pub async fn poll_until_healthy(config: HealthPollConfig) -> Result<()>
```

### Deprecation Strategy

1. Add new functions with better patterns
2. Mark old functions as `#[deprecated]` with migration hints
3. Update all lifecycle crates to use new patterns
4. Remove deprecated functions in next major version

## Expected Benefits

### For queen-lifecycle
- ✅ Use `poll_until_healthy()` instead of custom logic
- ✅ Use `graceful_shutdown()` for stop operation
- ✅ Consistent timeout enforcement

### For hive-lifecycle
- ✅ Move SSH utilities to shared crate (reduce duplication)
- ✅ Use standardized health polling
- ✅ Use standardized shutdown pattern

### For worker-lifecycle
- ✅ Use process management utilities
- ✅ Consistent spawn configuration
- ✅ Better timeout enforcement

### Overall Impact
- **~500 LOC** of duplicated code eliminated
- **Consistent patterns** across all lifecycle crates
- **Better error handling** with timeouts
- **Easier debugging** with narration helpers

## Decision: Implement or Not?

### Arguments FOR Implementation
1. Eliminates duplication across lifecycle crates
2. Provides battle-tested patterns
3. Makes future lifecycle crates easier to build
4. Improves consistency and reliability

### Arguments AGAINST Implementation
1. Adds complexity to shared crate
2. May not be needed by all consumers
3. Requires maintenance of more code
4. Some patterns may be too specific

### Recommendation

**Implement Phase 1 (P0) immediately:**
- Health polling with backoff (HIGH impact, used everywhere)
- Timeout enforcement (HIGH impact, prevents hangs)

**Evaluate Phase 2 (P1) after Phase 1:**
- See if patterns are actually reused
- Measure duplication reduction
- Get feedback from usage

**Defer Phase 3 (P2-P3):**
- Only implement if clear need emerges
- SSH support only if remote worker management added
- Process utilities only if used beyond worker-lifecycle

## Action Items

### Immediate (Phase 1)
- [ ] Create `health::poll_until_healthy()` module
- [ ] Create `timeout` module with wrapper
- [ ] Add comprehensive tests
- [ ] Update documentation
- [ ] Refactor queen-lifecycle to use new patterns

### Short-term (Phase 2)
- [ ] Evaluate duplication after Phase 1
- [ ] Decide on shutdown pattern implementation
- [ ] Consider config struct standardization

### Long-term (Phase 3)
- [ ] Monitor for SSH needs (remote worker management)
- [ ] Monitor for process utility needs
- [ ] Consider narration helper extraction

## Conclusion

The `daemon-lifecycle` crate can be significantly improved based on patterns from the three lifecycle crates. **Focus on Phase 1 (health polling + timeout)** as these provide the highest impact with reasonable effort.

The other improvements should be evaluated based on actual duplication and usage patterns rather than implemented speculatively.
