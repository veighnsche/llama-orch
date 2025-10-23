# TEAM-276: Health Polling with Exponential Backoff

**Status:** ✅ COMPLETE  
**Date:** Oct 23, 2025  
**File:** `src/health.rs`

## Mission

Add health polling with exponential backoff to `daemon-lifecycle` for daemon startup synchronization (NOT for ongoing worker monitoring).

## Context

After analyzing heartbeats vs polling patterns, we determined:
- **Heartbeats** = Correct for ongoing worker monitoring (scalable, rich data)
- **Health Polling** = Correct for daemon startup synchronization (one-time, wait until ready)

## Implementation

### New Types

**HealthPollConfig**
```rust
pub struct HealthPollConfig {
    pub base_url: String,
    pub health_endpoint: Option<String>,
    pub max_attempts: usize,         // Default: 10
    pub initial_delay_ms: u64,       // Default: 200ms
    pub backoff_multiplier: f64,     // Default: 1.5
    pub job_id: Option<String>,
    pub daemon_name: Option<String>,
}
```

Builder pattern with fluent API:
```rust
let config = HealthPollConfig::new("http://localhost:8500")
    .with_max_attempts(10)
    .with_job_id("job-123")
    .with_daemon_name("queen-rbee");
```

### New Function

**poll_until_healthy**
```rust
pub async fn poll_until_healthy(config: HealthPollConfig) -> anyhow::Result<()>
```

**Exponential Backoff Schedule:**
- Attempt 1: 200ms
- Attempt 2: 300ms (200 × 1.5)
- Attempt 3: 450ms (300 × 1.5)
- Attempt 4: 675ms (450 × 1.5)
- Attempt 5: 1012ms (675 × 1.5)
- ... up to max_attempts

**Narration Events:**
1. Start: "⏳ Waiting for {daemon} to become healthy"
2. Progress: "🔄 Attempt X/Y, retrying in Zms..."
3. Success: "✅ {daemon} is healthy (attempt X)"
4. Failure: "❌ {daemon} failed to become healthy after X attempts"

## Usage Examples

### Queen Startup
```rust
use daemon_lifecycle::{spawn_daemon, poll_until_healthy, HealthPollConfig};

// Spawn queen daemon
spawn_daemon(...).await?;

// Wait until healthy
let config = HealthPollConfig::new("http://localhost:8500")
    .with_daemon_name("queen-rbee");

poll_until_healthy(config).await?;
// Now queen is ready to accept requests!
```

### Hive Startup (with job_id)
```rust
// In hive-lifecycle/start.rs
pub async fn execute_hive_start(request: HiveStartRequest, config: Arc<RbeeConfig>) -> Result<()> {
    let job_id = &request.job_id;
    
    // Spawn hive
    spawn_hive_daemon(...).await?;
    
    // Poll until ready
    let poll_config = HealthPollConfig::new(&hive_endpoint)
        .with_job_id(job_id)
        .with_daemon_name("rbee-hive");
    
    poll_until_healthy(poll_config).await?;
    
    // Hive is ready, proceed with capabilities fetch
    Ok(())
}
```

### Simple Usage
```rust
// Minimal usage with defaults
poll_until_healthy(HealthPollConfig::new("http://localhost:8500")).await?;
```

## Benefits

### 1. **Daemon Startup Synchronization** ⭐⭐⭐
- Know exactly when daemon is accepting requests
- No race conditions
- Reliable orchestration

### 2. **Exponential Backoff** ⭐⭐⭐
- Starts fast (200ms first attempt)
- Backs off gracefully if daemon slow to start
- Doesn't hammer the daemon

### 3. **Narration Support** ⭐⭐
- Progress visibility via SSE
- Job ID propagation for proper routing
- Error tracking with error_kind

### 4. **Reusable Pattern** ⭐⭐⭐
- Works for queen, hive, worker daemons
- Configurable for different scenarios
- Single source of truth

### 5. **Fail Fast** ⭐⭐
- Clear timeout after max attempts
- Descriptive error messages
- No indefinite waiting

## NOT For

This function is **NOT for ongoing monitoring**:

❌ **Don't use for:**
- Periodic worker health checks (use heartbeats!)
- Monitoring 100s of workers (use heartbeats!)
- Continuous health verification (use heartbeats!)

✅ **Only use for:**
- Daemon startup (one-time wait until ready)
- Manual health verification (debugging)
- Service dependency checks (before starting dependent service)

## Comparison with Existing

### Before (hive-lifecycle had custom implementation)
```rust
// In hive-lifecycle/start.rs (lines 300-340)
// Custom exponential backoff implementation
for attempt in 1..=10 {
    if is_hive_healthy(...).await {
        break;
    }
    let delay = Duration::from_millis(200 * multiplier.powi(attempt));
    sleep(delay).await;
}
// ~40 LOC of custom logic
```

### After (use shared implementation)
```rust
// In hive-lifecycle/start.rs
let config = HealthPollConfig::new(&hive_endpoint)
    .with_job_id(job_id)
    .with_daemon_name("rbee-hive");

poll_until_healthy(config).await?;
// 4 LOC, same functionality
```

**Code Reduction**: ~36 LOC per usage

## Integration Points

### Current Lifecycle Crates

1. **queen-lifecycle** - Can use for queen startup in `start.rs`
2. **hive-lifecycle** - Can replace custom polling in `start.rs`
3. **worker-lifecycle** - Can use for worker daemon startup if needed

### Estimated Impact

- **hive-lifecycle**: Save ~40 LOC (replace custom implementation)
- **queen-lifecycle**: Save ~30 LOC (add startup sync)
- **Future crates**: Consistent pattern available

**Total savings**: ~70 LOC across crates

## Files Modified

1. **src/health.rs**
   - Added `HealthPollConfig` struct with builder pattern
   - Added `poll_until_healthy()` function
   - Added comprehensive narration
   - Added documentation with examples

2. **src/lib.rs**
   - Exported `poll_until_healthy` and `HealthPollConfig`
   - Added usage example in module docs

## Verification

```bash
# Compilation
cargo check -p daemon-lifecycle
# ✅ SUCCESS

# Future usage
# - Can be used by queen-lifecycle for startup
# - Can be used by hive-lifecycle for startup
# - Can be used for manual health checks
```

## Design Decisions

### 1. **Config Struct with Builder**
- Chosen: Struct with builder methods
- Alternative: Function with many parameters
- Reason: Flexible, extensible, self-documenting

### 2. **Exponential Backoff Formula**
- Chosen: `delay = initial_ms × multiplier^(attempt-1)`
- Default: 200ms × 1.5^n
- Reason: Fast initial attempts, reasonable backoff

### 3. **Narration Integration**
- Chosen: Emit progress at each attempt
- Alternative: Silent polling
- Reason: Observability is critical for debugging

### 4. **Error Handling**
- Chosen: Return Result, emit error narration
- Alternative: Return bool
- Reason: Caller can handle failure appropriately

## Future Enhancements (Optional)

### Phase 2 (if needed)
1. Add jitter to backoff (prevent thundering herd)
2. Add custom health check predicate (beyond just 2xx)
3. Add health check response parsing
4. Add circuit breaker pattern

### Phase 3 (if needed)
5. Add metrics collection (success rate, avg attempts)
6. Add adaptive backoff (based on historical data)

## Conclusion

Successfully added health polling with exponential backoff to `daemon-lifecycle`:

- ✅ **200 LOC** implementation with comprehensive narration
- ✅ **Builder pattern** for flexible configuration
- ✅ **Exponential backoff** (200ms → 300ms → 450ms → ...)
- ✅ **Job ID support** for SSE routing
- ✅ **Reusable** by all lifecycle crates
- ✅ **Clean compilation** with no warnings

This provides the **correct tool for daemon startup synchronization** while keeping heartbeats for ongoing worker monitoring.
