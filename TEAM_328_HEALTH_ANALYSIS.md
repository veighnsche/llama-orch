# TEAM-328: health.rs RULE ZERO Analysis

## Question
Does `daemon-lifecycle/src/health.rs` violate RULE ZERO?

## Analysis

### Functions Provided

**Two public functions:**

1. **`is_daemon_healthy()`** - Simple bool check (no narration)
   ```rust
   pub async fn is_daemon_healthy(
       base_url: &str,
       health_endpoint: Option<&str>,
       timeout: Option<Duration>,
   ) -> bool
   ```

2. **`poll_until_healthy()`** - Retry with exponential backoff (with narration)
   ```rust
   pub async fn poll_until_healthy(config: HealthPollConfig) -> anyhow::Result<()>
   ```

### Usage Analysis

**`is_daemon_healthy()` - Used 5 times:**
- `rebuild.rs` - Check if daemon running before rebuild
- `stop.rs` - Check if daemon running before stop
- `shutdown.rs` - Check if daemon running before shutdown
- `uninstall.rs` - Check if daemon running before uninstall
- `poll_until_healthy()` - Internal use for retry loop

**`poll_until_healthy()` - Used 1 time:**
- `start.rs` - Wait for daemon to become healthy after spawn

### RULE ZERO Assessment

**NO VIOLATION - These serve different purposes:**

#### Different Use Cases

**`is_daemon_healthy()` - Single Check:**
- Quick health check (one attempt)
- Returns bool (simple yes/no)
- No narration (silent check)
- Used for: "Is it running right now?"

**`poll_until_healthy()` - Retry Loop:**
- Multiple attempts with backoff
- Returns Result (success/failure after retries)
- With narration (progress updates)
- Used for: "Wait until it becomes healthy"

#### Not Backwards Compatibility

These are **complementary functions**, not duplicate implementations:

```rust
// Quick check - is it running?
if is_daemon_healthy(url, None, None).await {
    // Already running
}

// Wait for startup - retry until healthy
poll_until_healthy(config).await?;
```

You **cannot** replace one with the other:
- Can't use `poll_until_healthy()` for quick checks (too slow, unnecessary retries)
- Can't use `is_daemon_healthy()` for startup sync (no retry logic)

### Comparison to manager.rs Violations

**manager.rs violations (deleted):**
- `find_in_target()` vs `find_binary()` - **Same purpose**, different scope
- `spawn_daemon()` vs `DaemonManager::new().spawn()` - **Same purpose**, wrapper

**health.rs (NOT violations):**
- `is_daemon_healthy()` vs `poll_until_healthy()` - **Different purposes**
  - One is "check once"
  - Other is "retry until success"

### Design Pattern

This is a **common pattern** in health checking:

**Single check:**
```rust
is_healthy() -> bool  // One attempt
```

**Retry loop:**
```rust
wait_until_healthy() -> Result<()>  // Multiple attempts with backoff
```

Examples in other systems:
- Kubernetes: `readinessProbe` (single) vs `startupProbe` (retry)
- Docker: `healthcheck` (single) vs `--health-retries` (retry)
- HTTP clients: `get()` (single) vs `retry_get()` (retry)

## Conclusion

**NO, health.rs does NOT violate RULE ZERO.**

**Reasons:**
1. ✅ Different purposes (single check vs retry loop)
2. ✅ Different return types (bool vs Result)
3. ✅ Different behavior (silent vs narration)
4. ✅ Different use cases (status check vs startup sync)
5. ✅ Cannot replace one with the other

**This is good design:**
- Simple function for simple use case
- Complex function for complex use case
- Clear separation of concerns
- Both are necessary and used

**No action needed.**

---

**TEAM-328 Assessment:** health.rs is well-designed, no RULE ZERO violations
