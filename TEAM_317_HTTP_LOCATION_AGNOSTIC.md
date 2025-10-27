# TEAM-317: HTTP is Location-Agnostic

**Critical Insight:** HTTP shutdown works the same regardless of location.

## The Mistake

**Original hive-lifecycle/src/stop.rs had TWO implementations:**

1. **Local (17 LOC):** Used `daemon_lifecycle::stop_http_daemon()` with `http://localhost:{port}`
2. **Remote (32 LOC):** Used SSH + curl to send HTTP request

**Total:** 49 LOC for what should be a single function.

## The Realization

HTTP doesn't care if the server is local or remote. The protocol is identical:
- Local: `http://localhost:7835/v1/shutdown`
- Remote: `http://remote-host:7835/v1/shutdown`

**The only difference is the hostname in the URL.**

## The Fix

**Before (49 LOC - two implementations):**
```rust
pub async fn stop_hive(host: &str, port: u16) -> Result<()> {
    if host == "localhost" || host == "127.0.0.1" {
        stop_hive_local(port).await  // 17 LOC
    } else {
        stop_hive_remote(host, port).await  // 32 LOC
    }
}
```

**After (11 LOC - single implementation):**
```rust
pub async fn stop_hive(host: &str, port: u16) -> Result<()> {
    let base_url = format!("http://{}:{}", host, port);
    let config = HttpDaemonConfig::new("rbee-hive", PathBuf::from(""), base_url);
    stop_http_daemon(config).await?;
    Ok(())
}
```

**Savings:** 38 LOC removed by eliminating false distinction

## Why This Happened

**Mental model error:** Thinking "local vs remote" instead of "HTTP endpoint".

When you think in terms of:
- ❌ "Local daemon vs remote daemon" → leads to two implementations
- ✅ "HTTP endpoint at URL" → leads to single implementation

## Lesson

**Location is not a protocol concern.** HTTP works the same everywhere:
- Local HTTP server
- Remote HTTP server
- HTTP server on Mars (with high latency)

The protocol doesn't change. Only the URL changes.

## Impact

**Total savings from TEAM-317:**
- queen-lifecycle: 51 LOC removed
- hive-lifecycle: 97 LOC removed (including this 38 LOC fix)
- **Total: 148 LOC eliminated**

**Root cause:** Conflating network topology (local/remote) with protocol behavior (HTTP).

---

**Remember:** If you're writing separate code for "local" and "remote" HTTP operations, you're probably doing it wrong.
