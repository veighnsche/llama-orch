# Health-Poll Crate Specification Checklist

## 📋 Core Requirements

### **1. HTTP Health Polling**
- [ ] Poll HTTP endpoint until success or max attempts
- [ ] Support configurable timeout per request (default: 5s)
- [ ] Return success on any 2xx status code
- [ ] Return error after max attempts exhausted
- [ ] Handle network errors gracefully (connection refused, timeout, DNS)

### **2. Exponential Backoff**
- [ ] Initial delay configurable (default: 200ms)
- [ ] Backoff multiplier configurable (default: 1.5x)
- [ ] Max attempts configurable (default: 30)
- [ ] Calculate delay: `initial_delay * (multiplier ^ attempt)`
- [ ] Cap maximum delay (e.g., 30 seconds) to prevent infinite waits

### **3. Error Handling**
- [ ] Clear error messages indicating which attempt failed
- [ ] Include URL in error messages
- [ ] Distinguish between:
  - Connection refused (daemon not started)
  - Timeout (daemon started but slow)
  - HTTP error (daemon started but unhealthy)
  - Network errors (DNS, routing)
- [ ] Return `anyhow::Result<()>` for easy error propagation

### **4. Logging/Observability**
- [ ] Log each polling attempt (optional, via feature flag?)
- [ ] Log final success/failure
- [ ] Include attempt number, delay, and response status in logs
- [ ] Support structured logging (tracing crate?)

### **5. Performance**
- [ ] Async implementation (tokio)
- [ ] Non-blocking delays (tokio::time::sleep)
- [ ] Reuse HTTP client across attempts (connection pooling)
- [ ] Minimal memory footprint

### **6. Configuration**
- [ ] [poll_health()](cci:1://file:///home/vince/Projects/llama-orch/bin/96_lifecycle/health-poll/src/lib.rs:10:0-87:1) function with parameters:
  - `url: &str` - Full health endpoint URL
  - `max_attempts: usize` - Maximum polling attempts
  - `initial_delay_ms: u64` - Initial delay in milliseconds
  - `backoff_multiplier: f64` - Exponential backoff multiplier
- [ ] Optional: Builder pattern for advanced configuration
- [ ] Optional: Preset configurations (fast, normal, slow)

### **7. Integration Points**

#### **Used by lifecycle-local:**
- [ ] Poll `http://localhost:{port}/health` after starting local daemon
- [ ] Typical: 30 attempts, 200ms initial, 1.5x backoff (~30s total)
- [ ] Return PID only after health check succeeds

#### **Used by lifecycle-ssh:**
- [ ] Poll `http://{remote_host}:{port}/health` after SSH daemon start
- [ ] Same parameters as local (network may be slower)
- [ ] Handle SSH tunnel scenarios (localhost forwarding)

#### **Used by lifecycle-monitored:**
- [ ] Poll `http://localhost:{port}/health` after starting monitored process
- [ ] May need longer timeout for GPU initialization
- [ ] Integrate with process-monitor status

### **8. Edge Cases**
- [ ] Handle redirects (3xx status codes)
- [ ] Handle partial responses (connection closed mid-response)
- [ ] Handle very slow responses (streaming, chunked encoding)
- [ ] Handle IPv4 vs IPv6 (localhost = 127.0.0.1 vs ::1)
- [ ] Handle custom ports (not just 80/443)
- [ ] Handle HTTPS with self-signed certificates (optional)

### **9. Testing**
- [ ] Unit test: Successful polling (mock server)
- [ ] Unit test: Max attempts exhausted (mock server never responds)
- [ ] Unit test: Exponential backoff timing verification
- [ ] Unit test: Network errors (connection refused)
- [ ] Unit test: HTTP errors (500, 503)
- [ ] Unit test: Timeout handling
- [ ] Integration test: Real HTTP server (spawn test server)

### **10. Documentation**
- [ ] Crate-level docs explaining purpose
- [ ] Function docs with examples
- [ ] Parameter descriptions
- [ ] Common usage patterns
- [ ] Error handling examples
- [ ] Performance characteristics

### **11. Dependencies**
- [ ] `reqwest` - HTTP client (with rustls-tls, no default features)
- [ ] `tokio` - Async runtime (with time feature)
- [ ] `anyhow` - Error handling
- [ ] Optional: `tracing` - Structured logging

### **12. Optional Features**
- [ ] **Logging feature** - Enable tracing/logging (off by default)
- [ ] **Custom headers** - Support Authorization, User-Agent, etc.
- [ ] **Custom status codes** - Accept non-2xx as success (e.g., 404 for some APIs)
- [ ] **Retry strategy** - Linear, exponential, or custom
- [ ] **Jitter** - Add randomness to backoff to prevent thundering herd
- [ ] **Circuit breaker** - Stop retrying if consistently failing

### **13. API Design**

#### **Simple API (current):**
```rust
pub async fn poll_health(
    url: &str,
    max_attempts: usize,
    initial_delay_ms: u64,
    backoff_multiplier: f64,
) -> Result<()>
```

#### **Builder API (future):**
```rust
pub struct HealthPoller {
    url: String,
    max_attempts: usize,
    initial_delay: Duration,
    backoff_multiplier: f64,
    timeout: Duration,
    client: reqwest::Client,
}

impl HealthPoller {
    pub fn new(url: impl Into<String>) -> Self
    pub fn max_attempts(mut self, n: usize) -> Self
    pub fn initial_delay(mut self, d: Duration) -> Self
    pub fn backoff_multiplier(mut self, m: f64) -> Self
    pub fn timeout(mut self, t: Duration) -> Self
    pub async fn poll(self) -> Result<()>
}

// Usage:
HealthPoller::new("http://localhost:7833/health")
    .max_attempts(30)
    .initial_delay(Duration::from_millis(200))
    .backoff_multiplier(1.5)
    .poll()
    .await?;
```

### **14. Presets (Optional)**
```rust
pub mod presets {
    pub const FAST: HealthPollerConfig = HealthPollerConfig {
        max_attempts: 10,
        initial_delay_ms: 100,
        backoff_multiplier: 1.2,
    };
    
    pub const NORMAL: HealthPollerConfig = HealthPollerConfig {
        max_attempts: 30,
        initial_delay_ms: 200,
        backoff_multiplier: 1.5,
    };
    
    pub const SLOW: HealthPollerConfig = HealthPollerConfig {
        max_attempts: 50,
        initial_delay_ms: 500,
        backoff_multiplier: 1.3,
    };
}
```

### **15. Cross-Platform Considerations**
- [ ] Works on Linux, macOS, Windows
- [ ] Handle platform-specific network quirks
- [ ] Test on all platforms in CI

---

## 🎯 Priority Levels

### **P0 (Must Have - Current Implementation):**
- ✅ Basic HTTP polling
- ✅ Exponential backoff
- ✅ Error handling
- ✅ Async implementation

### **P1 (Should Have - Next Iteration):**
- [ ] Structured logging (tracing)
- [ ] Better error messages
- [ ] Max delay cap
- [ ] Connection pooling optimization

### **P2 (Nice to Have - Future):**
- [ ] Builder API
- [ ] Presets
- [ ] Custom headers
- [ ] Jitter

### **P3 (Optional - If Needed):**
- [ ] Circuit breaker
- [ ] Custom retry strategies
- [ ] HTTPS with self-signed certs

---

## ✅ Current Status

Based on the existing code, you have:
- ✅ Basic [poll_health()](cci:1://file:///home/vince/Projects/llama-orch/bin/96_lifecycle/health-poll/src/lib.rs:10:0-87:1) function
- ✅ Exponential backoff
- ✅ Async implementation
- ✅ Error handling with anyhow
- ✅ Configurable parameters

**Missing (Priority 1):**
- ⚠️ No max delay cap (could wait forever with high multiplier)
- ⚠️ No logging/observability
- ⚠️ No tests
- ⚠️ Limited error context