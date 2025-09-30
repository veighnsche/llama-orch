# Phase 5 Security Fix Checklist

**Owner**: Security & Backend Teams  
**Timeline**: 1 week (40 hours)  
**Status**: 🔴 NOT STARTED

---

## P0: Critical Security Fixes (Week 1, 18 hours)

### ☐ Task 1: Fix Timing Attack in orchestratord (2 hours)

**File**: `bin/orchestratord/src/api/nodes.rs`

**Changes**:
```diff
- fn validate_token(headers: &HeaderMap, state: &AppState) -> bool {
-     let expected_token = match std::env::var("LLORCH_API_TOKEN") {
-         Ok(token) if !token.is_empty() => token,
-         _ => return true,
-     };
-     let auth_header = match headers.get("authorization") {
-         Some(h) => h.to_str().unwrap_or(""),
-         None => return false,
-     };
-     if let Some(token) = auth_header.strip_prefix("Bearer ") {
-         token == expected_token  // ❌ TIMING ATTACK!
-     } else {
-         false
-     }
- }

+ fn validate_token(headers: &HeaderMap) -> Result<String, (StatusCode, &'static str)> {
+     let expected = std::env::var("LLORCH_API_TOKEN")
+         .ok()
+         .filter(|t| !t.is_empty())
+         .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "no token configured"))?;
+     
+     let auth = headers.get(http::header::AUTHORIZATION)
+         .and_then(|v| v.to_str().ok());
+     
+     let received = auth_min::parse_bearer(auth)
+         .ok_or((StatusCode::UNAUTHORIZED, "missing bearer token"))?;
+     
+     if !auth_min::timing_safe_eq(received.as_bytes(), expected.as_bytes()) {
+         let fp6 = auth_min::token_fp6(&received);
+         tracing::warn!(identity = %format!("token:{}", fp6), "invalid token");
+         return Err((StatusCode::UNAUTHORIZED, "invalid token"));
+     }
+     
+     let fp6 = auth_min::token_fp6(&received);
+     tracing::debug!(identity = %format!("token:{}", fp6), "authenticated");
+     Ok(received)
+ }
```

**Test**:
```bash
cargo test -p orchestratord test_timing_attack_resistance
```

**Done**: ☐

---

### ☐ Task 2: Add Timing Attack Test (2 hours)

**File**: `bin/orchestratord/tests/security_timing.rs` (NEW)

```rust
use std::time::Instant;

fn measure_comparison(correct: &str, attempt: &str) -> u128 {
    let start = Instant::now();
    for _ in 0..10000 {
        let _ = auth_min::timing_safe_eq(correct.as_bytes(), attempt.as_bytes());
    }
    start.elapsed().as_nanos() / 10000
}

#[test]
fn timing_safe_eq_constant_time() {
    let token = "a".repeat(64);
    let wrong_early = "b".repeat(64);  // Wrong at position 0
    let wrong_late = format!("{}b", "a".repeat(63));  // Wrong at position 63
    
    let time_early = measure_comparison(&token, &wrong_early);
    let time_late = measure_comparison(&token, &wrong_late);
    
    let variance = (time_early as i128 - time_late as i128).abs() as f64 / time_early as f64;
    assert!(variance < 0.1, "timing leak detected: {}% variance", variance * 100.0);
}

#[test]
fn node_registration_rejects_wrong_token_safely() {
    // Test that auth failures don't leak timing
}
```

**Test**:
```bash
cargo test -p orchestratord security_timing
```

**Done**: ☐

---

### ☐ Task 3: Implement pool-managerd Authentication (3 hours)

**File**: `bin/pool-managerd/Cargo.toml`

```diff
 [dependencies]
+auth-min = { path = "../../libs/auth-min" }
+http = { workspace = true }
```

**File**: `bin/pool-managerd/src/api/auth.rs` (NEW)

```rust
use axum::{extract::Request, http::StatusCode, middleware::Next, response::Response};

pub async fn auth_middleware(req: Request, next: Next) -> Result<Response, StatusCode> {
    // Skip auth for health checks
    if req.uri().path() == "/health" {
        return Ok(next.run(req).await);
    }
    
    let token = std::env::var("LLORCH_API_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
        .ok_or(StatusCode::INTERNAL_SERVER_ERROR)?;
    
    let auth = req.headers()
        .get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    
    let received = auth_min::parse_bearer(auth).ok_or(StatusCode::UNAUTHORIZED)?;
    
    if !auth_min::timing_safe_eq(received.as_bytes(), token.as_bytes()) {
        let fp6 = auth_min::token_fp6(&received);
        tracing::warn!(identity = %format!("token:{}", fp6), "auth failed");
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    let fp6 = auth_min::token_fp6(&received);
    tracing::debug!(identity = %format!("token:{}", fp6), "authenticated");
    
    Ok(next.run(req).await)
}
```

**File**: `bin/pool-managerd/src/api/mod.rs`

```diff
+pub mod auth;
 pub mod routes;
```

**File**: `bin/pool-managerd/src/api/routes.rs`

```diff
+use super::auth;

 pub fn create_router(state: AppState) -> Router {
     Router::new()
         .route("/health", get(health))
         .route("/pools/:id/preload", post(preload_pool))
         .route("/pools/:id/status", get(pool_status))
+        .layer(axum::middleware::from_fn(auth::auth_middleware))
         .with_state(state)
 }
```

**Done**: ☐

---

### ☐ Task 4: Test pool-managerd Authentication (1 hour)

**File**: `bin/pool-managerd/tests/auth_integration.rs` (NEW)

```rust
#[tokio::test]
async fn preload_requires_valid_token() {
    std::env::set_var("LLORCH_API_TOKEN", "test-token-123");
    
    let app = create_test_app();
    
    // Missing token → 401
    let resp = app.clone().oneshot(
        Request::post("/pools/test/preload")
            .body(Body::from("{}")).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    
    // Invalid token → 401
    let resp = app.clone().oneshot(
        Request::post("/pools/test/preload")
            .header("Authorization", "Bearer wrong")
            .body(Body::from("{}")).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    
    // Valid token → proceeds (may fail for other reasons)
    let resp = app.clone().oneshot(
        Request::post("/pools/test/preload")
            .header("Authorization", "Bearer test-token-123")
            .body(Body::from("{}")).unwrap()
    ).await.unwrap();
    assert_ne!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn health_does_not_require_token() {
    std::env::set_var("LLORCH_API_TOKEN", "test-token-123");
    
    let app = create_test_app();
    let resp = app.oneshot(
        Request::get("/health").body(Body::empty()).unwrap()
    ).await.unwrap();
    
    assert_eq!(resp.status(), StatusCode::OK);
}
```

**Test**:
```bash
cargo test -p pool-managerd auth_integration
```

**Done**: ☐

---

### ☐ Task 5: Add Bearer Token to HTTP Client (1 hour)

**File**: `bin/orchestratord/src/clients/pool_manager.rs`

```diff
 pub struct PoolManagerClient {
     base_url: String,
     client: reqwest::Client,
+    api_token: Option<String>,
 }

 impl PoolManagerClient {
     pub fn new(base_url: String) -> Self {
+        let api_token = std::env::var("LLORCH_API_TOKEN").ok();
         Self {
             base_url,
             client: reqwest::Client::builder()
                 .timeout(std::time::Duration::from_secs(5))
                 .build()
                 .expect("failed to build reqwest client"),
+            api_token,
         }
     }
     
     pub async fn get_pool_status(&self, pool_id: &str) -> Result<PoolStatus> {
         let url = format!("{}/pools/{}/status", self.base_url, pool_id);
-        let resp = self.client.get(&url).send().await?;
+        let mut req = self.client.get(&url);
+        if let Some(token) = &self.api_token {
+            req = req.bearer_auth(token);
+        }
+        let resp = req.send().await?;
         // ...
     }
 }
```

**Done**: ☐

---

### ☐ Task 6: E2E Test with Authentication (3 hours)

**File**: `test-harness/e2e-cloud/tests/auth_e2e.rs` (NEW)

```rust
#[tokio::test]
async fn test_full_flow_with_authentication() {
    // 1. Start pool-managerd with token
    std::env::set_var("LLORCH_API_TOKEN", "test-secret-abc123");
    let pool_daemon = start_pool_managerd().await;
    
    // 2. Start orchestratord with same token
    let orchestrator = start_orchestratord().await;
    
    // 3. Register node (should succeed with matching token)
    let registration = register_node(&orchestrator.url, "test-secret-abc123").await;
    assert!(registration.is_ok());
    
    // 4. Try with wrong token (should fail)
    let bad_registration = register_node(&orchestrator.url, "wrong-token").await;
    assert!(bad_registration.is_err());
    
    // 5. Heartbeat with correct token
    let heartbeat = send_heartbeat(&orchestrator.url, "test-secret-abc123").await;
    assert!(heartbeat.is_ok());
    
    // 6. Query pool status from orchestrator
    let status = orchestrator.query_pool_status("pool-0").await;
    assert!(status.is_ok());
}
```

**Done**: ☐

---

### ☐ Task 7: Code Review (4 hours)

**Checklist**:
- [ ] All token comparisons use `auth_min::timing_safe_eq()`
- [ ] All Bearer parsing uses `auth_min::parse_bearer()`
- [ ] All auth logs use `auth_min::token_fp6()` for identity
- [ ] No raw tokens in logs, errors, or debug output
- [ ] Tests cover timing attack resistance
- [ ] Tests cover valid/invalid/missing token scenarios
- [ ] Documentation updated with security notes

**Reviewers**: Security team + Senior backend engineer

**Done**: ☐

---

### ☐ Task 8: Deploy & Monitor (2 hours)

**Deployment Steps**:
1. Generate secure token: `openssl rand -hex 32`
2. Set `LLORCH_API_TOKEN` on both orchestratord and pool-managerd
3. Deploy orchestratord with updated code
4. Deploy pool-managerd with updated code
5. Verify authentication working in logs
6. Monitor for auth failures

**Monitoring**:
```bash
# Check for auth failures
journalctl -u orchestratord | grep "auth_failed"
journalctl -u pool-managerd | grep "auth_failed"

# Verify token fingerprints in logs
journalctl -u orchestratord | grep "identity=token:"
```

**Done**: ☐

---

## P1: Complete Coverage (Week 2, 22 hours)

### ☐ Task 9: Add Auth to Data Plane (4 hours)

**File**: `bin/orchestratord/src/api/data.rs`

Apply auth to:
- `POST /v2/tasks`
- `GET /v2/tasks/{id}/events`
- `POST /v2/tasks/{id}/cancel`
- `POST /v2/sessions`
- `GET /v2/sessions/{id}`

**Pattern**: Create middleware or use existing auth_min layer

**Done**: ☐

---

### ☐ Task 10: Add Auth to Control Plane (2 hours)

**File**: `bin/orchestratord/src/api/control.rs`

Apply auth to:
- `GET /v1/capabilities`
- `GET /control/pools/{id}/health`
- `POST /control/pools/{id}/drain`
- `POST /control/pools/{id}/reload`

**Done**: ☐

---

### ☐ Task 11: Add Auth to Catalog/Artifacts (2 hours)

**Files**:
- `bin/orchestratord/src/api/catalog.rs`
- `bin/orchestratord/src/api/artifacts.rs`

Apply auth to all endpoints

**Done**: ☐

---

### ☐ Task 12: auth-min Comprehensive Tests (4 hours)

**File**: `libs/auth-min/src/lib.rs`

Add test module:
```rust
#[cfg(test)]
mod tests {
    #[test] fn timing_safe_eq_constant_time() { }
    #[test] fn timing_safe_eq_equal() { }
    #[test] fn timing_safe_eq_unequal() { }
    #[test] fn token_fp6_deterministic() { }
    #[test] fn token_fp6_collision_resistance() { }
    #[test] fn parse_bearer_valid() { }
    #[test] fn parse_bearer_whitespace() { }
    #[test] fn parse_bearer_empty() { }
    #[test] fn parse_bearer_no_bearer_prefix() { }
    #[test] fn is_loopback_ipv4() { }
    #[test] fn is_loopback_ipv6() { }
}
```

**Done**: ☐

---

### ☐ Task 13: BDD Auth Scenarios (6 hours)

**File**: `test-harness/bdd/features/auth.feature`

Implement scenarios from `.specs/11_min_auth_hooks.md`:
- Loopback + AUTH_OPTIONAL=true
- Loopback + AUTH_OPTIONAL=false
- Non-loopback without token (startup refused)
- Wrong token rejected
- Correct token accepted
- Worker registration without token

**Done**: ☐

---

### ☐ Task 14: Security Documentation (4 hours)

**Create**:
- `.docs/guides/TOKEN_GENERATION.md` - How to generate & store tokens
- `.docs/guides/DEPLOYMENT_SECURITY.md` - Security checklist for deployments
- `.docs/runbooks/SECURITY_INCIDENTS.md` - How to respond to auth failures

**Update**:
- `README.md` - Add security configuration section
- `CLOUD_PROFILE_MIGRATION_PLAN.md` - Update Phase 5 status

**Done**: ☐

---

## Final Verification

### ☐ Run Full Test Suite

```bash
# Format check
cargo fmt --all -- --check

# Clippy
cargo clippy --all-targets --all-features -- -D warnings

# All tests
cargo test --workspace --all-features -- --nocapture

# BDD tests
cargo test -p test-harness-bdd -- --nocapture

# Specific security tests
cargo test -p auth-min
cargo test -p orchestratord test_timing_attack_resistance
cargo test -p pool-managerd auth_integration
```

**Done**: ☐

---

### ☐ Security Audit Commands

```bash
# 1. Find non-timing-safe comparisons
rg 'token.*==' --type rust | grep -v 'timing_safe_eq'
# Should be EMPTY

# 2. Find manual Bearer parsing
rg 'strip_prefix.*Bearer' --type rust
# Should only show auth-min implementation

# 3. Find token logging
rg 'tracing.*token(?!_fp6|:)' --type rust
# Should be EMPTY

# 4. Verify auth-min usage
rg 'auth_min::' --type rust --stats
# Should show usage in all auth code
```

**Done**: ☐

---

### ☐ Manual Security Review

- [ ] All endpoints have appropriate authentication
- [ ] `/health` and `/metrics` exempt (as intended)
- [ ] All token comparisons timing-safe
- [ ] All auth failures logged with fp6
- [ ] No raw tokens in any logs
- [ ] Tests cover security edge cases
- [ ] Documentation complete

**Done**: ☐

---

### ☐ Security Sign-off

- [ ] Security team reviewed code changes
- [ ] Penetration test passed (optional)
- [ ] Timing attack tests pass
- [ ] Token leakage tests pass
- [ ] E2E tests pass with auth enabled
- [ ] Documentation approved
- [ ] Ready for Phase 6

**Signed**: _________________  
**Date**: _________________

**Done**: ☐

---

## Progress Tracking

**P0 (Critical)**: ☐☐☐☐☐☐☐☐ (0/8 complete)  
**P1 (Coverage)**: ☐☐☐☐☐☐ (0/6 complete)  
**Verification**: ☐☐☐ (0/3 complete)

**Overall**: 0/17 tasks complete (0%)

**ETA**: _______________  
**Actual Complete**: _______________

---

## Quick Reference

**Key Files**:
```
libs/auth-min/src/lib.rs                    # Security library
bin/orchestratord/src/api/nodes.rs          # Fix timing attack
bin/pool-managerd/src/api/auth.rs           # NEW: Auth middleware
bin/pool-managerd/src/api/routes.rs         # Apply middleware
bin/orchestratord/src/clients/pool_manager.rs  # Add Bearer token
```

**Key Commands**:
```bash
# Test specific auth
cargo test -p auth-min
cargo test -p orchestratord security_timing
cargo test -p pool-managerd auth_integration

# Full dev loop
cargo xtask dev:loop

# Security audit
rg 'token.*==' --type rust | grep -v timing_safe_eq
```

**Documentation**:
- `.specs/12_auth-min-hardening.md` - Implementation spec
- `.docs/SECURITY_AUDIT_AUTH_MIN.md` - Full audit
- `.docs/PHASE5_SECURITY_FINDINGS.md` - Detailed findings
- `.docs/PHASE5_AUDIT_SUMMARY.md` - Executive summary
