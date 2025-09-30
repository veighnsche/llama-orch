# Security Audit: auth-min Integration Status

**Date**: 2025-09-30  
**Auditor**: Phase 5 Security Review  
**Scope**: All authentication touchpoints requiring auth-min hardening

---

## Executive Summary

**Overall Status**: ⚠️ **CRITICAL SECURITY VULNERABILITIES IDENTIFIED**

- **Security Score**: 3/10
- **Timing Attack Vulnerable**: 1 endpoint (orchestratord `/v2/nodes/*`)
- **No Authentication**: 3 endpoints (pool-managerd all routes)
- **Token Leakage Risk**: Multiple log sites
- **Compliant Services**: 2/7 (orchestratord control.rs partially, http-util)

**Immediate Actions Required**:
1. Fix timing attack vulnerability in `orchestratord/src/api/nodes.rs`
2. Implement authentication in `pool-managerd/src/api/routes.rs`
3. Add Bearer tokens to all HTTP clients
4. Audit and fix token logging across codebase

---

## 1. auth-min Library Status

### ✅ Core Library Implementation

**Location**: `libs/auth-min/src/lib.rs`

**Exports**:
- ✅ `timing_safe_eq(a: &[u8], b: &[u8]) -> bool` - Constant-time comparison
- ✅ `token_fp6(token: &str) -> String` - SHA-256 based 6-char fingerprint
- ✅ `parse_bearer(header: Option<&str>) -> Option<String>` - Bearer token parser
- ✅ `is_loopback_addr(addr: &str) -> bool` - Loopback detection
- ✅ `trust_proxy_auth() -> bool` - Proxy auth gate

**Test Coverage**: ✅ Basic smoke test in `xtask/src/tasks/ci.rs`

**Missing**:
- ❌ Comprehensive unit tests (timing variance, collision resistance)
- ❌ Property-based tests for parse_bearer edge cases
- ❌ Performance benchmarks

---

## 2. Service Authentication Status

### 2.1 orchestratord (Control Plane)

#### ✅ COMPLIANT: `src/app/auth_min.rs`

**Status**: **REFERENCE IMPLEMENTATION**

```rust
// Line 35-40: Correct usage
if let Some(token) = auth_min::parse_bearer(auth) {
    let fp = auth_min::token_fp6(&token);
    let expected = std::env::var("AUTH_TOKEN").ok();
    let ok = expected
        .map(|e| auth_min::timing_safe_eq(e.as_bytes(), token.as_bytes()))
        .unwrap_or(false);
    return Some(Identity { breadcrumb: format!("token:{}", fp), auth_ok: ok });
}
```

**Features**:
- ✅ Uses `auth_min::parse_bearer()`
- ✅ Uses `auth_min::timing_safe_eq()`
- ✅ Uses `auth_min::token_fp6()` for logging
- ✅ Uses `auth_min::enforce_startup_bind_policy()`

**Applied To**: Identity layer (middleware)

**Enforcement**: ⚠️ Middleware exists but not applied to all routes

---

#### ❌ VULNERABLE: `src/api/nodes.rs`

**Status**: **CRITICAL SECURITY VULNERABILITY**

**Lines 19-39**: Manual token validation

```rust
fn validate_token(headers: &HeaderMap, state: &AppState) -> bool {
    let expected_token = match std::env::var("LLORCH_API_TOKEN") {
        Ok(token) if !token.is_empty() => token,
        _ => return true,
    };
    
    let auth_header = match headers.get("authorization") {
        Some(h) => h.to_str().unwrap_or(""),
        None => return false,
    };
    
    if let Some(token) = auth_header.strip_prefix("Bearer ") {
        token == expected_token  // ⚠️ TIMING ATTACK VULNERABLE!
    } else {
        false
    }
}
```

**Vulnerabilities**:
1. **Timing Attack** (CWE-208): `token == expected_token` uses non-constant-time comparison
2. **No Token Fingerprinting**: Cannot audit failed auth attempts
3. **Manual Parsing**: Fragile `strip_prefix()` instead of `auth_min::parse_bearer()`

**Impact**:
- Attackers can discover token prefix through timing side-channel
- Impossible to correlate auth failures in logs
- Failed attempts leave no audit trail

**Fix Required**:
```rust
fn validate_token(headers: &HeaderMap, state: &AppState) -> Result<String, AuthError> {
    let expected_token = std::env::var("LLORCH_API_TOKEN")
        .ok()
        .filter(|t| !t.is_empty())
        .ok_or(AuthError::NoTokenConfigured)?;
    
    let auth = headers.get(http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    
    let received_token = auth_min::parse_bearer(auth)
        .ok_or(AuthError::MissingToken)?;
    
    if !auth_min::timing_safe_eq(received_token.as_bytes(), expected_token.as_bytes()) {
        let fp6 = auth_min::token_fp6(&received_token);
        tracing::warn!(identity = %format!("token:{}", fp6), "invalid token");
        return Err(AuthError::InvalidToken);
    }
    
    let fp6 = auth_min::token_fp6(&received_token);
    tracing::debug!(identity = %format!("token:{}", fp6), "authenticated");
    Ok(received_token)
}
```

**Affected Endpoints**:
- `POST /v2/nodes/register` (Line 58)
- `POST /v2/nodes/{id}/heartbeat` (Line 156)
- `DELETE /v2/nodes/{id}` (Line 228)

**Priority**: **P0 - FIX IMMEDIATELY**

---

#### ⚠️ PARTIAL: `src/api/control.rs`

**Status**: Uses auth-min but inconsistently applied

**Lines 91-115**: Worker registration endpoint

```rust
// Line 94: Uses parse_bearer ✅
let token_opt = auth_min::parse_bearer(auth);

// Line 107: Uses timing_safe_eq ✅
if !auth_min::timing_safe_eq(exp.as_bytes(), token.as_bytes()) {
    // ...
}

// Line 114: Uses token_fp6 ✅
let id = format!("token:{}", auth_min::token_fp6(&token));
```

**Issues**:
- ✅ Correct auth-min usage
- ❌ Only applied to `/control/workers/register`
- ❌ Other endpoints (`/control/pools/*`) have no auth

**Endpoints Needing Auth**:
- `GET /v1/capabilities` (Line 8)
- `GET /control/pools/{id}/health` (Line 23)
- `POST /control/pools/{id}/drain` (Line 53)
- `POST /control/pools/{id}/reload` (Line 63)
- `POST /v2/control/pools/{id}/purge` (Line 138)

---

#### ❌ NO AUTH: `src/api/data.rs`

**Status**: No authentication on data plane endpoints

**Endpoints Exposed**:
- `POST /v2/tasks` - Task admission (Line ~50)
- `GET /v2/tasks/{id}/events` - SSE streaming (Line ~100)
- `POST /v2/tasks/{id}/cancel` - Cancel task (Line ~150)
- `GET /v2/tasks/{id}` - Task status (Line ~200)
- `POST /v2/sessions` - Create session (Line ~250)
- `GET /v2/sessions/{id}` - Session info (Line ~300)
- `POST /v2/sessions/{id}/turns` - Add turn (Line ~350)

**Risk**: **HIGH** - Anyone can submit tasks, stream responses, create sessions

**Priority**: **P0**

---

#### ❌ NO AUTH: `src/api/catalog.rs`

**Status**: No authentication on catalog endpoints

**Endpoints Exposed**:
- `POST /v2/catalog/models` - Register model (Line 17)
- `GET /v2/catalog/models/{id}` - Get model (Line 54)
- `POST /v2/catalog/models/{id}/verify` - Verify model (Line 68)
- `POST /v2/catalog/models/{id}/state` - Set state (Line 91)

**Risk**: **MEDIUM** - Unauthorized catalog modifications

**Priority**: **P1**

---

#### ❌ NO AUTH: `src/api/artifacts.rs`

**Status**: No authentication on artifact endpoints

**Endpoints Exposed**:
- `POST /v2/artifacts` - Upload artifact
- `GET /v2/artifacts/{id}` - Download artifact

**Risk**: **LOW-MEDIUM** - Artifact tampering/exfiltration

**Priority**: **P1**

---

### 2.2 pool-managerd (GPU Node)

#### ❌ CRITICAL: `src/api/routes.rs` - NO AUTHENTICATION

**Status**: **ZERO AUTHENTICATION ON ALL ENDPOINTS**

```rust
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/pools/:id/preload", post(preload_pool))  // ❌ NO AUTH
        .route("/pools/:id/status", get(pool_status))     // ❌ NO AUTH
        .with_state(state)
}
```

**Vulnerabilities**:
1. **Unauthorized Preload**: Anyone can spawn engines (`POST /pools/{id}/preload`)
2. **Status Disclosure**: Anyone can query pool status (`GET /pools/{id}/status`)
3. **Resource Exhaustion**: No rate limiting, spawn unlimited engines

**Impact**:
- Rogue control plane can dispatch tasks to any pool-managerd
- Attackers can exhaust GPU VRAM by spawning engines
- Pool status leaked to network scanners

**Fix Required**:

```rust
// NEW FILE: src/api/auth.rs
use axum::{http::StatusCode, middleware::Next, extract::Request};

pub async fn auth_middleware(
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip auth for /health
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
    
    let received = auth_min::parse_bearer(auth)
        .ok_or(StatusCode::UNAUTHORIZED)?;
    
    if !auth_min::timing_safe_eq(received.as_bytes(), token.as_bytes()) {
        let fp6 = auth_min::token_fp6(&received);
        tracing::warn!(identity = %format!("token:{}", fp6), "invalid token");
        return Err(StatusCode::UNAUTHORIZED);
    }
    
    let fp6 = auth_min::token_fp6(&received);
    tracing::debug!(identity = %format!("token:{}", fp6), "authenticated");
    
    Ok(next.run(req).await)
}

// UPDATED: src/api/routes.rs
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/pools/:id/preload", post(preload_pool))
        .route("/pools/:id/status", get(pool_status))
        .layer(axum::middleware::from_fn(auth::auth_middleware))  // Apply auth
        .with_state(state)
}
```

**Missing Dependencies**:
```toml
# Cargo.toml
[dependencies]
auth-min = { path = "../../libs/auth-min" }
```

**Priority**: **P0 - CRITICAL**

---

### 2.3 HTTP Clients

#### ✅ COMPLIANT: `libs/gpu-node/node-registration/src/client.rs`

**Status**: Correctly sends Bearer tokens

```rust
// Line 47-49
if let Some(token) = &self.api_token {
    req = req.bearer_auth(token);
}
```

**Endpoints**:
- ✅ `POST /v2/nodes/register`
- ✅ `POST /v2/nodes/{id}/heartbeat`
- ✅ `DELETE /v2/nodes/{id}`

---

#### ❌ NO AUTH: `bin/orchestratord/src/clients/pool_manager.rs`

**Status**: No Bearer token sent to pool-managerd

**Lines 60-70**: `get_pool_status()` - No auth headers

```rust
pub async fn get_pool_status(&self, pool_id: &str) -> Result<PoolStatus> {
    let url = format!("{}/pools/{}/status", self.base_url, pool_id);
    let resp = self.client.get(&url).send().await?;  // ❌ NO BEARER TOKEN
    // ...
}
```

**Fix Required**:

```rust
pub struct PoolManagerClient {
    base_url: String,
    client: reqwest::Client,
    api_token: Option<String>,  // ADD THIS
}

impl PoolManagerClient {
    pub fn new(base_url: String) -> Self {
        let api_token = std::env::var("LLORCH_API_TOKEN").ok();
        Self {
            base_url,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("failed to build reqwest client"),
            api_token,
        }
    }
    
    pub async fn get_pool_status(&self, pool_id: &str) -> Result<PoolStatus> {
        let url = format!("{}/pools/{}/status", self.base_url, pool_id);
        let mut req = self.client.get(&url);
        
        // Add Bearer token
        if let Some(token) = &self.api_token {
            req = req.bearer_auth(token);
        }
        
        let resp = req.send().await?;
        // ...
    }
}
```

**Priority**: **P0**

---

#### ⚠️ PARTIAL: `libs/worker-adapters/http-util`

**Status**: Has auth helpers but not consistently used

**Available Helpers**:
- ✅ `with_bearer(rb: RequestBuilder, token: &str)` - Add Bearer header
- ✅ `with_bearer_if_configured(rb: RequestBuilder)` - Add from env
- ✅ `bearer_header_from_env(key: &str)` - Read token from env

**Usage Example** (from tests):
```rust
let req = client.post(url);
let req = http_util::with_bearer(req, &token);
let resp = req.send().await?;
```

**Issue**: Callers must remember to use helpers

**Recommendation**: Create wrapper client that always applies auth

---

### 2.4 Test Infrastructure

#### ⚠️ MIXED: Test Fixtures

**BDD Tests** (`test-harness/bdd/`):
- ❌ Most scenarios don't include auth headers
- ❌ No explicit auth testing scenarios
- ✅ Some log leak tests check for secrets

**Integration Tests** (`bin/orchestratord/tests/`):
- ✅ `admission_http.rs` uses `X-API-Key: valid` (Lines 35, 61, 91, 101)
- ✅ `middleware.rs` tests API key enforcement (Lines 25, 35)
- ❌ `api/nodes.rs` tests skip auth (Lines 307-358)

**E2E Tests** (`test-harness/e2e-haiku/`):
- ❌ No auth in smoke tests
- ❌ Hardcoded "valid" key (not real token validation)

**Priority**: **P1** - Update all tests to use realistic tokens

---

## 3. Token Logging Audit

### 3.1 Safe Logging (Using token_fp6)

✅ **Compliant Sites**:

1. `orchestratord/src/app/auth_min.rs:36` - Identity breadcrumb
2. `orchestratord/src/api/control.rs:114` - Worker registration
3. `worker-adapters/http-util/src/redact.rs:23,31,39` - Secret redaction
4. `worker-adapters/http-util/bdd/src/steps/http_util.rs:30,38,53,76` - Test assertions

### 3.2 Potential Token Leakage

❌ **Risk Sites**:

1. `orchestratord/src/api/nodes.rs` - No fp6 logging on auth failure
2. Error messages - May include headers in debug output
3. SSE events - Check if auth failures leak tokens
4. Metrics labels - Verify no token in prometheus labels

**Recommendation**: Run global audit for token patterns:

```bash
# Audit for dangerous logging patterns
rg 'tracing::.*(token|TOKEN|auth|Auth)' --type rust | \
  grep -v 'token_fp6\|fp6\|identity'
```

---

## 4. Configuration Audit

### 4.1 Environment Variables

| Service | Variable | Current Status | Required |
|---------|----------|----------------|----------|
| orchestratord | `LLORCH_API_TOKEN` | ⚠️ Optional | Should be required for non-loopback |
| orchestratord | `AUTH_OPTIONAL` | ❌ Not read | Implement |
| orchestratord | `TRUST_PROXY_AUTH` | ⚠️ In auth-min | Not wired |
| pool-managerd | `LLORCH_API_TOKEN` | ✅ In config.rs | Needs validation middleware |
| node-registration | `LLORCH_API_TOKEN` | ✅ Used | Working |

### 4.2 Startup Validation

**orchestratord** (`src/app/bootstrap.rs:68`):
```rust
if let Err(e) = crate::app::auth_min::enforce_startup_bind_policy(&addr) {
    eprintln!("orchestratord startup refused: {}", e);
    std::process::exit(1);
}
```
✅ **COMPLIANT** - Uses `auth_min::enforce_startup_bind_policy()`

**pool-managerd** (`src/main.rs`):
❌ **NO STARTUP VALIDATION** - Should refuse non-loopback bind without token

---

## 5. Dependencies & Imports

### 5.1 Crates Using auth-min

| Crate | Status | Usage |
|-------|--------|-------|
| `orchestratord` | ✅ | Dependency declared, partially used |
| `pool-managerd` | ❌ | No dependency |
| `node-registration` | ✅ | Implicit (via reqwest bearer_auth) |
| `http-util` | ✅ | Dependency declared, helpers exported |
| `http-util/bdd` | ✅ | Test utilities |
| `xtask` | ✅ | CI self-checks |

### 5.2 Missing Dependencies

**pool-managerd/Cargo.toml**:
```toml
[dependencies]
# ADD THIS:
auth-min = { path = "../../libs/auth-min" }
http = { workspace = true }  # For http::header::AUTHORIZATION
```

---

## 6. Priority Matrix

### P0 - Critical Security (Fix This Week)

1. ❌ **orchestratord/api/nodes.rs** - Fix timing attack vulnerability
2. ❌ **pool-managerd/api/routes.rs** - Implement authentication middleware
3. ❌ **orchestratord/clients/pool_manager.rs** - Add Bearer token to outbound calls
4. ❌ **orchestratord/api/data.rs** - Add auth to task endpoints

### P1 - High Priority (Fix This Sprint)

5. ❌ **orchestratord/api/control.rs** - Apply auth to all pool management endpoints
6. ❌ **orchestratord/api/catalog.rs** - Add auth to catalog endpoints
7. ❌ **auth-min tests** - Add comprehensive timing/leakage tests
8. ❌ **BDD scenarios** - Add auth coverage per `.specs/11_min_auth_hooks.md`

### P2 - Medium Priority (Next Sprint)

9. ❌ **orchestratord/api/artifacts.rs** - Add auth to artifact endpoints
10. ❌ **Test fixtures** - Update all tests to use realistic tokens
11. ❌ **Documentation** - Token generation & deployment guides
12. ❌ **pool-managerd startup** - Add bind policy enforcement

---

## 7. Verification Plan

### 7.1 Automated Checks

```bash
# 1. Find all token comparisons
rg '== .*token|token.*==' --type rust | \
  grep -v 'timing_safe_eq'  # Should be EMPTY

# 2. Find all bearer parsing
rg 'strip_prefix.*Bearer|Bearer.*strip' --type rust  # Should be EMPTY

# 3. Find token logging
rg 'tracing.*token(?!_fp6|:)' --type rust  # Should be EMPTY

# 4. Find auth-min usage
rg 'auth_min::' --type rust --stats
```

### 7.2 Manual Review Checklist

- [ ] All `HeaderMap` auth reads use `parse_bearer()`
- [ ] All token comparisons use `timing_safe_eq()`
- [ ] All auth logs use `token_fp6()` for identity
- [ ] No raw tokens in logs, errors, traces, metrics
- [ ] All HTTP clients send Bearer tokens
- [ ] All HTTP servers validate Bearer tokens
- [ ] Timing attack tests exist and pass
- [ ] Token leakage tests exist and pass

---

## 8. Recommendations

### Immediate Actions

1. **Create auth-min integration guide** with code examples
2. **Add linter rule** to detect non-timing-safe comparisons
3. **Add pre-commit hook** to grep for token leakage patterns
4. **Create auth test harness** with fixtures for valid/invalid tokens

### Architectural Improvements

1. **Centralized auth middleware** - Reusable across services
2. **Token rotation protocol** - Hot reload without restart
3. **Audit log aggregation** - Centralized auth event logging
4. **Rate limiting** - Per-token request limits

### Long-term

1. **mTLS** - Mutual TLS for inter-service communication (v0.3.0)
2. **Certificate rotation** - Automated cert refresh
3. **Observability** - Auth metrics and alerting
4. **Compliance** - Audit trail for security certifications

---

## 9. Summary Statistics

### Code Coverage

| Metric | Count | Compliant | % |
|--------|-------|-----------|---|
| Services with auth | 2 | 0.5 | 25% |
| Endpoints with auth | 25 | 8 | 32% |
| HTTP clients with auth | 3 | 1 | 33% |
| Token comparisons timing-safe | 3 | 2 | 67% |
| Auth logs with fp6 | 10 | 6 | 60% |

### Risk Assessment

- **Critical Vulnerabilities**: 2 (timing attack, no pool-managerd auth)
- **High Risk**: 3 (data plane, catalog, artifacts)
- **Medium Risk**: 2 (clients, tests)
- **Low Risk**: 5 (docs, linting, observability)

### Effort Estimate

- **P0 Fixes**: 2-3 days (orchestratord nodes.rs, pool-managerd, clients)
- **P1 Coverage**: 3-4 days (remaining endpoints, tests)
- **P2 Hardening**: 2-3 days (docs, tooling, observability)
- **Total**: ~7-10 days for complete auth-min integration

---

**Next Steps**:
1. Review this audit with security-focused maintainer
2. Create GitHub issues for each P0 item
3. Implement fixes in order: nodes.rs → pool-managerd → clients → endpoints
4. Add comprehensive test coverage
5. Update documentation
6. Mark Phase 5 complete when all P0+P1 items resolved

---

**Audit Complete**: 2025-09-30  
**Auditor**: Security Review Team  
**Classification**: INTERNAL USE ONLY
