# auth-min Security Hardening Specification

**Status**: Draft → Review  
**Date**: 2025-09-30  
**Owners**: @llama-orch-maintainers  
**Related**: `.specs/11_min_auth_hooks.md`, `.specs/00_llama-orch.md` §2.7

---

## Executive Summary

This specification extends the minimal auth hooks (AUTH-1001..AUTH-1008) with **hardened implementation requirements** for the `auth-min` library and its integration across all service boundaries. While named "min" for minimal dependencies and operational overhead, the implementation MUST use security best practices: constant-time comparisons, secure token fingerprinting, comprehensive audit logging, and defense-in-depth against timing attacks, token leaks, and unauthorized access.

**Scope**: Control plane (orchestratord), GPU nodes (pool-managerd), inter-service communication, HTTP clients, and test infrastructure.

---

## 1. Threat Model & Security Goals

### 1.1 Threats Mitigated (SEC-AUTH-1001)

The system MUST defend against:

1. **Timing Attacks** - Token comparison timing leaks reveal token prefixes
2. **Token Leakage** - Full tokens exposed in logs, metrics, error messages, or traces
3. **Replay Attacks** - Captured tokens used after deregistration or rotation
4. **Unauthorized Registration** - Rogue GPU nodes registering with control plane
5. **Unauthorized Dispatch** - Control plane sending tasks to unauthorized workers
6. **MitM Token Capture** - Plaintext tokens captured over non-loopback network
7. **Drive-by Scans** - Automated scanners exploiting open HTTP endpoints

### 1.2 Out of Scope (Non-Goals)

- User management, RBAC, multi-tenancy
- Token rotation protocol (manual restart required)
- Certificate pinning or mTLS (future: v0.3.0)
- Rate limiting beyond admission queue policy
- OAuth2/OIDC/SSO integration

---

## 2. Core Security Requirements

### 2.1 Constant-Time Token Comparison (SEC-AUTH-2001)

**Requirement**: All token comparisons MUST use timing-safe equality to prevent timing attacks.

**Implementation**:
```rust
// REQUIRED: Use auth_min::timing_safe_eq()
if auth_min::timing_safe_eq(received.as_bytes(), expected.as_bytes()) {
    // Valid token
}

// FORBIDDEN: Direct string comparison
if received == expected {  // ❌ VULNERABLE to timing attacks
    // This leaks timing information!
}
```

**Rationale**: String comparison (`==`) short-circuits on first mismatch, leaking position information through timing. Constant-time comparison examines all bytes regardless of match status.

**Test Coverage**: MUST include timing attack regression tests measuring comparison duration variance.

### 2.2 Secure Token Fingerprinting (SEC-AUTH-2002)

**Requirement**: Token fingerprints for logging MUST be cryptographically derived (SHA-256) and limited to 6 hex characters (fp6).

**Implementation**:
```rust
// REQUIRED: Use auth_min::token_fp6()
let fp6 = auth_min::token_fp6(token);  // Returns "a3f2c1"
tracing::info!(identity = %format!("token:{}", fp6), "request authenticated");

// FORBIDDEN: Direct token logging
tracing::info!(token = %token, "authenticated");  // ❌ LEAKS TOKEN
```

**Properties**:
- **Non-reversible**: SHA-256 ensures one-way derivation
- **Collision resistance**: 24-bit space (16.7M combinations) sufficient for audit correlation
- **Log-safe**: 6 chars concise enough for grep/correlation without PII concerns

### 2.3 Bearer Token Parsing (SEC-AUTH-2003)

**Requirement**: All `Authorization` header parsing MUST use `auth_min::parse_bearer()` with strict validation.

**Implementation**:
```rust
// REQUIRED: Use auth_min::parse_bearer()
let auth = headers.get(http::header::AUTHORIZATION)
    .and_then(|v| v.to_str().ok());
let token = auth_min::parse_bearer(auth);

// FORBIDDEN: Manual string manipulation
let token = auth_header.strip_prefix("Bearer ");  // ❌ FRAGILE
```

**Validation Rules**:
- MUST accept `Authorization: Bearer <token>` format
- MUST trim whitespace around token
- MUST reject empty tokens after `Bearer` prefix
- MUST reject malformed headers (missing `Bearer`, empty values)

### 2.4 Loopback Detection (SEC-AUTH-2004)

**Requirement**: Loopback address detection MUST use `auth_min::is_loopback_addr()` to enforce bind policy.

**Implementation**:
```rust
// REQUIRED: Use auth_min::is_loopback_addr()
if !auth_min::is_loopback_addr(&bind_addr) && api_token.is_none() {
    return Err("refusing non-loopback bind without AUTH_TOKEN");
}
```

**Supported Formats**:
- IPv4: `127.0.0.1`, `127.0.0.1:8080`
- IPv6: `::1`, `[::1]`, `[::1]:8080`

---

## 3. Service Integration Requirements

### 3.1 orchestratord Control Plane (SEC-AUTH-3001)

**Endpoints Requiring Authentication**:

| Endpoint | Method | Token Required | Notes |
|----------|--------|----------------|-------|
| `/v2/nodes/register` | POST | **YES** | Node registration |
| `/v2/nodes/{id}/heartbeat` | POST | **YES** | Health reporting |
| `/v2/nodes/{id}` | DELETE | **YES** | Deregistration |
| `/v2/nodes` | GET | **YES** | List nodes (admin) |
| `/v2/tasks` | POST | **YES** | Task admission |
| `/v2/tasks/{id}/events` | GET | **YES** | SSE streaming |
| `/v2/tasks/{id}/cancel` | POST | **YES** | Cancel task |
| `/v1/capabilities` | GET | **YES** | Capability discovery |
| `/control/pools/{id}/health` | GET | **YES** | Pool health |
| `/control/pools/{id}/drain` | POST | **YES** | Drain pool |
| `/metrics` | GET | **NO** | Prometheus scrape exempt |
| `/health` | GET | **NO** | Liveness probe exempt |

**Implementation Location**: `bin/orchestratord/src/api/`

**Current Status**:
- ❌ `api/nodes.rs` uses manual validation (INSECURE)
- ✅ `app/auth_min.rs` uses auth-min library correctly (REFERENCE IMPL)
- ❌ `api/control.rs` uses auth-min but not consistently
- ❌ `api/data.rs` no auth validation
- ❌ `api/catalog.rs` no auth validation
- ❌ `api/artifacts.rs` no auth validation

### 3.2 pool-managerd GPU Node (SEC-AUTH-3002)

**Endpoints Requiring Authentication**:

| Endpoint | Method | Token Required | Notes |
|----------|--------|----------------|-------|
| `/pools/{id}/preload` | POST | **YES** | Engine preload |
| `/pools/{id}/status` | GET | **YES** | Pool status query |
| `/pools/{id}/dispatch` | POST | **YES** | Task dispatch (future) |
| `/health` | GET | **NO** | Liveness probe exempt |

**Implementation Location**: `bin/pool-managerd/src/api/routes.rs`

**Current Status**:
- ❌ **NO AUTHENTICATION** on any endpoint (CRITICAL VULNERABILITY)
- ❌ No Bearer token validation middleware
- ❌ No `LLORCH_API_TOKEN` configuration

**Required Changes**:
1. Add `auth_min` dependency to `Cargo.toml`
2. Create auth middleware in `api/auth.rs`
3. Apply middleware to router (except `/health`)
4. Add `LLORCH_API_TOKEN` to config.rs
5. Add integration tests with valid/invalid tokens

### 3.3 HTTP Client Authentication (SEC-AUTH-3003)

**Services Making Outbound Requests**:

| Service | Target | Current Status | Required |
|---------|--------|----------------|----------|
| pool-managerd → orchestratord | `/v2/nodes/*` | ✅ Uses Bearer | Verify auth-min helpers |
| orchestratord → pool-managerd | `/pools/*/status` | ❌ No auth | Add Bearer token |
| CLI → orchestratord | `/v2/*` | ⚠️ Partial | Use SDK with token |
| Test harness → services | All endpoints | ❌ No auth | Add token fixtures |

**Implementation Pattern**:
```rust
use auth_min::with_bearer;

let client = reqwest::Client::new();
let req = client.post(url).json(&body);
let req = with_bearer(req, &token);  // Add Bearer token
let resp = req.send().await?;
```

---

## 4. Configuration & Environment Variables

### 4.1 Token Configuration (SEC-AUTH-4001)

**Environment Variables**:

| Variable | Service | Required | Default | Notes |
|----------|---------|----------|---------|-------|
| `LLORCH_API_TOKEN` | orchestratord | If non-loopback | None | Control plane token |
| `LLORCH_API_TOKEN` | pool-managerd | If cloud profile | None | Must match control plane |
| `AUTH_OPTIONAL` | orchestratord | No | `false` | Allow loopback without token |
| `TRUST_PROXY_AUTH` | orchestratord | No | `false` | Trust proxy-injected auth |

**Token Requirements**:
- MUST be at least 32 characters
- SHOULD be generated cryptographically (e.g., `openssl rand -hex 32`)
- MUST NOT be hardcoded in source or configs
- MUST be stored in secure environment or secret manager

**Example Generation**:
```bash
# Generate secure token
export LLORCH_API_TOKEN=$(openssl rand -hex 32)

# Store in systemd service
echo "Environment=LLORCH_API_TOKEN=$LLORCH_API_TOKEN" >> /etc/systemd/system/orchestratord.service
```

### 4.2 Bind Policy Enforcement (SEC-AUTH-4002)

**Startup Validation**:

```rust
// orchestratord startup (src/app/bootstrap.rs)
let bind_addr = std::env::var("ORCHD_ADDR").unwrap_or("0.0.0.0:8080".to_string());
auth_min::enforce_startup_bind_policy(&bind_addr)?;
```

**Rules**:
- Loopback bind (`127.0.0.1`, `::1`) → Token optional if `AUTH_OPTIONAL=true`
- Non-loopback bind (`0.0.0.0`, public IPs) → Token REQUIRED, startup MUST fail if missing
- Error message MUST be explicit and include remediation steps

---

## 5. Logging & Audit Requirements

### 5.1 Identity Breadcrumbs (SEC-AUTH-5001)

**Requirement**: All authenticated requests MUST log identity breadcrumb using fp6.

**Format**:
```rust
// Successful authentication
tracing::info!(
    identity = %format!("token:{}", auth_min::token_fp6(&token)),
    node_id = %node_id,
    event = "node_registered"
);

// Loopback request
tracing::info!(
    identity = "localhost",
    event = "task_admitted"
);

// Failed authentication
tracing::warn!(
    identity = %format!("token:{}", auth_min::token_fp6(&bad_token)),
    remote_addr = %remote_addr,
    event = "auth_failed",
    reason = "invalid_token"
);
```

**Required Fields**:
- `identity`: Either `"localhost"` or `"token:<fp6>"`
- `event`: High-level action (e.g., `node_registered`, `task_dispatched`)
- Context fields: `node_id`, `pool_id`, `task_id` as appropriate

### 5.2 Security Event Logging (SEC-AUTH-5002)

**Events Requiring Audit Logs**:

| Event | Level | Fields | Trigger |
|-------|-------|--------|---------|
| `auth_success` | INFO | identity, endpoint, method | Valid token |
| `auth_failed` | WARN | identity, endpoint, method, reason | Invalid/missing token |
| `startup_bind_refused` | ERROR | bind_addr, reason | Non-loopback without token |
| `node_registered` | INFO | identity, node_id, pools | Node registration |
| `node_deregistered` | INFO | identity, node_id, reason | Graceful shutdown |
| `token_rotation_detected` | WARN | old_fp6, new_fp6 | Different token seen |

### 5.3 Secret Redaction (SEC-AUTH-5003)

**Requirement**: All log output MUST redact tokens using `auth_min::token_fp6()`.

**Covered Contexts**:
- HTTP request/response logs
- Error messages
- Debug traces
- Metrics labels
- SSE event payloads

**Test Coverage**: BDD scenarios MUST assert no token leakage in logs.

---

## 6. Testing Requirements

### 6.1 Unit Tests (SEC-AUTH-6001)

**auth-min Library Tests**:
- ✅ `timing_safe_eq()` correctness (equal/unequal)
- ✅ `timing_safe_eq()` constant-time property (variance < 10%)
- ✅ `token_fp6()` determinism (same token → same fp6)
- ✅ `token_fp6()` collision resistance (different tokens → different fp6)
- ✅ `parse_bearer()` valid formats
- ✅ `parse_bearer()` invalid/malformed headers
- ✅ `is_loopback_addr()` IPv4/IPv6 detection

**Service Tests**:
- ❌ orchestratord: Valid token → 200
- ❌ orchestratord: Invalid token → 401 BAD_TOKEN
- ❌ orchestratord: Missing token → 401 MISSING_TOKEN
- ❌ pool-managerd: Authentication (NOT IMPLEMENTED)

### 6.2 Integration Tests (SEC-AUTH-6002)

**E2E Scenarios**:
- ❌ Multi-node registration with valid token
- ❌ Registration rejected with invalid token
- ❌ Heartbeat with token rotation
- ❌ Task dispatch with client token
- ❌ Loopback bypass with `AUTH_OPTIONAL=true`

### 6.3 Security Tests (SEC-AUTH-6003)

**Timing Attack Resistance**:
```rust
#[test]
fn timing_safe_eq_constant_time() {
    let token = "a".repeat(64);
    let wrong_early = "b".repeat(64); // Differs at position 0
    let wrong_late = format!("{}b", "a".repeat(63)); // Differs at position 63
    
    let time_early = measure_comparison(&token, &wrong_early);
    let time_late = measure_comparison(&token, &wrong_late);
    
    // Variance must be < 10%
    let variance = (time_early - time_late).abs() / time_early;
    assert!(variance < 0.1, "timing leak detected: {}% variance", variance * 100.0);
}
```

**Token Leakage Detection**:
```rust
#[test]
fn logs_never_contain_raw_tokens() {
    let logs = capture_logs(|| {
        authenticate_request("secret-token-12345");
    });
    
    assert!(!logs.contains("secret-token-12345"), "token leaked in logs");
    assert!(logs.contains("token:"), "missing identity breadcrumb");
}
```

---

## 7. Migration Plan

### 7.1 Immediate Actions (P0 - Security Critical)

1. **Fix orchestratord/api/nodes.rs** (CURRENT VULNERABILITY)
   - Replace manual validation with `auth_min::timing_safe_eq()`
   - Add `auth_min::token_fp6()` logging
   - Add timing attack regression test

2. **Implement pool-managerd authentication** (MISSING)
   - Create `api/auth.rs` middleware
   - Add Bearer token validation to all endpoints except `/health`
   - Wire `LLORCH_API_TOKEN` configuration

3. **Add HTTP client authentication** (INCOMPLETE)
   - Update `orchestratord` → `pool-managerd` client to send Bearer token
   - Add token configuration to `PoolManagerClient`

### 7.2 Testing & Validation (P0)

4. **Add security test suite**
   - Timing attack tests for all comparison paths
   - Token leakage tests for logs/errors
   - Integration tests with valid/invalid tokens

5. **BDD coverage**
   - Update existing features to include auth headers
   - Add auth-specific scenarios per `.specs/11_min_auth_hooks.md`

### 7.3 Documentation (P1)

6. **Security documentation**
   - Token generation guide
   - Deployment security checklist
   - Incident response runbook

7. **Update deployment guides**
   - Add token configuration to Docker Compose
   - Add token configuration to Kubernetes manifests
   - Add token configuration to bare metal setup

---

## 8. Security Checklist for Code Review

### Pre-Merge Checklist

- [ ] All token comparisons use `auth_min::timing_safe_eq()`
- [ ] All Bearer parsing uses `auth_min::parse_bearer()`
- [ ] All auth logs use `auth_min::token_fp6()` for identity
- [ ] No raw tokens in logs, errors, or traces
- [ ] Bind policy enforced at startup
- [ ] Tests cover timing attack resistance
- [ ] Tests cover token leakage scenarios
- [ ] Integration tests pass with authentication enabled
- [ ] Documentation updated with token configuration

### Runtime Security Checklist

- [ ] `LLORCH_API_TOKEN` configured for all services
- [ ] Token length ≥ 32 characters
- [ ] Token stored securely (not in git, configs, or logs)
- [ ] Non-loopback binds refuse to start without token
- [ ] Logs show identity breadcrumbs for all requests
- [ ] No auth failures from legitimate services
- [ ] Metrics endpoints exempt from auth

---

## 9. Success Criteria

**Phase 5 Authentication Complete When**:

1. ✅ `auth-min` library has 100% test coverage
2. ✅ All orchestratord endpoints use `auth-min` utilities
3. ✅ All pool-managerd endpoints have Bearer token validation
4. ✅ All HTTP clients send Bearer tokens
5. ✅ Timing attack tests pass with < 10% variance
6. ✅ Token leakage tests pass (grep logs for token = FAIL)
7. ✅ E2E tests pass with authentication enabled
8. ✅ Security documentation complete

---

## 10. References

- `.specs/11_min_auth_hooks.md` - Minimal auth hooks specification
- `.specs/00_llama-orch.md` §2.7 - Security & policy requirements
- `libs/auth-min/src/lib.rs` - Implementation reference
- `bin/orchestratord/src/app/auth_min.rs` - Correct usage example
- `OWASP Testing Guide` - Timing attack testing methodology
- `CWE-208` - Observable Timing Discrepancy

---

**Status**: DRAFT - Needs review and approval before implementation.

**Next Actions**:
1. Review with security-focused maintainer
2. Audit all token comparison sites in codebase
3. Implement pool-managerd authentication (P0)
4. Add comprehensive test coverage
5. Update cloud profile migration plan with security gates
