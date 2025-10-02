# auth-min

**Minimal authentication primitives for llama-orch**

Simple, focused security utilities: timing-safe token comparison, secure fingerprinting, and Bearer token parsing.

---

## What This Library Does

auth-min provides basic authentication primitives:

- **Timing-safe comparison** — Constant-time token validation
- **Token fingerprinting** — SHA-256 based, safe for logs
- **Bearer token parsing** — RFC 6750 compliant
- **Bind policy** — Loopback detection and startup validation

**Used by**: orchestratord, pool-managerd, http-util, audit-logging

---

## Key APIs

### Timing-Safe Comparison

```rust
use auth_min::timing_safe_eq;

let token = "secret-token";
let expected = "secret-token";

if timing_safe_eq(token.as_bytes(), expected.as_bytes()) {
    println!("Authenticated");
} else {
    println!("Invalid token");
}
```

**Security**: Constant-time execution prevents timing attacks (CWE-208)

### Token Fingerprinting

```rust
use auth_min::token_fp6;

let token = "secret-abc123";
let fingerprint = token_fp6(token);

println!("Token fingerprint: {}", fingerprint); // "a3f2c1"
```

**Use case**: Safe for logs, audit trails, and debugging

### Bearer Token Parsing

```rust
use auth_min::parse_bearer;

let header = Some("Bearer secret-token");
let token = parse_bearer(header);

match token {
    Some(t) => println!("Token: {}", t),
    None => println!("No token"),
}
```

**Compliance**: RFC 6750 (OAuth 2.0 Bearer Token Usage)

### Bind Policy

```rust
use auth_min::{enforce_startup_bind_policy, is_loopback_addr};

// Check if address is loopback
if is_loopback_addr("127.0.0.1:8080") {
    println!("Loopback address");
}

// Enforce bind policy (requires token for non-loopback)
enforce_startup_bind_policy("0.0.0.0:8080")?;
```

---

## Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `LLORCH_API_TOKEN` | Conditional | None | Required for non-loopback binds |

### Token Generation

```bash
# Generate secure token
export LLORCH_API_TOKEN=$(openssl rand -hex 32)

# Loopback development (no token required)
ORCHD_ADDR=127.0.0.1:8080

# Production (token required)
ORCHD_ADDR=0.0.0.0:8080
LLORCH_API_TOKEN=your-secret-token-here
```

---

## Logging Pattern

### Safe Logging

```rust
use auth_min::{timing_safe_eq, token_fp6, parse_bearer};

// Parse and validate
let token = parse_bearer(auth_header)?;
if !timing_safe_eq(token.as_bytes(), expected.as_bytes()) {
    let fp6 = token_fp6(&token);
    tracing::warn!(identity = %format!("token:{}", fp6), "auth failed");
    return Err(AuthError::InvalidToken);
}

// Success
let fp6 = token_fp6(&token);
tracing::info!(identity = %format!("token:{}", fp6), "authenticated");
```

### Log Safety Rules

- ✅ **SAFE**: `token:a3f2c1` (fingerprint)
- ❌ **UNSAFE**: `token:secret-abc123` (raw token)

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test -p auth-min -- --nocapture

# Specific test suites
cargo test -p auth-min compare  # Timing-safe comparison
cargo test -p auth-min timing   # Timing attack resistance
cargo test -p auth-min leakage  # Token leakage detection
```

### Security Tests

```bash
# Timing attack resistance (must pass)
cargo test -p auth-min test_timing_variance -- --nocapture

# Token leakage detection
cargo test -p auth-min test_fingerprint -- --nocapture
```

---

## Implementation Details

### Timing-Safe Comparison

- Constant-time execution (variance < 10%)
- Bitwise OR accumulation
- Prevents CWE-208 timing attacks

### Token Fingerprinting

- SHA-256 hash → first 6 hex chars
- Non-reversible, 24-bit collision space
- Safe for logs and audit trails

### Bind Policy

- Startup validation (fail-fast)
- Loopback detection (127.0.0.1, ::1)
- Requires token for public binds

---

## Security Audit

### Check for Timing Vulnerabilities

```bash
# Verify all comparisons are timing-safe
rg '== .*token|token.*==' libs/auth-min/src/ | grep -v timing_safe_eq
# Should be EMPTY
```

### Check for Token Leakage

```bash
# Verify fingerprints used for logging
rg 'token_fp6' libs/auth-min/src/
# Should show usage in all auth code

# Check for raw token logging
rg 'tracing.*token(?!_fp6|:)' --type rust
# Should be EMPTY
```

### Run Security Test Suite

```bash
cargo test -p auth-min --test timing
cargo test -p auth-min --test leakage
```

---

## Dependencies

### Internal

- None (foundational security library)

### External

- `sha2` — SHA-256 hashing
- `subtle` — Constant-time comparison (optional, for verification)

---

## Security References

- **CWE-208**: Observable Timing Discrepancy
- **RFC 6750**: OAuth 2.0 Bearer Token Usage
- **OWASP**: Timing Attack Testing

---

## Scope

This library provides low-level primitives only:

- ❌ No user management or sessions
- ❌ No OAuth2/OIDC/SSO flows
- ❌ No mTLS support
- ❌ No rate limiting

---

## Specifications

Implements requirements from:
- AUTH-1001..AUTH-1008 (Minimal auth hooks)
- SEC-AUTH-* (Security hardening)

---

## Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Production-ready (security-hardened)
- **Maintainers**: @llama-orch-maintainers
- **Security Review**: Required for all changes
