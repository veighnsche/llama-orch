# auth-min

**Minimal authentication with maximum security**

`libs/auth-min` — Timing-safe token comparison, secure fingerprinting, and Bearer token parsing for llama-orch services.

---

## What This Library Does

auth-min provides **security-hardened authentication** for llama-orch:

- **Timing-safe comparison** — Prevents timing attacks (CWE-208)
- **Token fingerprinting** — SHA-256 based, non-reversible
- **Bearer token parsing** — Robust RFC 6750 parsing
- **Bind policy** — Refuses non-loopback without token
- **Proxy trust gate** — Optional proxy auth trust (dangerous!)

**Used by**: orchestratord, pool-managerd, node-registration, http-util

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

**Security**: Execution time independent of mismatch position (prevents CWE-208)

### Token Fingerprinting

```rust
use auth_min::token_fp6;

let token = "secret-abc123";
let fingerprint = token_fp6(token);

println!("Token fingerprint: {}", fingerprint); // "a3f2c1"
```

**Security**: SHA-256 based, non-reversible, safe for logs

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

**Handles**: Whitespace, case-insensitive "Bearer", validation

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
| `LLORCH_API_TOKEN` | Conditional | None | Authentication token (required for non-loopback) |
| `TRUST_PROXY_AUTH` | No | `false` | Trust proxy-injected auth headers (⚠️ dangerous!) |

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

## Security Properties

### Timing-Safe Comparison

- **Property**: Execution time independent of mismatch position
- **Test**: Variance < 10% for early vs. late mismatches
- **Implementation**: Bitwise OR accumulation examines all bytes
- **Prevents**: CWE-208 (Observable Timing Discrepancy)

### Token Fingerprinting

- **Property**: Non-reversible, collision-resistant
- **Algorithm**: SHA-256 → first 6 hex chars (24-bit space)
- **Use Case**: Safe for audit logs, correlation, debugging
- **Cannot**: Recover original token from fingerprint

### Bind Policy

- **Property**: Refuses non-loopback bind without token
- **Enforcement**: Startup validation (fail-fast)
- **Override**: Not allowed (security by design)
- **Loopback**: No token required for 127.0.0.1 or ::1

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

## What This Library Is Not

- ❌ Not a full authentication framework (no users, roles, sessions)
- ❌ Not OAuth2/OIDC/SSO (minimal shared-secret only)
- ❌ Not mTLS (future: v0.3.0)
- ❌ Not rate limiting (handled at admission layer)

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
