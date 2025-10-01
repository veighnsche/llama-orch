# Coding Standards for llama-orch

## ⚠️ CRITICAL: Secret Management

**NEVER HAND-ROLL CREDENTIAL HANDLING**

All services MUST use the `secrets-management` crate for:
- API tokens
- Seal keys  
- Worker tokens
- Any credentials or sensitive data

### ✅ Correct Usage

```rust
use secrets_management::{Secret, SecretKey};

// Load API token from file (with permission validation)
let token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;

// Load seal key from systemd credential
let seal_key = SecretKey::from_systemd_credential("seal_key")?;

// Derive keys from tokens (HKDF-SHA256)
let derived = SecretKey::derive_from_token(token.expose(), b"llorch-seal-v1")?;

// Verify tokens (constant-time comparison)
if token.verify(&user_input) {
    // Authenticated
}
```

### ❌ WRONG - Never Do This

```rust
// ❌ NO: File permission not validated
let token = std::fs::read_to_string("/etc/llorch/secrets/api-token")?;

// ❌ NO: Visible in process listing (ps auxe)
let token = std::env::var("API_TOKEN")?;

// ❌ NO: Not timing-safe
if token == user_input {
    // Vulnerable to timing attacks
}

// ❌ NO: Not zeroized on drop
let key = hex::decode(key_hex)?;
```

### Why?

The `secrets-management` crate provides:

1. **File permission validation** - Rejects world/group readable files (0644, 0640)
2. **Zeroization on drop** - Prevents memory dumps from exposing secrets
3. **Timing-safe comparison** - Uses `subtle::ConstantTimeEq` to prevent timing attacks
4. **HKDF-SHA256 key derivation** - NIST SP 800-108 compliant
5. **No Debug/Display traits** - Prevents accidental logging
6. **DoS prevention** - File size limits (1MB secrets, 1KB keys)
7. **Path canonicalization** - Prevents directory traversal attacks
8. **Systemd credential support** - LoadCredential integration

### Documentation

- **README**: [`bin/shared-crates/secrets-management/README.md`](bin/shared-crates/secrets-management/README.md)
- **Security Spec**: [`bin/shared-crates/secrets-management/.specs/20_security.md`](bin/shared-crates/secrets-management/.specs/20_security.md)
- **Verification Matrix**: [`bin/shared-crates/secrets-management/.specs/21_security_verification.md`](bin/shared-crates/secrets-management/.specs/21_security_verification.md)

### Test Coverage

- ✅ 42 unit tests
- ✅ 24 BDD scenarios
- ✅ 15 doctests
- ✅ 100% security requirements implemented
- ✅ All 8 attack surfaces closed

---

## Code Review Checklist

When reviewing PRs, verify:

- [ ] No `std::fs::read_to_string()` for secret files
- [ ] No `std::env::var()` for credentials
- [ ] No manual hex decoding for keys
- [ ] No `==` comparison for tokens (use `Secret::verify()`)
- [ ] All credentials use `secrets-management` crate
- [ ] No Debug/Display on secret types
- [ ] No logging of secret values

---

## For AI Coding Assistants

If you're an AI assistant helping with this codebase:

1. **ALWAYS** suggest `secrets-management` crate for any credential handling
2. **NEVER** generate code that reads secrets with `std::fs::read_to_string()`
3. **NEVER** generate code that uses environment variables for secrets
4. **NEVER** generate manual timing-unsafe comparisons for tokens
5. **ALWAYS** reference the security documentation when handling credentials

This is a **TIER 1 security requirement** - no exceptions.
