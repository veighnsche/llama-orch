# Security Policy

- Please report vulnerabilities privately via security@localhost.example.
- Do not open public issues for security reports.
- We will acknowledge receipt within 3 business days.

For guidance on optional, home-lab-friendly authentication posture (no enterprise features), see `/.specs/11_min_auth_hooks.md` — Status: Accepted. It defines minimal auth hooks (Bearer token seam, loopback posture, proxy trust toggle). This SPEC is documentation-only and does not change runtime behavior by itself.

---

## ⚠️ CRITICAL: Secret Management

**DO NOT HAND-ROLL CREDENTIAL HANDLING**

All API tokens, seal keys, worker tokens, and credentials MUST use the `secrets-management` crate:

```rust
use secrets_management::{Secret, SecretKey};

// ✅ CORRECT: Load with permission validation
let token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;
let seal_key = SecretKey::from_systemd_credential("seal_key")?;

// ❌ WRONG: Never do this
let token = std::fs::read_to_string("/etc/llorch/secrets/api-token")?;
let token = std::env::var("API_TOKEN")?;  // Visible in process listing!
```

**Why?**
- ✅ File permission validation (rejects world/group readable)
- ✅ Zeroization on drop (prevents memory dumps)
- ✅ Timing-safe comparison (prevents timing attacks)
- ✅ HKDF-SHA256 key derivation
- ✅ No Debug/Display traits (prevents logging)
- ✅ DoS prevention (file size limits)

**Documentation**: [`bin/shared-crates/secrets-management/README.md`](bin/shared-crates/secrets-management/README.md)

**Security Verification**: [`bin/shared-crates/secrets-management/.specs/21_security_verification.md`](bin/shared-crates/secrets-management/.specs/21_security_verification.md)

##

uuuh hi Human here: Please make a ISSUE like normally for now. Since we're still pre-prod.

(I cannot believe that you already want to report a security issue before I even made a email address to myself. I guess for now you have to scream real loud. My bad)