# ⚠️  CRITICAL SECURITY VULNERABILITY - AUTH BYPASS ⚠️

**Status:** 🔴 **UNFIXED - DO NOT DEPLOY TO NETWORK**  
**Severity:** **CRITICAL**  
**Discovered:** 2025-10-19  
**Affected:** All worker binaries (main, cpu, cuda, metal)

---

## 🚨 THE PROBLEM

**Authentication can be completely bypassed by not providing a token!**

### Current Broken Behavior

```bash
# NO AUTH - Anyone can use the worker!
LLORCH_API_TOKEN="" ./llm-worker-rbee --port 8080 ...

# AUTH REQUIRED
LLORCH_API_TOKEN="secret" ./llm-worker-rbee --port 8080 ...
```

**The vulnerability:** An attacker can simply unset the environment variable to disable authentication entirely!

---

## 💥 ATTACK SCENARIO

1. Worker deployed on network with `LLORCH_API_TOKEN` set
2. Attacker gains access to restart the worker process
3. Attacker restarts worker WITHOUT setting `LLORCH_API_TOKEN`
4. **Worker now accepts requests from ANYONE without authentication!**

OR:

1. Developer forgets to set `LLORCH_API_TOKEN` in production
2. Worker starts with NO authentication
3. **Anyone on the network can use the worker for free!**

---

## 🔍 ROOT CAUSE

**File:** `src/http/routes.rs` (lines 61-97)  
**Introduced by:** TEAM-102  
**Modified by:** Current fix attempt (made it worse!)

```rust
// BROKEN CODE:
let protected_routes = if expected_token.is_empty() {
    // NO AUTH - Just don't send a token!
    worker_routes
} else {
    // AUTH REQUIRED
    worker_routes.layer(auth_middleware)
};
```

**Why this is broken:**
- No way to distinguish "local dev mode" from "production without token"
- Absence of token = no auth (should be an ERROR in production!)
- Worker binds to `0.0.0.0` (all interfaces) even in "local mode"

---

## ✅ PROPER FIX REQUIRED

### 1. Add Explicit Mode Flag

```rust
#[derive(Parser)]
struct Args {
    // ... existing args ...
    
    /// Run in local development mode (no auth, 127.0.0.1 only)
    #[arg(long, default_value = "false")]
    local_mode: bool,
}
```

### 2. Enforce Security Rules

```rust
if args.local_mode {
    // LOCAL MODE: No auth, but ONLY bind to localhost
    if expected_token.is_empty() {
        tracing::warn!("Local mode: No authentication");
        let addr = SocketAddr::from(([127, 0, 0, 1], args.port)); // localhost only!
        // ... no auth middleware
    } else {
        anyhow::bail!("Local mode cannot use auth token - remove LLORCH_API_TOKEN");
    }
} else {
    // NETWORK MODE: Auth is MANDATORY
    if expected_token.is_empty() {
        anyhow::bail!("Network mode requires LLORCH_API_TOKEN - set it or use --local-mode");
    }
    tracing::info!("Network mode: Authentication enabled");
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port)); // all interfaces
    // ... apply auth middleware
}
```

### 3. Security Matrix

| Mode | Token | Bind Address | Auth | Result |
|------|-------|--------------|------|--------|
| `--local-mode` | Empty | 127.0.0.1 | ❌ No | ✅ OK (dev only) |
| `--local-mode` | Set | - | - | ❌ ERROR (conflicting) |
| Network (default) | Empty | - | - | ❌ ERROR (insecure) |
| Network (default) | Set | 0.0.0.0 | ✅ Yes | ✅ OK (secure) |

---

## 📋 IMPLEMENTATION CHECKLIST

- [ ] Add `--local-mode` flag to all binaries (main.rs, cpu.rs, cuda.rs, metal.rs)
- [ ] Make network mode FAIL if token is empty
- [ ] Make local mode bind to 127.0.0.1 ONLY
- [ ] Make local mode FAIL if token is provided (conflicting config)
- [ ] Update routes.rs to use explicit mode instead of token presence
- [ ] Add integration tests for all security scenarios
- [ ] Update documentation
- [ ] Add deployment checklist warning about auth

---

## 🚫 TEMPORARY MITIGATION

**Until this is fixed, follow these rules:**

1. **ALWAYS set `LLORCH_API_TOKEN` in production**
2. **NEVER expose worker on network without token**
3. **Use firewall rules to restrict access**
4. **Monitor for unauthenticated requests**
5. **Bind to 127.0.0.1 for local development:**
   ```bash
   # Edit main.rs line 202:
   let addr = SocketAddr::from(([127, 0, 0, 1], args.port)); // localhost only
   ```

---

## 📍 AFFECTED FILES

**Need immediate fixes:**
- `src/main.rs` (lines 206-232)
- `src/bin/cpu.rs` (lines 94-99)
- `src/bin/cuda.rs` (lines 108-113)
- `src/bin/metal.rs` (lines 106-111)
- `src/http/routes.rs` (lines 61-97)

**All contain the same vulnerability!**

---

## 🔗 REFERENCES

- **CWE-306:** Missing Authentication for Critical Function
- **OWASP:** Broken Authentication
- **Severity:** CRITICAL (CVSS 9.8 - Network exploitable, no auth required)

---

## ⚠️  DO NOT DEPLOY WITHOUT FIXING THIS! ⚠️

**This is a CRITICAL security vulnerability that allows complete authentication bypass.**

**Status:** 🔴 **UNFIXED**  
**Action Required:** Implement proper fix before ANY network deployment  
**Assigned To:** TODO - TEAM-XXX

---

**Last Updated:** 2025-10-19  
**Discovered By:** Security review during worker isolation testing
