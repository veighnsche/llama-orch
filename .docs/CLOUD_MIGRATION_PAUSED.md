# Cloud Profile Migration - PAUSED ⏸️

**Date**: 2025-09-30  
**Status**: 🔴 **MIGRATION PAUSED - SECURITY GATE**  
**Reason**: Critical security vulnerabilities in Phase 5 (Authentication)  
**Resume ETA**: ~1 week after P0 fixes complete

---

## Why We're Pausing

Cloud profile migration **MUST STOP** at Phase 5 due to **2 critical security vulnerabilities**:

### 🔴 CRITICAL 1: Timing Attack in Node Registration

**Location**: `bin/orchestratord/src/api/nodes.rs:34`

```rust
// Current code - VULNERABLE
token == expected_token  // Non-constant-time comparison
```

**Impact**: 
- Attackers can discover `LLORCH_API_TOKEN` character-by-character
- Affects node registration, heartbeat, deregistration
- **Cloud profile specific** - HOME_PROFILE doesn't use these endpoints

**CVSS**: 7.5 (HIGH)

---

### 🔴 CRITICAL 2: Zero Authentication on pool-managerd

**Location**: `bin/pool-managerd/src/api/routes.rs`

```rust
// Current code - NO AUTH AT ALL
Router::new()
    .route("/pools/:id/preload", post(preload_pool))  // Anyone can spawn engines!
    .route("/pools/:id/status", get(pool_status))      // Anyone can query GPUs!
```

**Impact**:
- Anyone on network can spawn engines (resource exhaustion, crypto mining)
- Anyone can query GPU status (reconnaissance)
- Anyone can control your GPU workers
- **Cloud profile specific** - pool-managerd only runs in cloud profile

**CVSS**: 9.1 (CRITICAL)

---

## What This Means

### Cannot Continue To:

- ❌ **Phase 6**: Catalog Distribution (requires secure pool-managerd)
- ❌ **Phase 7**: Observability (need secure foundations)
- ❌ **Phase 8**: Testing & Validation (would test insecure code)
- ❌ **Phase 9**: Documentation (would document vulnerable setup)

### Can Continue:

- ✅ **HOME_PROFILE** work (not affected - uses filesystem, no network auth)
- ✅ **HOME_PROFILE** improvements (models, catalog, sessions, etc.)
- ✅ **Security hardening** (fixing Phase 5 properly)

---

## Security Gate Requirements

Phase 5 can **ONLY** be marked complete when:

### P0 Critical Fixes (1 week)

- [ ] orchestratord timing attack fixed (use `auth_min::timing_safe_eq()`)
- [ ] pool-managerd authentication implemented (Bearer token validation)
- [ ] HTTP clients send Bearer tokens (orchestratord → pool-managerd)
- [ ] Timing attack regression tests pass
- [ ] Integration tests with auth pass
- [ ] Security team sign-off

### Verification

```bash
# 1. No timing-vulnerable comparisons
rg 'token.*==' --type rust | grep -v timing_safe_eq
# MUST be EMPTY

# 2. All pool-managerd routes have auth
rg 'route\(' bin/pool-managerd/src/api/routes.rs
# MUST show auth middleware

# 3. All tests pass
cargo xtask dev:loop
# MUST be green

# 4. Security tests pass
cargo test -p orchestratord security_timing
cargo test -p pool-managerd auth_integration
# MUST pass
```

---

## Timeline

### Week 1: P0 Security Fixes

| Day | Task | Hours | Owner |
|-----|------|-------|-------|
| Mon | Fix timing attack + test | 4 | Backend |
| Tue | Implement pool-managerd auth | 4 | Backend |
| Wed | Add client Bearer tokens + E2E test | 4 | Backend |
| Thu | Code review + fixes | 4 | Security |
| Fri | Deploy + monitor | 2 | DevOps |

**Total**: 18 hours (2-3 days with team)

### Week 2: Resume Migration

- ✅ Security gate passed
- ✅ Phase 5 complete (properly)
- ✅ Continue to Phase 6 (Catalog Distribution)

---

## What to Work On Instead

### Priority 1: Fix Phase 5 Security (P0)

**Checklist**: `.docs/PHASE5_FIX_CHECKLIST.md`

Tasks:
1. Fix timing attack in orchestratord/api/nodes.rs
2. Add timing attack regression test
3. Implement pool-managerd authentication middleware
4. Test pool-managerd auth
5. Add Bearer tokens to HTTP clients
6. E2E test with authentication
7. Code review
8. Security sign-off

### Priority 2: HOME_PROFILE Improvements (Parallel)

While security fixes are in code review, you can work on:

- ✅ Model provisioner enhancements
- ✅ Session management improvements
- ✅ Catalog enhancements
- ✅ Determinism suite expansions
- ✅ BDD test coverage
- ✅ Documentation improvements

These don't require cloud profile and don't touch security-sensitive code.

---

## Impact Assessment

### If We Don't Fix First

Deploying cloud profile with these vulnerabilities would result in:

1. **Token Compromise** (HIGH likelihood)
   - Attacker discovers `LLORCH_API_TOKEN` via timing attack
   - Full control over orchestrator

2. **GPU Hijacking** (HIGH likelihood)
   - Attacker spawns engines on your GPUs
   - Resource theft, crypto mining, malicious inference

3. **Data Breach** (MEDIUM likelihood)
   - Unauthorized task submission
   - Model theft, prompt/response exfiltration

4. **Compliance Failure** (CERTAIN)
   - Cannot pass security audit
   - Cannot deploy to production
   - Cannot certify for enterprise use

5. **Reputational Damage** (HIGH)
   - "Open-source LLM orchestrator with timing attacks"
   - Loss of community trust

### With Fixes

- ✅ Production-ready security
- ✅ Industry best practices
- ✅ Audit trail for compliance
- ✅ Safe to deploy to cloud

---

## Communication Plan

### Internal Team

**Message**: "Cloud migration paused for 1 week to fix critical security vulnerabilities discovered in Phase 5. We're fixing timing attack and implementing proper authentication. Resume next week with secure foundations."

### External (if applicable)

**DO NOT** mention:
- Specific vulnerability details (timing attack, CWE-208)
- Affected code locations
- Token extraction method

**DO** say:
- "Conducting security hardening of authentication layer"
- "Following security best practices before cloud deployment"
- "ETA: 1 week, then resume migration"

---

## Decision Authority

**Decision**: Pause cloud migration, fix security first  
**Made by**: Security & Architecture review  
**Date**: 2025-09-30  
**Approved by**: _________________  

**Rationale**: 
1. Critical vulnerabilities in cloud-specific code
2. Cannot deploy insecure cloud profile
3. Fixes are quick (1 week P0)
4. Better to pause now than rollback later

---

## Resume Criteria

Cloud migration can **ONLY** resume when:

- [ ] All P0 security fixes merged
- [ ] Security tests passing (timing attack, auth integration)
- [ ] E2E tests passing with authentication enabled
- [ ] Security team sign-off received
- [ ] Documentation updated with security configuration
- [ ] Phase 5 marked complete (properly)

**Sign-off Required From**:
- [ ] Security Team Lead
- [ ] Backend Team Lead  
- [ ] Architecture Team

**Expected Resume Date**: ~2025-10-07 (1 week)

---

## References

**Security Audit Documents**:
- `.specs/12_auth-min-hardening.md` - Security specification
- `.docs/SECURITY_AUDIT_AUTH_MIN.md` - Complete vulnerability audit
- `.docs/PHASE5_SECURITY_FINDINGS.md` - Detailed findings & fixes
- `.docs/PHASE5_AUDIT_SUMMARY.md` - Executive summary
- `.docs/PHASE5_FIX_CHECKLIST.md` - Implementation checklist

**Cloud Migration Documents** (UPDATED):
- `TODO_CLOUD_PROFILE.md` - Phase 5 marked BLOCKED
- `CLOUD_PROFILE_MIGRATION_PLAN.md` - Security gate added
- `.docs/PHASE5_COMPLETE.md` - Status: INCOMPLETE

---

**Status**: 🔴 **PAUSED**  
**Next Action**: Implement P0 security fixes per `.docs/PHASE5_FIX_CHECKLIST.md`  
**Resume**: After security gate passed (~1 week)
