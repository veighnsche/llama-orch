# Cloud Profile Migration - Current Status

**Last Updated**: 2025-09-30 23:15  
**Status**: 🔴 **PAUSED - SECURITY GATE**

---

## Quick Status

```
Phase 1: Foundation Libraries     ✅ COMPLETE
Phase 2: orchestratord Integration ✅ COMPLETE  
Phase 3: pool-managerd Integration ✅ COMPLETE
Phase 4: Multi-Node Placement     ✅ COMPLETE
Phase 5: Authentication           🔴 INCOMPLETE (35% with critical flaws)
Phase 6: Catalog Distribution     ⏸️ BLOCKED
Phase 7: Observability            ⏸️ BLOCKED
Phase 8: Testing & Validation     ⏸️ BLOCKED
Phase 9: Documentation            ⏸️ BLOCKED
```

**Progress**: 4.5 / 9 phases complete (50%)  
**Blocked By**: Phase 5 security vulnerabilities

---

## Why We're Paused

Security audit of Phase 5 (Authentication) identified **2 critical vulnerabilities**:

### 🔴 CRITICAL 1: Timing Attack Vulnerability

**Location**: `bin/orchestratord/src/api/nodes.rs:34`  
**Issue**: Non-constant-time token comparison  
**Impact**: Attackers can discover token character-by-character  
**CVSS**: 7.5 (HIGH)

```rust
// VULNERABLE CODE
token == expected_token  // Leaks timing information!
```

### 🔴 CRITICAL 2: No Authentication on pool-managerd

**Location**: `bin/pool-managerd/src/api/routes.rs`  
**Issue**: All endpoints completely unprotected  
**Impact**: Anyone can control GPU workers  
**CVSS**: 9.1 (CRITICAL)

```rust
// NO AUTH AT ALL
Router::new()
    .route("/pools/:id/preload", post(preload_pool))  // Open to anyone!
    .route("/pools/:id/status", get(pool_status))      // Open to anyone!
```

---

## What Needs to Happen

### P0 Critical Fixes (1 week)

1. ☐ Fix timing attack in orchestratord (2h)
2. ☐ Add timing attack test (2h)
3. ☐ Implement pool-managerd auth (3h)
4. ☐ Test pool-managerd auth (1h)
5. ☐ Add client Bearer tokens (1h)
6. ☐ E2E test with auth (3h)
7. ☐ Code review (4h)
8. ☐ Deploy & monitor (2h)

**Total**: 18 hours (2-3 days with team)

### Security Gate Criteria

- [ ] All token comparisons use `auth_min::timing_safe_eq()`
- [ ] All Bearer parsing uses `auth_min::parse_bearer()`
- [ ] All auth logs use `auth_min::token_fp6()`
- [ ] pool-managerd has Bearer token validation
- [ ] Timing attack tests pass
- [ ] Token leakage tests pass
- [ ] E2E tests with auth pass
- [ ] Security team sign-off

---

## When Can We Resume?

**Resume ETA**: ~2025-10-07 (1 week from now)

**Criteria**:
1. All P0 security fixes merged
2. Security tests passing
3. Security team sign-off received

**Then**: Continue to Phase 6 (Catalog Distribution)

---

## What to Work On Instead

### Option 1: Fix Phase 5 Security (RECOMMENDED)

**Who**: Backend + Security teams  
**What**: Implement P0 fixes per `.docs/PHASE5_FIX_CHECKLIST.md`  
**Why**: Unblocks migration

### Option 2: HOME_PROFILE Improvements (PARALLEL)

**Who**: Anyone not on security fixes  
**What**: Features that don't require cloud profile:
- Model provisioner enhancements
- Session management
- Catalog improvements
- Determinism suite
- BDD test coverage
- Documentation

**Why**: Continues progress while security team fixes Phase 5

---

## Key Documents

**Security Audit**:
- `.docs/CLOUD_MIGRATION_PAUSED.md` - Why we paused (executive summary)
- `.docs/PHASE5_FIX_CHECKLIST.md` - Step-by-step fixes (implementation guide)
- `.specs/12_auth-min-hardening.md` - Security spec (requirements)
- `.docs/SECURITY_AUDIT_AUTH_MIN.md` - Full audit (all touchpoints)
- `.docs/PHASE5_SECURITY_FINDINGS.md` - Detailed findings (analysis)

**Migration Docs** (Updated):
- `TODO_CLOUD_PROFILE.md` - Phase 5 marked BLOCKED
- `CLOUD_PROFILE_MIGRATION_PLAN.md` - Timeline revised

---

## Questions?

**Q**: Can we skip Phase 5 and come back later?  
**A**: **NO**. Authentication is foundational for cloud profile. The vulnerabilities are in cloud-specific code.

**Q**: Can we deploy with a different auth approach?  
**A**: **NO**. The `auth-min` library already exists and provides the correct implementation. Use it.

**Q**: How long will this delay the migration?  
**A**: ~1 week. Original estimate was 5-6 weeks; revised to 6-7 weeks.

**Q**: Is HOME_PROFILE affected?  
**A**: **NO**. HOME_PROFILE doesn't use these endpoints. It's safe to continue HOME_PROFILE work.

**Q**: Who approves resuming the migration?  
**A**: Security team must sign off after P0 fixes are complete and tested.

---

**Status**: 🔴 **PAUSED**  
**Next Review**: After P0 security fixes complete  
**Owner**: Security + Backend teams
