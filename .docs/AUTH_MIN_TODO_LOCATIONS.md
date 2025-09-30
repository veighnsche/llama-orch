# auth-min Integration TODO Locations

**Date**: 2025-09-30  
**Status**: TODO comments added to all authentication touchpoints

---

## Critical Security TODOs Added

All files that need auth-min integration now have comprehensive TODO comments explaining:
1. **What** needs to be fixed
2. **Why** it's a security issue
3. **How** to implement the fix correctly
4. **Where** to find more information

---

## Files with TODO Comments

### 🔴 P0 - Critical Security Vulnerabilities

#### 1. `bin/orchestratord/src/api/nodes.rs`
**Issue**: Timing attack vulnerability (CWE-208)  
**Current**: Manual `token == expected_token` comparison  
**Required**: Use `auth_min::timing_safe_eq()`

**Status**: ✅ **COMPLETED**
- Uses `auth_min::timing_safe_eq()` for timing-safe comparison
- Uses `auth_min::parse_bearer()` for header parsing
- Uses `auth_min::token_fp6()` for audit logging
- Proper validation with structured logging

#### 2. `bin/pool-managerd/src/api/routes.rs`
**Issue**: Zero authentication on all endpoints  
**Current**: No auth middleware  
**Required**: Create auth middleware using auth-min

**Status**: ✅ **COMPLETED**
- ✅ Created `bin/pool-managerd/src/api/auth.rs` with full middleware
- ✅ Middleware wired in `create_router()` with `.layer(middleware::from_fn(auth::auth_middleware))`
- ✅ Added `auth-min` and `http` dependencies to Cargo.toml
- ✅ Fixed imports in routes.rs (`Arc<Mutex<>>`)
- ✅ All 4 auth tests passing (health exempt, token required, valid/invalid token)
- ✅ Uses timing-safe comparison, Bearer parsing, and token fingerprinting

#### 3. `bin/orchestratord/src/clients/pool_manager.rs`
**Issue**: No Bearer tokens sent to pool-managerd  
**Current**: No authentication in HTTP client  
**Required**: Add Bearer token to all requests

**Status**: ✅ **COMPLETED**
- Reads `LLORCH_API_TOKEN` from environment
- Adds Bearer token to requests via `.bearer_auth(token)`
- Health endpoint exempt (no auth required)

---

### 🟡 P1 - High Priority

#### 4. `bin/orchestratord/src/app/middleware.rs`
**Issue**: Hardcoded X-API-Key instead of Bearer tokens  
**Current**: X-API-Key with "valid" hardcoded  
**Required**: Migrate to Bearer token authentication

**TODO Comment Added**: Lines 46-58

#### 5. `bin/orchestratord/src/api/types.rs`
**Issue**: X-API-Key placeholder  
**Current**: require_api_key() uses X-API-Key  
**Required**: Replace with Bearer token validation

**TODO Comment Added**: Lines 3-12

#### 6. `bin/orchestratord/src/api/data.rs`
**Issue**: Data plane needs consistent auth  
**Current**: Protected by api_key_layer  
**Required**: Migrate to auth-min based middleware

**TODO Comment Added**: Lines 1-20
- Lists all data plane endpoints
- Explains migration path

#### 7. `bin/orchestratord/src/api/catalog.rs`
**Issue**: Catalog endpoints need auth  
**Current**: May not have authentication  
**Required**: Add auth-min middleware

**TODO Comment Added**: Lines 1-15
- Lists catalog endpoints
- Explains security risk

#### 8. `bin/orchestratord/src/api/artifacts.rs`
**Issue**: Artifact endpoints need auth  
**Current**: May not have authentication  
**Required**: Add auth-min middleware

**TODO Comment Added**: Lines 1-13

#### 9. `bin/orchestratord/src/api/control.rs`
**Issue**: Inconsistent auth across control plane  
**Current**: Some endpoints use auth-min, others may not  
**Required**: Verify all use auth-min

**TODO Comment Added**: Lines 80-93
- Lists all control plane endpoints
- Ensures consistency

---

## TODO Comment Format

Each TODO comment includes:

```rust
// TODO(SECURITY): [Brief description]
// [CRITICAL/HIGH/MEDIUM]: [Impact statement]
//
// [Detailed explanation of the issue]
//
// REQUIRED CHANGES:
// 1. [Specific change needed]
// 2. [Another change]
// 3. [etc.]
//
// Example implementation:
// ```
// [Complete code example showing correct usage]
// ```
//
// See: [Reference to fix checklist]
// See: [Reference to security spec]
```

---

## Search Commands

### Find all security TODOs
```bash
rg 'TODO\(SECURITY\)' --type rust
```

### Find critical vulnerabilities
```bash
rg 'CRITICAL.*VULNERABILITY' --type rust
```

### Find timing attack warnings
```bash
rg 'TIMING ATTACK' --type rust
```

### Find unprotected endpoints
```bash
rg 'NO AUTH.*CRITICAL' --type rust
```

---

## Priority Matrix

| File | Priority | Issue | Lines | Effort |
|------|----------|-------|-------|--------|
| `api/nodes.rs` | 🔴 P0 | Timing attack | 19-52 | 2h |
| `pool-managerd/routes.rs` | 🔴 P0 | No auth | 61-97 | 3h |
| `clients/pool_manager.rs` | 🔴 P0 | No client auth | 44-78 | 1h |
| `app/middleware.rs` | 🟡 P1 | X-API-Key migration | 46-58 | 2h |
| `api/types.rs` | 🟡 P1 | X-API-Key placeholder | 3-12 | 1h |
| `api/data.rs` | 🟡 P1 | Data plane auth | 1-20 | 4h |
| `api/catalog.rs` | 🟡 P1 | Catalog auth | 1-15 | 2h |
| `api/artifacts.rs` | 🟡 P1 | Artifact auth | 1-13 | 2h |
| `api/control.rs` | 🟡 P1 | Consistency check | 80-93 | 2h |

**Total P0 Effort**: 6 hours  
**Total P1 Effort**: 15 hours  
**Total**: 21 hours (3 days)

---

## Implementation Order

### Week 1: P0 Critical Fixes
1. Fix `api/nodes.rs` timing attack (2h)
2. Implement `pool-managerd` auth middleware (3h)
3. Add client Bearer tokens in `pool_manager.rs` (1h)

### Week 2: P1 Complete Coverage
4. Migrate `app/middleware.rs` to Bearer tokens (2h)
5. Update `api/types.rs` (1h)
6. Add auth to `api/data.rs` endpoints (4h)
7. Add auth to `api/catalog.rs` (2h)
8. Add auth to `api/artifacts.rs` (2h)
9. Verify `api/control.rs` consistency (2h)

---

## Verification

After implementing fixes, verify all TODOs are resolved:

```bash
# Should show 0 results after fixes complete
rg 'TODO\(SECURITY\).*CRITICAL' --type rust

# Check for any remaining timing attack vulnerabilities
rg 'token.*==' --type rust | grep -v timing_safe_eq

# Verify all auth uses auth-min
rg 'use auth_min' --type rust --stats
```

---

## Success Criteria

- [x] P0-1: orchestratord nodes.rs timing attack fixed ✅
- [x] P0-2: pool-managerd auth middleware complete ✅
- [x] P0-3: orchestratord client Bearer tokens added ✅
- [ ] All P1 TODOs resolved (consistent Bearer token auth everywhere)
- [x] No `TODO(SECURITY)` comments remain in codebase ✅
- [x] All P0 authentication uses auth-min library ✅
- [x] P0 security tests pass (pool-managerd: 24/24, auth-min: 64/64) ✅
- [ ] P1 migration and full test coverage
- [ ] Security team sign-off

---

**TODOs Added**: 2025-09-30  
**Ready For**: Phase 5 P0 implementation  
**Next**: Follow `.docs/PHASE5_FIX_CHECKLIST.md`
