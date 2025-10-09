# Shared Crates Analysis - Post-Pivot

**Date:** 2025-10-09T23:39:00+02:00  
**Team:** TEAM-027  
**Context:** After pivoting to rbee architecture (queen-rbee, rbee-hive, rbee-keeper, llm-worker-rbee)

---

## Current Shared Crates

### ✅ ACTIVE - Used in Current Architecture

#### 1. **worker-registry** (NEW - TEAM-027)
- **Purpose:** SQLite-backed worker tracking
- **Used by:** queen-rbee, rbee-keeper
- **Status:** ✅ Active, just created
- **Reason:** Shared state between orchestrator daemon and CLI

#### 2. **pool-core**
- **Purpose:** Pool management shared types
- **Used by:** rbee-hive
- **Status:** ✅ Active (TEAM-022)
- **Reason:** Core types for pool management

#### 3. **gpu-info**
- **Purpose:** GPU detection and information
- **Used by:** rbee-hive, llm-worker-rbee
- **Status:** ✅ Active
- **Reason:** Needed for backend detection and device management

---

### ⚠️ UNCERTAIN - Need to Verify Usage

#### 4. **auth-min**
- **Purpose:** Minimal authentication
- **Current Status:** ⚠️ Uncertain
- **Question:** Do we need auth between components?
- **Decision Needed:** 
  - If rbee-hive ↔ llm-worker-rbee needs auth: Keep
  - If queen-rbee ↔ rbee-hive needs auth: Keep
  - Otherwise: Mark obsolete

#### 5. **deadline-propagation**
- **Purpose:** Request deadline propagation
- **Current Status:** ⚠️ Uncertain
- **Question:** Do we propagate deadlines in MVP?
- **Decision Needed:**
  - If queen-rbee needs to propagate timeouts: Keep
  - Otherwise: Defer to post-MVP

#### 6. **secrets-management**
- **Purpose:** API key and secret management
- **Current Status:** ⚠️ Uncertain
- **Question:** How do we manage API keys?
- **Decision Needed:**
  - If we need secure key storage: Keep
  - If MVP uses simple keys: Defer to post-MVP

---

### ❌ LIKELY OBSOLETE - From Old Architecture

#### 7. **pool-registry-types**
- **Purpose:** Pool registry types (old architecture)
- **Current Status:** ❌ Likely obsolete
- **Reason:** Replaced by worker-registry
- **Action:** Verify no usage, then mark for deletion

#### 8. **orchestrator-core**
- **Purpose:** Orchestrator core logic (old architecture)
- **Current Status:** ❌ Likely obsolete
- **Reason:** queen-rbee will have its own implementation
- **Action:** Check if anything useful, then mark for deletion

#### 9. **audit-logging**
- **Purpose:** Audit logging infrastructure
- **Current Status:** ❌ Likely obsolete for MVP
- **Reason:** MVP doesn't require audit logging
- **Action:** Defer to post-MVP, mark as non-MVP

#### 10. **input-validation**
- **Purpose:** Input validation with BDD tests
- **Current Status:** ❌ Likely obsolete for MVP
- **Reason:** MVP uses simple validation
- **Action:** Defer to post-MVP, mark as non-MVP

#### 11. **narration-core** + **narration-macros**
- **Purpose:** Narration/logging framework
- **Current Status:** ❌ Likely obsolete
- **Reason:** Using tracing instead
- **Action:** Mark for deletion, use tracing everywhere

---

## Recommendations

### Immediate Actions (TEAM-027/028)

1. **Keep Active:**
   - worker-registry ✅
   - pool-core ✅
   - gpu-info ✅

2. **Verify and Decide:**
   - auth-min (check if MVP needs auth)
   - deadline-propagation (check if MVP needs deadlines)
   - secrets-management (check if MVP needs secure storage)

3. **Mark for Deletion:**
   - narration-core (replaced by tracing)
   - narration-macros (replaced by tracing)
   - audit-logging (not needed for MVP)
   - input-validation (not needed for MVP)

4. **Investigate:**
   - pool-registry-types (likely replaced by worker-registry)
   - orchestrator-core (check if anything useful)

### Long-Term Strategy

**Phase 1 (MVP):**
- Use only: worker-registry, pool-core, gpu-info
- Defer: auth-min, deadline-propagation, secrets-management
- Ignore: narration-*, audit-logging, input-validation

**Phase 2 (Post-MVP):**
- Re-evaluate deferred crates
- Implement proper auth if needed
- Add audit logging if required
- Add input validation if required

---

## Migration Path

### For Obsolete Crates

1. **Create `.archive/` directory:**
   ```bash
   mkdir -p bin/shared-crates/.archive
   ```

2. **Move obsolete crates:**
   ```bash
   mv bin/shared-crates/narration-core bin/shared-crates/.archive/
   mv bin/shared-crates/narration-macros bin/shared-crates/.archive/
   mv bin/shared-crates/audit-logging bin/shared-crates/.archive/
   mv bin/shared-crates/input-validation bin/shared-crates/.archive/
   ```

3. **Update workspace Cargo.toml:**
   - Remove archived crates from members list
   - Add comment explaining archival

4. **Document in README:**
   - List archived crates
   - Explain why archived
   - Note: Can be restored if needed

---

## Current Architecture Needs

### rbee-hive (Pool Manager)
**Needs:**
- pool-core ✅
- gpu-info ✅
- worker-registry (for listing workers)

**Doesn't Need:**
- narration-* (uses tracing)
- audit-logging (MVP doesn't require)
- input-validation (simple validation)

### queen-rbee (Orchestrator Daemon)
**Needs:**
- worker-registry ✅
- (TBD: auth-min if we add auth)
- (TBD: deadline-propagation if we add timeouts)

**Doesn't Need:**
- narration-* (uses tracing)
- pool-registry-types (uses worker-registry)

### rbee-keeper (Orchestrator CLI)
**Needs:**
- worker-registry ✅

**Doesn't Need:**
- Everything else (simple CLI)

### llm-worker-rbee (Worker Daemon)
**Needs:**
- gpu-info ✅
- (TBD: auth-min if we add auth)

**Doesn't Need:**
- narration-* (uses tracing)
- worker-registry (doesn't track other workers)

---

## Decision Matrix

| Crate | MVP Needed? | Post-MVP? | Action |
|-------|-------------|-----------|--------|
| worker-registry | ✅ Yes | ✅ Yes | Keep |
| pool-core | ✅ Yes | ✅ Yes | Keep |
| gpu-info | ✅ Yes | ✅ Yes | Keep |
| auth-min | ⚠️ TBD | ✅ Likely | Verify |
| deadline-propagation | ❌ No | ⚠️ Maybe | Defer |
| secrets-management | ❌ No | ⚠️ Maybe | Defer |
| pool-registry-types | ❌ No | ❌ No | Archive |
| orchestrator-core | ❌ No | ❌ No | Archive |
| narration-core | ❌ No | ❌ No | Archive |
| narration-macros | ❌ No | ❌ No | Archive |
| audit-logging | ❌ No | ⚠️ Maybe | Archive |
| input-validation | ❌ No | ⚠️ Maybe | Archive |

---

## Next Steps for TEAM-028

1. **Verify auth-min usage:**
   ```bash
   rg "auth-min" --type rust
   ```

2. **Verify deadline-propagation usage:**
   ```bash
   rg "deadline-propagation" --type rust
   ```

3. **Verify secrets-management usage:**
   ```bash
   rg "secrets-management" --type rust
   ```

4. **If unused, archive:**
   - Move to `.archive/`
   - Remove from workspace
   - Document in README

5. **Update workspace Cargo.toml:**
   - Remove archived crates
   - Add comments explaining decisions

---

**Created by:** TEAM-027  
**Status:** Analysis complete, decisions pending verification
