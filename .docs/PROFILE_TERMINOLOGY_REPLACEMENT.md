# Profile/Mode Abstraction Removal

**Date**: 2025-10-01  
**Status**: PROPOSAL  
**Impact**: Architecture simplification, documentation, code, configuration

---

## Problem Statement

The current codebase uses `HOME_PROFILE` and `CLOUD_PROFILE` terminology throughout specs, documentation, code comments, and configuration. This creates a **false architectural distinction**.

### Why This Is Wrong

1. **False abstraction**: There are no "modes" or "profiles"—just two services that communicate via HTTP
2. **Misleading**: Suggests different product variants when it's just deployment flexibility
3. **Obsolete**: The "cloud profile migration" is complete; this was a migration artifact, not a design
4. **Confusing**: A home can have multiple PCs; a single PC can run both services on localhost

### The Reality

llama-orch has **two services**:
- **orchestratord**: Control plane (no GPU needed) — makes placement decisions, routes requests
- **pool-managerd**: GPU worker (needs GPU) — manages engines, reports capacity

The only difference between deployments is **where you run them**:
- Same machine: `ORCHD_POOL_MANAGERS=http://localhost:9200`
- Different machines: `ORCHD_POOL_MANAGERS=http://gpu-1:9200,http://gpu-2:9200`

That's it. No modes, no profiles, no architectural variants.

## Current Usage

The false abstraction appears in:
- **Specs**: `.specs/00_home_profile.md`, `.specs/01_cloud_profile.md`
- **Documentation**: 59 files with `HOME_PROFILE`, 59 files with `CLOUD_PROFILE`
- **Code**: Conditional logic like `if cloud_profile_enabled() { ... } else { ... }`
- **Configuration**: `ORCHESTRATORD_CLOUD_PROFILE` environment variable
- **Test files**: BDD scenarios, integration tests

## Proposed Solution: Remove the Abstraction

Instead of renaming `HOME_PROFILE` → something else, **delete the concept entirely**.

---

## Two Approaches

### Option A: Remove Abstraction (Recommended)

**Goal**: Unify code paths, remove conditional logic, treat all deployments the same way.

#### What Changes

**Code**:
```rust
// DELETE: cloud_profile_enabled() checks
// DELETE: select_pool_home() / select_pool_cloud() split
// KEEP: Single placement algorithm that queries service registry

// BEFORE
if state.cloud_profile_enabled() {
    self.select_pool_cloud(state)
} else {
    self.select_pool_home(state)
}

// AFTER
pub fn select_pool(&self, state: &AppState) -> Option<PlacementDecision> {
    // Query service registry for available pools
    // Works whether pools are local (localhost:9200) or remote
    state.service_registry.select_best_pool(self.strategy)
}
```

**Configuration**:
```bash
# DELETE: ORCHESTRATORD_CLOUD_PROFILE variable
# KEEP: Just list pool-managerd endpoints

# Single machine
ORCHD_POOL_MANAGERS=http://localhost:9200

# Multiple machines
ORCHD_POOL_MANAGERS=http://gpu-1:9200,http://gpu-2:9200
```

**Specs**:
- DELETE: `.specs/00_home_profile.md`, `.specs/01_cloud_profile.md`
- KEEP/UPDATE: `.specs/20_orchestratord.md` (control plane requirements)
- KEEP/UPDATE: `.specs/30_pool_managerd.md` (GPU worker requirements)
- ADD: `.specs/00_deployment.md` (example configurations, optional)

**Documentation**:
- DELETE: "Deployment Profiles" section in README
- ADD: Simple explanation: "orchestratord talks to pool-managerd via HTTP"
- PROVIDE: Example configs for common setups (1 machine, multiple machines)

#### Benefits
- ✅ Simpler code (one placement path)
- ✅ No false abstractions
- ✅ Easier to understand
- ✅ More flexible (any topology works the same way)

#### Risks
- ⚠️ More invasive code changes
- ⚠️ Requires careful testing
- ⚠️ May uncover hidden assumptions

---

### Option B: Rename Only (Conservative)

**Goal**: Keep code structure, just rename to neutral terms and clarify documentation.

#### What Changes

**Terminology**:
- `HOME_PROFILE` → `LOCAL_DEPLOYMENT` (or `COLOCATED`)
- `CLOUD_PROFILE` → `REMOTE_DEPLOYMENT` (or `NETWORKED`)

**Code**: Minimal
- Rename functions: `cloud_profile_enabled()` → `remote_deployment_enabled()`
- Update comments
- Keep conditional logic as-is

**Configuration**:
- `ORCHESTRATORD_CLOUD_PROFILE` → `ORCHESTRATORD_REMOTE_DEPLOYMENT`
- Support both with deprecation warning

**Specs**:
- Rename files but keep structure
- Clarify that this is just a deployment detail, not architectural modes

#### Benefits
- ✅ Low risk
- ✅ Fast to implement
- ✅ Preserves existing logic

#### Risks
- ⚠️ Still perpetuates the false abstraction
- ⚠️ Doesn't simplify the codebase
- ⚠️ May need Option A eventually anyway

---/home/vince/Projects/llama-orch/.docs/PROFILE_TERMINOLOGY_REPLACEMENT.md

## Recommended Approach: Option A (Remove Abstraction)

The profile/mode concept is fundamentally wrong. Renaming it just kicks the problem down the road.

---

## Implementation Plan (Option A)

### Phase 1: Code Simplification (4-6 hours)

**Goal**: Remove conditional logic, unify placement algorithm.

1. **Delete profile detection**:
   - Remove `cloud_profile_enabled()` from `bin/orchestratord/src/state.rs`
   - Remove `ORCHESTRATORD_CLOUD_PROFILE` environment variable handling
   - Service registry is always used (initialized with pool-managerd endpoints)

2. **Unify placement service**:
   - File: `bin/orchestratord/src/services/placement_v2.rs`
   - Delete `select_pool_home()` and `select_pool_cloud()` split
   - Keep single `select_pool()` that queries service registry
   - Delete `is_pool_dispatchable_home()` / `is_pool_dispatchable_cloud()` split

3. **Update service registry**:
   - File: `libs/control-plane/service-registry/src/lib.rs`
   - Remove "CLOUD_PROFILE" from module docs
   - Registry is always active (not conditional)

4. **Update tests**:
   - Remove `std::env::set_var("ORCHESTRATORD_CLOUD_PROFILE", "true")` from tests
   - Tests work the same way regardless of deployment
   - Rename `tests/cloud_profile_integration.rs` → `tests/service_registry_integration.rs`

### Phase 2: Specs & Requirements (2-3 hours)

**Goal**: Reorganize specs around service responsibilities, not deployment scenarios.

1. **Delete profile specs**:
   - Remove `.specs/00_home_profile.md`
   - Remove `.specs/01_cloud_profile.md`
   - Remove `requirements/00_home_profile.yaml`

2. **Update service specs**:
   - `.specs/20_orchestratord.md`: Add requirements from home/cloud specs (control plane responsibilities)
   - `.specs/30_pool_managerd.md`: Add requirements from home/cloud specs (GPU worker responsibilities)

3. **Optional deployment guide**:
   - Create `.specs/00_deployment.md` with example configurations
   - Show single-machine and multi-machine setups
   - Emphasize it's just URL configuration

4. **Update requirement IDs**:
   - Migrate `HME-` requirements → `ORCH-` or `ORCHD-` series
   - Update `requirements/*.yaml` files

### Phase 3: Documentation (2-3 hours)

**Goal**: Remove "profiles" language, focus on service architecture.

1. **README.md**:
   - Delete "Deployment Profiles" section
   - Add simple explanation: "orchestratord (control) + pool-managerd (GPU workers)"
   - Show example configs for common setups

2. **Archive migration docs**:
   ```bash
   mkdir -p .docs/DONE/cloud-profile-migration
   git mv CLOUD_PROFILE_*.md .docs/DONE/cloud-profile-migration/
   git mv TODO_CLOUD_PROFILE.md .docs/DONE/cloud-profile-migration/
   git mv .docs/CLOUD_PROFILE_*.md .docs/DONE/cloud-profile-migration/
   git mv .docs/HOME_PROFILE.md .docs/DONE/cloud-profile-migration/
   ```

3. **Update operational docs**:
   - `docs/CONFIGURATION.md`: Remove `ORCHESTRATORD_CLOUD_PROFILE`, document `ORCHD_POOL_MANAGERS`
   - `docs/runbooks/`: Remove "cloud profile" language
   - `.docs/ARCHITECTURE_*.md`: Update to reflect unified architecture

4. **Update AGENTS.md**:
   - Remove profile terminology from guidelines
   - Emphasize service-based architecture

### Phase 4: Comments & Cleanup (1-2 hours)

**Goal**: Remove profile references from code comments.

1. **Update code comments**:
   - Remove `// HOME_PROFILE:` and `// CLOUD_PROFILE:` comments
   - Replace with service-specific comments if needed
   - Update module documentation

2. **Update observability specs**:
   - Rename `libs/observability/narration-core/.specs/CLOUD_PROFILE_NARRATION_REQUIREMENTS.md`
   - Remove profile language from narration specs

3. **Update CI/dashboards**:
   - Rename `ci/alerts/cloud_profile.yml` → `ci/alerts/distributed_deployment.yml` (or just remove profile from names)
   - Update dashboard titles

### Phase 5: Verification (1 hour)

1. **Run full test suite**:
   ```bash
   cargo xtask dev:loop
   ```

2. **Check for remaining references**:
   ```bash
   rg "HOME_PROFILE|CLOUD_PROFILE|home.profile|cloud.profile" \
      --type-not md --glob '!.docs/DONE/**'
   ```

3. **Update CHANGELOG.md**:
   - Document removal of `ORCHESTRATORD_CLOUD_PROFILE` variable
   - Explain architectural simplification
   - Note: No functional changes, just cleaner code

---

## Breaking Changes

### Removed Environment Variable
- **Deleted**: `ORCHESTRATORD_CLOUD_PROFILE`
- **Replacement**: None needed—service registry is always active
- **Migration**: Remove the variable from your configs; orchestratord will ignore it

### Configuration Change
- **Before**: Set `ORCHESTRATORD_CLOUD_PROFILE=true` to enable multi-node
- **After**: Just configure `ORCHD_POOL_MANAGERS` with your pool-managerd endpoints
  - Single machine: `ORCHD_POOL_MANAGERS=http://localhost:9200`
  - Multiple machines: `ORCHD_POOL_MANAGERS=http://gpu-1:9200,http://gpu-2:9200`

---

## Estimated Effort

**Total**: 10-15 hours

**Breakdown**:
- Code simplification: 4-6 hours
- Specs & requirements: 2-3 hours
- Documentation: 2-3 hours
- Comments & cleanup: 1-2 hours
- Verification: 1 hour

---

## Risks & Mitigation

### Risk: Hidden Dependencies on Profile Logic
**Mitigation**: Comprehensive test suite will catch issues; BDD scenarios cover both localhost and remote deployments

### Risk: Breaking User Configs
**Mitigation**: Document breaking change clearly in CHANGELOG; provide migration guide

### Risk: Incomplete Removal
**Mitigation**: Use `rg` to find all references; review archived docs to ensure nothing active remains

---

## Success Criteria

- [ ] No `cloud_profile_enabled()` checks in code
- [ ] Single placement algorithm (no home/cloud split)
- [ ] No `ORCHESTRATORD_CLOUD_PROFILE` environment variable
- [ ] Specs organized by service, not deployment scenario
- [ ] Documentation focuses on architecture, not "profiles"
- [ ] All tests pass
- [ ] No profile terminology in active code/docs (except archived history)

---

## Next Steps

1. **Approve this approach**
2. **Create feature branch**: `refactor/remove-profile-abstraction`
3. **Execute phases 1-5**
4. **Review PR with focus on**:
   - Code simplification correctness
   - Test coverage maintained
   - Documentation clarity
5. **Merge and update TODO.md**

---

## References

- Current code: `bin/orchestratord/src/services/placement_v2.rs`, `bin/orchestratord/src/state.rs`
- Current specs: `.specs/00_home_profile.md`, `.specs/01_cloud_profile.md`
- Service registry: `libs/control-plane/service-registry/`
- Testing: `bin/orchestratord/tests/cloud_profile_integration.rs`
