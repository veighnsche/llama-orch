# Profile Terminology Find/Replace Plan

**Date**: 2025-10-01  
**Related**: `.docs/PROFILE_TERMINOLOGY_REPLACEMENT.md`

---

## Terminology Mapping

### Primary Replacements

| Find | Replace | Context |
|------|---------|---------|
| `HOME_PROFILE` | `EMBEDDED_MODE` | All caps (constants, env vars) |
| `CLOUD_PROFILE` | `DISTRIBUTED_MODE` | All caps (constants, env vars) |
| `Home Profile` | `Embedded Mode` | Title case (headings, prose) |
| `Cloud Profile` | `Distributed Mode` | Title case (headings, prose) |
| `home profile` | `embedded mode` | Lower case (inline prose) |
| `cloud profile` | `distributed mode` | Lower case (inline prose) |
| `home-profile` | `embedded-mode` | Kebab case (file names, URLs) |
| `cloud-profile` | `distributed-mode` | Kebab case (file names, URLs) |
| `home_profile` | `embedded_mode` | Snake case (function names, variables) |
| `cloud_profile` | `distributed_mode` | Snake case (function names, variables) |

### Specific Code Patterns

| Find | Replace | File Pattern |
|------|---------|--------------|
| `cloud_profile_enabled()` | `distributed_mode_enabled()` | `*.rs` |
| `select_pool_home()` | `select_pool_embedded()` | `*.rs` |
| `select_pool_cloud()` | `select_pool_distributed()` | `*.rs` |
| `is_pool_dispatchable_home()` | `is_pool_dispatchable_embedded()` | `*.rs` |
| `is_pool_dispatchable_cloud()` | `is_pool_dispatchable_distributed()` | `*.rs` |
| `ORCHESTRATORD_CLOUD_PROFILE` | `ORCHESTRATORD_DISTRIBUTED_MODE` | All files |
| `HME-` | `EMB-` | Requirement IDs in specs |

### Environment Variables

| Old Variable | New Variable | Deprecation Plan |
|--------------|--------------|------------------|
| `ORCHESTRATORD_CLOUD_PROFILE` | `ORCHESTRATORD_DISTRIBUTED_MODE` | Support both in v0.2.x, warn on old, remove in v0.3.0 |

---

## File Renames

### Specs
```bash
.specs/00_home_profile.md → .specs/00_embedded_mode.md
.specs/01_cloud_profile.md → .specs/01_distributed_mode.md
.specs/01_cloud_profile_.md → .specs/01_distributed_mode_.md  # (or delete if duplicate)
```

### Requirements
```bash
requirements/00_home_profile.yaml → requirements/00_embedded_mode.yaml
```

### Documentation
```bash
.docs/HOME_PROFILE.md → .docs/EMBEDDED_MODE.md
.docs/CLOUD_PROFILE_*.md → .docs/DONE/CLOUD_PROFILE_*.md  # Archive completed migration docs
CLOUD_PROFILE_MIGRATION_PLAN.md → .docs/DONE/CLOUD_PROFILE_MIGRATION_PLAN.md
TODO_CLOUD_PROFILE.md → .docs/DONE/TODO_CLOUD_PROFILE.md
docs/runbooks/CLOUD_PROFILE_INCIDENTS.md → docs/runbooks/DISTRIBUTED_MODE_INCIDENTS.md
```

### Observability Specs
```bash
libs/observability/narration-core/.specs/CLOUD_PROFILE_NARRATION_REQUIREMENTS.md → 
libs/observability/narration-core/.specs/DISTRIBUTED_MODE_NARRATION_REQUIREMENTS.md
```

### Tests
```bash
bin/orchestratord/tests/cloud_profile_integration.rs → bin/orchestratord/tests/distributed_mode_integration.rs
```

### Proof Bundle Templates
```bash
.proof_bundle/templates/home-profile-smoke/ → .proof_bundle/templates/embedded-mode-smoke/
```

### CI/Dashboards
```bash
ci/alerts/cloud_profile.yml → ci/alerts/distributed_mode.yml
ci/dashboards/cloud_profile_overview.json → ci/dashboards/distributed_mode_overview.json
```

---

## Files to Update (Content Only)

### High Priority (Core Functionality)

#### Rust Source Files
- `bin/orchestratord/src/state.rs` — `cloud_profile_enabled()` function
- `bin/orchestratord/src/services/placement_v2.rs` — Comments and function names
- `bin/orchestratord/src/api/nodes.rs` — Comments
- `bin/orchestratord/src/api/catalog_availability.rs` — Comments
- `bin/orchestratord/src/app/router.rs` — Comments
- `bin/orchestratord/bdd/src/steps/background.rs` — Comments
- `bin/pool-managerd/src/config.rs` — Module doc comments
- `bin/pool-managerd/src/main.rs` — Comments
- `libs/control-plane/service-registry/src/lib.rs` — Module doc comments
- `libs/shared/pool-registry-types/src/lib.rs` — Module doc comments
- `libs/observability/narration-core/src/auto.rs` — Comments
- `libs/observability/narration-core/src/otel.rs` — Comments
- `libs/observability/narration-core/src/http.rs` — Comments

#### Test Files
- `bin/orchestratord/tests/placement_v2_tests.rs` — Test names, env vars, comments
- `bin/orchestratord/tests/cloud_profile_integration.rs` — File rename + content

#### Configuration & Docs
- `README.md` — "Deployment Profiles" section
- `docs/CONFIGURATION.md` — Environment variable descriptions
- `AGENTS.md` — Repository guidelines (if mentioned)
- `CHECKLIST_HAIKU.md` — Checklist items
- `COMPLIANCE.md` — Requirement references

### Medium Priority (Documentation)

#### Specs
- `.specs/00_home_profile.md` → `.specs/00_embedded_mode.md`
- `.specs/01_cloud_profile.md` → `.specs/01_distributed_mode.md`
- `.specs/01_cloud_profile_.md` → Delete or merge
- `libs/observability/narration-core/.specs/00_narration_core.md`
- `libs/observability/narration-core/.specs/CLOUD_PROFILE_NARRATION_REQUIREMENTS.md`

#### Migration/History Docs (Archive)
- `CLOUD_PROFILE_MIGRATION_PLAN.md` → Archive to `.docs/DONE/`
- `TODO_CLOUD_PROFILE.md` → Archive to `.docs/DONE/`
- `NARRATION_CORE_CLOUD_PROFILE_SUMMARY.md` → Archive to `.docs/DONE/`
- `.docs/CLOUD_PROFILE_*.md` → Archive all to `.docs/DONE/`

#### Implementation Summaries
- `.docs/PHASE*.md` files — Update references
- `.docs/ARCHITECTURE_LIBRARY_ORGANIZATION.md`
- `.docs/workflow.md`
- `.docs/PROJECT_GUIDE.md`

#### BDD/Testing Docs
- `bin/orchestratord/HANDOFF_WATCHER_*.md`
- `bin/pool-managerd/bdd/HANDOFF_WATCHER_RESPONSE.md`
- `bin/pool-managerd/STUB_ANALYSIS.md`
- `bin/pool-managerd/RESPONSIBILITY_AUDIT.md`
- `.docs/testing/TESTING_POLICY.md`
- `.docs/testing/TEST_TYPES_GUIDE.md`
- `.docs/testing/types/smoke.md`

### Low Priority (Historical/Reference)

#### Completed Work Docs
- `.docs/DONE/TODO-*.md` — Leave as-is (historical)
- `.docs/MIGRATION_*.md` — Update if actively referenced
- `.docs/DEAD_CODE_ANALYSIS.md` — Update references
- `.docs/EMAIL_DOCUMENTATION_WRITER.md` — Update if needed

#### Reports
- `.reports/LANG-DISCOVERY/*.md` — Update references
- `.reports/financial-plan/DEV_NOTES.md` — Likely unrelated

#### Plans
- `.plan/*.md` — Update references in active plans

---

## Execution Order

### Step 1: Rename Files (Git)
```bash
# Specs
git mv .specs/00_home_profile.md .specs/00_embedded_mode.md
git mv .specs/01_cloud_profile.md .specs/01_distributed_mode.md
git mv requirements/00_home_profile.yaml requirements/00_embedded_mode.yaml

# Docs
git mv .docs/HOME_PROFILE.md .docs/EMBEDDED_MODE.md
git mv docs/runbooks/CLOUD_PROFILE_INCIDENTS.md docs/runbooks/DISTRIBUTED_MODE_INCIDENTS.md

# Observability
git mv libs/observability/narration-core/.specs/CLOUD_PROFILE_NARRATION_REQUIREMENTS.md \
       libs/observability/narration-core/.specs/DISTRIBUTED_MODE_NARRATION_REQUIREMENTS.md

# Tests
git mv bin/orchestratord/tests/cloud_profile_integration.rs \
       bin/orchestratord/tests/distributed_mode_integration.rs

# Proof bundles
git mv .proof_bundle/templates/home-profile-smoke \
       .proof_bundle/templates/embedded-mode-smoke

# CI
git mv ci/alerts/cloud_profile.yml ci/alerts/distributed_mode.yml
git mv ci/dashboards/cloud_profile_overview.json ci/dashboards/distributed_mode_overview.json

# Archive completed migration docs
mkdir -p .docs/DONE/cloud-profile-migration
git mv CLOUD_PROFILE_MIGRATION_PLAN.md .docs/DONE/cloud-profile-migration/
git mv TODO_CLOUD_PROFILE.md .docs/DONE/cloud-profile-migration/
git mv NARRATION_CORE_CLOUD_PROFILE_SUMMARY.md .docs/DONE/cloud-profile-migration/
git mv .docs/CLOUD_PROFILE_*.md .docs/DONE/cloud-profile-migration/
```

### Step 2: Update File Contents (Find/Replace)

Use `sed` or editor find/replace across all files:

```bash
# Find all files that need updating (excluding .git, target, node_modules)
fd -t f -e md -e rs -e toml -e yaml -e yml -e json \
   --exclude target --exclude node_modules --exclude .git

# For each pattern, run replacements
# (Use MultiEdit tool or sed for batch operations)
```

### Step 3: Update Code (Manual Review)

Carefully update:
- Function names in `*.rs` files
- Environment variable handling with deprecation warnings
- Test assertions and expectations
- Module documentation

### Step 4: Verify

```bash
# Check for remaining references
rg "HOME_PROFILE|CLOUD_PROFILE" --type-not md --glob '!.docs/DONE/**'

# Run tests
cargo xtask dev:loop

# Check links
bash ci/scripts/check_links.sh
```

---

## Deprecation Warning Implementation

Add to `bin/orchestratord/src/state.rs`:

```rust
pub fn distributed_mode_enabled(&self) -> bool {
    // Support both old and new env var names
    let new_var = std::env::var("ORCHESTRATORD_DISTRIBUTED_MODE")
        .ok()
        .and_then(|v| v.parse::<bool>().ok());
    
    let old_var = std::env::var("ORCHESTRATORD_CLOUD_PROFILE")
        .ok()
        .and_then(|v| v.parse::<bool>().ok());
    
    if old_var.is_some() {
        tracing::warn!(
            "ORCHESTRATORD_CLOUD_PROFILE is deprecated, use ORCHESTRATORD_DISTRIBUTED_MODE instead. \
             Support for the old variable will be removed in v0.3.0"
        );
    }
    
    new_var.or(old_var).unwrap_or(false)
}
```

---

## Post-Migration Cleanup (v0.3.0)

In v0.3.0, remove:
- Deprecation warning code
- Support for `ORCHESTRATORD_CLOUD_PROFILE`
- Any remaining references in archived docs

---

## Verification Checklist

- [ ] No references to `HOME_PROFILE` in active code/docs (except archived)
- [ ] No references to `CLOUD_PROFILE` in active code/docs (except archived)
- [ ] All tests pass: `cargo test --workspace --all-features`
- [ ] BDD tests pass: `cargo test -p test-harness-bdd`
- [ ] Clippy clean: `cargo clippy --all-targets --all-features`
- [ ] Links valid: `bash ci/scripts/check_links.sh`
- [ ] Environment variable deprecation warning works
- [ ] Documentation is consistent and clear
- [ ] CHANGELOG.md updated with breaking changes
- [ ] README.md "Deployment Profiles" section updated
- [ ] AGENTS.md updated if profile terminology was mentioned

---

## Rollback Plan

If issues arise:
1. Revert the feature branch
2. Keep the proposal doc for future attempt
3. Document what went wrong
4. Fix issues before retry

Git makes file renames reversible, and content changes are in version control.

---

## Communication

Update:
- [ ] CHANGELOG.md — Breaking change notice
- [ ] Migration guide in docs/
- [ ] Release notes for v0.2.x
- [ ] Update any external documentation/wiki

---

**Ready to execute**: Once terminology decision is finalized (EMBEDDED/DISTRIBUTED vs SINGLE_NODE/MULTI_NODE)
