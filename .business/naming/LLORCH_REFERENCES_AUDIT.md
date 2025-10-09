# llorch References Audit

**Date:** 2025-10-09  
**Branch:** rebrand/rbees-naming  
**Status:** Analysis Complete

---

## Summary

Found **~800+ references** to old binary names across the codebase. Categorized by priority and type.

---

## Critical References (MUST Change)

### 1. Rust Source Code (23 matches)

**Library imports in test files:**
```rust
// bin/rbees-workerd/tests/*.rs
use llorch_candled::backend::CandleInferenceBackend;
use llorch_candled::device::init_cpu_device;
use llorch_candled::{InferenceBackend, SamplingConfig};
```

**Files:**
- `bin/rbees-workerd/tests/team_009_smoke.rs`
- `bin/rbees-workerd/tests/team_011_integration.rs`
- `bin/rbees-workerd/tests/team_013_cuda_integration.rs`
- `bin/rbees-workerd/tests/multi_model_support.rs`
- `bin/rbees-workerd/src/main.rs`
- `bin/rbees-workerd/src/bin/cpu.rs`
- `bin/rbees-workerd/src/bin/cuda.rs`
- `bin/rbees-workerd/src/bin/metal.rs`

**Action:** Change `llorch_candled` → `rbees_workerd`

---

### 2. Binary Command References in Rust (114 matches)

**In `bin/rbees-ctl/src/commands/pool.rs`:**
```rust
// OLD
"cd ~/Projects/llama-orch && ./target/release/llorch-pool models download {}"
"cd ~/Projects/llama-orch && cargo build --release -p pool-ctl"

// NEW
"cd ~/Projects/llama-orch && ./target/release/rbees-pool models download {}"
"cd ~/Projects/llama-orch && cargo build --release -p rbees-pool"
```

**Files:**
- `bin/rbees-ctl/src/commands/pool.rs` (10+ matches)
- `bin/rbees-ctl/src/main.rs` (comments about orchestratord)
- `bin/rbees-ctl/src/cli.rs` (comment about pool-ctl)

**Action:** Update all binary paths and package names

---

### 3. Binary Names in Source Code (5 matches)

**In `bin/rbees-workerd/src/main.rs`:**
```rust
#[command(name = "llorch-candled")]  // OLD
#[command(name = "rbees-workerd")]   // NEW
```

**In `bin/rbees-workerd/src/narration.rs`:**
```rust
pub const ACTOR_LLORCH_CANDLED: &str = "llorch-candled";  // OLD
pub const ACTOR_RBEES_WORKERD: &str = "rbees-workerd";    // NEW
```

**Files:**
- `bin/rbees-workerd/src/main.rs` (clap command name, doc comments)
- `bin/rbees-workerd/src/error.rs` (doc comment)
- `bin/rbees-workerd/src/narration.rs` (actor constant)
- `bin/rbees-workerd/src/lib.rs` (doc comment)

**Action:** Update command names and constants

---

### 4. Shell Scripts (7 matches)

**Files:**
- `bin/rbees-workerd/download_test_model.sh` (3 matches)
- `.docs/testing/test_multi_model.sh` (2 matches)

**Action:** Update binary references in scripts

---

### 5. Config Files (2 matches)

**Files:**
- `bin/rbees-workerd/.clippy.toml` (1 match)
- `bin/rbees-workerd/.llorch-test.toml` (1 match - filename and content)

**Action:** Rename `.llorch-test.toml` → `.rbees-test.toml` and update content

---

## High Priority References (SHOULD Change)

### 6. Documentation - README Files (50+ matches)

**In `bin/rbees-ctl/README.md`:**
```markdown
# llorch-ctl  → # rbees-ctl
cargo build --release -p llorch-ctl  → cargo build --release -p rbees-ctl
llorch-pool binary  → rbees-pool binary
llorch-candled (worker)  → rbees-workerd (worker)
pool-ctl (on mac)  → rbees-pool (on mac)
```

**Files:**
- `bin/rbees-ctl/README.md`
- `bin/rbees-pool/README.md` (likely)
- `bin/rbees-workerd/README.md` (likely)
- Root `README.md` (likely)

**Action:** Update all README files with new binary names

---

### 7. Shared Crates Documentation (674 matches in 63 files)

**Major files:**
- `bin/shared-crates/narration-core/README.md` (50+ matches of "orchestratord")
- `bin/shared-crates/narration-core/TEAM_RESPONSIBILITY.md`
- `bin/shared-crates/secrets-management/README.md`
- `consumers/llama-orch-sdk/TEAM_RESPONSIBILITIES.md`
- `consumers/llama-orch-sdk/REFACTOR_PLAN.md`

**Content examples:**
```markdown
orchestratord → rbees-orcd
pool-managerd → (removed - no daemon)
worker-orcd → rbees-workerd
llorch-candled → rbees-workerd
```

**Action:** Update all documentation references

---

### 8. Test Files - Narration Macros (20+ matches)

**In `bin/shared-crates/narration-macros/tests/*.rs`:**
```rust
// Test module names
mod orchestratord { }  → mod rbees_orcd { }

// Test assertions
assert_eq!(test_function(), "orchestratord");  → assert_eq!(test_function(), "rbees-orcd");
```

**Files:**
- `bin/shared-crates/narration-macros/tests/test_actor_inference.rs`
- `bin/shared-crates/narration-macros/tests/smoke_foundation_engineer.rs`

**Action:** Update test module names and assertions

---

## Medium Priority References (CONSIDER Changing)

### 9. Architecture Decision Comments

**In `bin/rbees-ctl/src/main.rs`:**
```rust
//! The orchestrator HTTP daemon is `orchestratord` (separate binary, M2)
//! Binary: `orchestratord` (not yet built, M2)
```

**Action:** Update to `rbees-orcd` and milestone M1

---

### 10. Spec Files and Plans

**Likely in:**
- `bin/.specs/*.md`
- `bin/.plan/*.md`
- `.docs/*.md`

**Action:** Update spec references (lower priority, historical docs)

---

## Low Priority References (Optional)

### 11. Historical Team Handoffs

**In `bin/rbees-workerd/.specs/TEAM_*_HANDOFF.md`:**
- Historical references to old names
- May not need updating (historical record)

**Action:** Leave as-is (historical documentation)

---

## Breakdown by File Type

| Type | Count | Priority | Action |
|------|-------|----------|--------|
| Rust source (`.rs`) | 23 | CRITICAL | Update all |
| Rust source (binary refs) | 114 | CRITICAL | Update all |
| Shell scripts (`.sh`) | 7 | CRITICAL | Update all |
| Config files (`.toml`) | 2 | CRITICAL | Update all |
| README files (`.md`) | 50+ | HIGH | Update all |
| Documentation (`.md`) | 674 | HIGH | Update most |
| Test files (`.rs`) | 20+ | HIGH | Update all |
| Spec files (`.md`) | Unknown | MEDIUM | Update selectively |
| Historical docs | Unknown | LOW | Leave as-is |

**Total:** ~800+ references

---

## Recommended Update Order

### Phase 1: Critical (Breaks Compilation)
1. ✅ Cargo.toml package names (DONE)
2. ✅ Cargo.toml binary names (DONE)
3. ⏳ Rust library imports (`llorch_candled` → `rbees_workerd`)
4. ⏳ Rust binary command names (clap)
5. ⏳ Rust constants (ACTOR_LLORCH_CANDLED)

### Phase 2: High Priority (Breaks Functionality)
6. ⏳ Binary path references in `rbees-ctl/src/commands/pool.rs`
7. ⏳ Shell scripts
8. ⏳ Config files (.toml)

### Phase 3: Documentation
9. ⏳ README files
10. ⏳ Shared crates documentation
11. ⏳ Test files

### Phase 4: Optional
12. ⏳ Spec files (selective)
13. ⏳ Historical docs (leave as-is)

---

## Automated Update Strategy

### Find & Replace Patterns

**Rust imports:**
```bash
find . -name "*.rs" -type f -not -path "*/target/*" -not -path "*/.git/*" | xargs sed -i 's/llorch_candled/rbees_workerd/g'
```

**Binary names in strings:**
```bash
find . -name "*.rs" -type f -not -path "*/target/*" -not -path "*/.git/*" | xargs sed -i 's/llorch-candled/rbees-workerd/g'
find . -name "*.rs" -type f -not -path "*/target/*" -not -path "*/.git/*" | xargs sed -i 's/llorch-pool/rbees-pool/g'
find . -name "*.rs" -type f -not -path "*/target/*" -not -path "*/.git/*" | xargs sed -i 's/llorch-ctl/rbees-ctl/g'
```

**Documentation:**
```bash
find . -name "*.md" -type f -not -path "*/target/*" -not -path "*/.git/*" -not -path "*/.business/*" | xargs sed -i 's/llorch-candled/rbees-workerd/g'
find . -name "*.md" -type f -not -path "*/target/*" -not -path "*/.git/*" -not -path "*/.business/*" | xargs sed -i 's/llorch-pool/rbees-pool/g'
find . -name "*.md" -type f -not -path "*/target/*" -not -path "*/.git/*" -not -path "*/.business/*" | xargs sed -i 's/llorch-ctl/rbees-ctl/g'
find . -name "*.md" -type f -not -path "*/target/*" -not -path "*/.git/*" -not -path "*/.business/*" | xargs sed -i 's/orchestratord/rbees-orcd/g'
find . -name "*.md" -type f -not -path "*/target/*" -not -path "*/.git/*" -not -path "*/.business/*" | xargs sed -i 's/pool-ctl/rbees-pool/g'
```

**Shell scripts:**
```bash
find . -name "*.sh" -type f -not -path "*/target/*" -not -path "*/.git/*" | xargs sed -i 's/llorch-candled/rbees-workerd/g'
find . -name "*.sh" -type f -not -path "*/target/*" -not -path "*/.git/*" | xargs sed -i 's/llorch-pool/rbees-pool/g'
```

**Config files:**
```bash
find . -name "*.toml" -type f -not -path "*/target/*" -not -path "*/.git/*" -not -name "Cargo.toml" | xargs sed -i 's/llorch-candled/rbees-workerd/g'
```

---

## Special Cases

### 1. Narration Actor Constants

**File:** `bin/rbees-workerd/src/narration.rs`

```rust
// OLD
pub const ACTOR_LLORCH_CANDLED: &str = "llorch-candled";

// NEW
pub const ACTOR_RBEES_WORKERD: &str = "rbees-workerd";
```

**Impact:** All narration logs will change actor name

---

### 2. Test Module Names

**File:** `bin/shared-crates/narration-macros/tests/*.rs`

```rust
// OLD
mod orchestratord { }

// NEW  
mod rbees_orcd { }
```

**Impact:** Test module structure changes

---

### 3. Config File Rename

**File:** `bin/rbees-workerd/.llorch-test.toml`

```bash
git mv bin/rbees-workerd/.llorch-test.toml bin/rbees-workerd/.rbees-test.toml
```

---

## Verification Commands

After updates, verify no old references remain:

```bash
# Check Rust files
rg 'llorch_candled|llorch-candled|llorch-pool|llorch-ctl|pool-ctl' --type rust

# Check documentation
rg 'llorch-candled|llorch-pool|llorch-ctl|orchestratord|pool-ctl' --type md -g '!.business/*'

# Check shell scripts
rg 'llorch-candled|llorch-pool|llorch-ctl' --type sh

# Check config files
rg 'llorch-candled|llorch-pool' --type toml -g '!Cargo.toml'
```

---

## Estimated Time

- **Phase 1 (Critical):** 30 minutes
- **Phase 2 (High Priority):** 30 minutes
- **Phase 3 (Documentation):** 1 hour
- **Phase 4 (Optional):** 30 minutes

**Total:** ~2.5 hours

---

**Status:** Ready for execution  
**Next Step:** Run automated updates for Phase 1-2, then manual review

---

**rbees: Your distributed swarm, consistently named.** 🐝
