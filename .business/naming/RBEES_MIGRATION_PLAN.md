# rbees Migration Plan - Complete Renaming

**Date:** 2025-10-09  
**Status:** Ready to Execute  
**Estimated Time:** 2-3 hours

---

## Overview

Rename all binaries, crates, and references from `llorch`/`orchestratord`/`pool-ctl` to the `rbees` brand.

**Key Changes:**
- `orchestratord` → `rbees-orcd`
- `llorch-ctl` (`llorch`) → `rbees-ctl` (`rbees`)
- `pool-ctl` (`llorch-pool`) → `rbees-pool`
- `llorch-candled` → `rbees-workerd`
- `orchestrator-core` → `rbees-orchestrator-core`
- `pool-core` → `rbees-pool-core`

---

## Phase 1: Rename Binary Directories

### 1.1 Rename Orchestrator Daemon (if exists)

```bash
# Check if directory exists
if [ -d "bin/orchestratord" ]; then
  git mv bin/orchestratord bin/rbees-orcd
fi
```

**Files to update in `bin/rbees-orcd/`:**
- `Cargo.toml` - package name, binary name
- `src/main.rs` - any self-references
- `.specs/*.md` - update all references

---

### 1.2 Rename Orchestrator CLI

```bash
# Rename directory
git mv bin/llorch-ctl bin/rbees-ctl
```

**Files to update in `bin/rbees-ctl/`:**
- `Cargo.toml`:
  ```toml
  [package]
  name = "rbees-ctl"
  
  [[bin]]
  name = "rbees"  # Command name
  path = "src/main.rs"
  ```
- `src/main.rs` - update clap app name
- `.specs/*.md` - update all references
- `README.md` - update examples

---

### 1.3 Rename Pool CLI

```bash
# Rename directory
git mv bin/pool-ctl bin/rbees-pool
```

**Files to update in `bin/rbees-pool/`:**
- `Cargo.toml`:
  ```toml
  [package]
  name = "rbees-pool"
  
  [[bin]]
  name = "rbees-pool"
  path = "src/main.rs"
  ```
- `src/main.rs` - update clap app name
- `.specs/*.md` - update all references
- `README.md` - update examples

---

### 1.4 Rename Worker Daemon

```bash
# Rename directory
git mv bin/llorch-candled bin/rbees-workerd
```

**Files to update in `bin/rbees-workerd/`:**
- `Cargo.toml`:
  ```toml
  [package]
  name = "rbees-workerd"
  
  [[bin]]
  name = "rbees-workerd"
  path = "src/main.rs"
  ```
- `src/main.rs` - update any self-references
- `.specs/*.md` - update all references
- `README.md` - update examples

---

## Phase 2: Rename Shared Crates

### 2.1 Rename Orchestrator Core (if exists)

```bash
# Check if directory exists
if [ -d "libs/orchestrator-core" ]; then
  git mv libs/orchestrator-core libs/rbees-orchestrator-core
fi
```

**Files to update in `libs/rbees-orchestrator-core/`:**
- `Cargo.toml`:
  ```toml
  [package]
  name = "rbees-orchestrator-core"
  ```

---

### 2.2 Rename Pool Core (if exists)

```bash
# Check if directory exists
if [ -d "libs/pool-core" ]; then
  git mv libs/pool-core libs/rbees-pool-core
fi
```

**Files to update in `libs/rbees-pool-core/`:**
- `Cargo.toml`:
  ```toml
  [package]
  name = "rbees-pool-core"
  ```

---

## Phase 3: Update Dependencies

### 3.1 Update All Cargo.toml Files

**Find all Cargo.toml files that reference old names:**

```bash
# Find all Cargo.toml files
find . -name "Cargo.toml" -type f | grep -v target | grep -v .git

# Search for old crate names
rg "orchestrator-core|pool-core|llorch-candled|llorch-ctl|pool-ctl" --type toml
```

**Update dependencies in each file:**

```toml
# Old
[dependencies]
orchestrator-core = { path = "../libs/orchestrator-core" }
pool-core = { path = "../libs/pool-core" }

# New
[dependencies]
rbees-orchestrator-core = { path = "../libs/rbees-orchestrator-core" }
rbees-pool-core = { path = "../libs/rbees-pool-core" }
```

---

### 3.2 Update Rust Imports

**Find all Rust files with old crate imports:**

```bash
# Search for old crate names in Rust files
rg "use orchestrator_core|use pool_core" --type rust
rg "extern crate orchestrator_core|extern crate pool_core" --type rust
```

**Update imports:**

```rust
// Old
use orchestrator_core::*;
use pool_core::*;

// New
use rbees_orchestrator_core::*;
use rbees_pool_core::*;
```

---

## Phase 4: Update Scripts

### 4.1 Update Bash Scripts

**Scripts to update:**
- `scripts/llorch-git` → `scripts/rbees-git` (or keep as-is)
- `scripts/llorch-models` → `scripts/rbees-models` (or keep as-is)
- `scripts/homelab/llorch-remote` → `scripts/homelab/rbees-remote` (or keep as-is)

**Find all script references:**

```bash
# Find all bash scripts
find scripts/ -type f -name "*.sh" -o -name "llorch-*"

# Search for binary names in scripts
rg "llorch-candled|llorch-pool|llorch|orchestratord|pool-ctl" scripts/
```

**Update binary references:**

```bash
# Old
llorch-candled --model ...
llorch pool status
llorch-pool models download

# New
rbees-workerd --model ...
rbees pool status
rbees-pool models download
```

---

### 4.2 Update CI/CD

**Files to update:**
- `.github/workflows/*.yml`

```bash
# Search for old binary names in CI
rg "llorch-candled|llorch-pool|llorch|orchestratord|pool-ctl" .github/
```

**Update workflow files:**

```yaml
# Old
- name: Build llorch-candled
  run: cargo build --release --bin llorch-candled

# New
- name: Build rbees-workerd
  run: cargo build --release --bin rbees-workerd
```

---

## Phase 5: Update Documentation

### 5.1 Update README Files

**Files to update:**
- `/README.md`
- `/bin/rbees-orcd/README.md`
- `/bin/rbees-ctl/README.md`
- `/bin/rbees-pool/README.md`
- `/bin/rbees-workerd/README.md`

```bash
# Search for old names in README files
rg "llorch-candled|llorch-pool|llorch|orchestratord|pool-ctl" --type md
```

**Update examples:**

```markdown
# Old
## Installation
cargo install --path bin/llorch-candled

## Usage
llorch pool status

# New
## Installation
cargo install --path bin/rbees-workerd

## Usage
rbees pool status
```

---

### 5.2 Update Specification Files

**Files to update:**
- `/bin/.specs/*.md`
- All component specs

```bash
# Search for old names in spec files
rg "llorch-candled|llorch-pool|llorch|orchestratord|pool-ctl" bin/.specs/
```

**Update references:**

```markdown
# Old
The `llorch-candled` binary is the worker daemon.
The `orchestratord` binary is the orchestrator daemon.

# New
The `rbees-workerd` binary is the worker daemon.
The `rbees-orcd` binary is the orchestrator daemon.
```

---

### 5.3 Update Architecture Docs

**Files to update:**
- `/ORCHESTRATION_OVERVIEW.md`
- `/QUICK_STATUS.md`
- `/bin/.specs/COMPLETE_BINARY_ARCHITECTURE.md`
- `/bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md`

```bash
# Search in architecture docs
rg "llorch-candled|llorch-pool|llorch|orchestratord|pool-ctl" \
  ORCHESTRATION_OVERVIEW.md \
  QUICK_STATUS.md \
  bin/.specs/COMPLETE_BINARY_ARCHITECTURE.md \
  bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md
```

---

### 5.4 Update Plan Documents

**Files to update:**
- `/bin/.plan/*.md`
- `/ONE_MONTH_PLAN/*.md`

```bash
# Search in plan documents
rg "llorch-candled|llorch-pool|llorch|orchestratord|pool-ctl" bin/.plan/ ONE_MONTH_PLAN/
```

---

## Phase 6: Update Configuration

### 6.1 Update Config Files

**Files to update:**
- `.llorch.toml.example` → `.rbees.toml.example`
- Any other config files

```bash
# Rename config example
git mv .llorch.toml.example .rbees.toml.example
```

**Update config content:**

```toml
# Old
[llorch]
binary_path = "/usr/local/bin/llorch-candled"

# New
[rbees]
binary_path = "/usr/local/bin/rbees-workerd"
```

---

### 6.2 Update Catalog Files

**Files to update:**
- `bin/rbees-ctl/catalog.toml`
- `bin/rbees-pool/catalog.toml`

```bash
# Search for old binary names in catalog files
rg "llorch-candled|llorch-pool|llorch" --type toml
```

---

## Phase 7: Update Tests

### 7.1 Update Test Files

**Find all test files:**

```bash
# Search for old names in test files
rg "llorch-candled|llorch-pool|llorch|orchestratord|pool-ctl" --type rust -g "*test*"
rg "llorch-candled|llorch-pool|llorch|orchestratord|pool-ctl" test-harness/
```

**Update test code:**

```rust
// Old
let binary = "llorch-candled";
let cmd = Command::new("llorch-pool");

// New
let binary = "rbees-workerd";
let cmd = Command::new("rbees-pool");
```

---

### 7.2 Update BDD Tests

**Files to update:**
- `test-harness/**/*.feature`
- `test-harness/**/*.rs`

```bash
# Search in test harness
rg "llorch-candled|llorch-pool|llorch|orchestratord|pool-ctl" test-harness/
```

---

## Phase 8: Update Build Configuration

### 8.1 Update Root Cargo.toml

**File:** `/Cargo.toml`

```toml
# Old
[workspace]
members = [
    "bin/orchestratord",
    "bin/llorch-ctl",
    "bin/pool-ctl",
    "bin/llorch-candled",
    "libs/orchestrator-core",
    "libs/pool-core",
]

# New
[workspace]
members = [
    "bin/rbees-orcd",
    "bin/rbees-ctl",
    "bin/rbees-pool",
    "bin/rbees-workerd",
    "libs/rbees-orchestrator-core",
    "libs/rbees-pool-core",
]
```

---

### 8.2 Update .gitignore

**File:** `/.gitignore`

```bash
# Search for old binary names
rg "llorch-candled|llorch-pool|llorch|orchestratord|pool-ctl" .gitignore
```

**Update if needed:**

```gitignore
# Old
/target/release/llorch-candled
/target/release/llorch
/target/release/llorch-pool

# New
/target/release/rbees-workerd
/target/release/rbees
/target/release/rbees-pool
```

---

## Phase 9: Verification

### 9.1 Build All Binaries

```bash
# Clean build
cargo clean

# Build all binaries
cargo build --release

# Verify binary names
ls -la target/release/ | grep rbees
```

**Expected output:**
```
rbees
rbees-orcd
rbees-pool
rbees-workerd
```

---

### 9.2 Run Tests

```bash
# Run all tests
cargo test --workspace

# Run specific binary tests
cargo test -p rbees-workerd
cargo test -p rbees-pool
cargo test -p rbees-ctl
```

---

### 9.3 Check for Remaining References

```bash
# Search for any remaining old names (excluding .git and target)
rg "llorch-candled|llorch-pool|orchestratord|pool-ctl" \
  --type-not gitignore \
  -g '!target/*' \
  -g '!.git/*' \
  -g '!*.lock'

# Search for "llorch" as standalone word (might be intentional in some docs)
rg '\bllorch\b' \
  --type-not gitignore \
  -g '!target/*' \
  -g '!.git/*' \
  -g '!*.lock'
```

---

### 9.4 Test Binary Execution

```bash
# Test each binary can run
./target/release/rbees --help
./target/release/rbees-pool --help
./target/release/rbees-workerd --help

# If orchestrator daemon exists
if [ -f "./target/release/rbees-orcd" ]; then
  ./target/release/rbees-orcd --help
fi
```

---

## Phase 10: Git Commit

### 10.1 Stage Changes

```bash
# Stage all renamed files
git add -A

# Review changes
git status
```

---

### 10.2 Commit

```bash
git commit -m "Rebrand: Rename all binaries to rbees

- orchestratord → rbees-orcd
- llorch-ctl (llorch) → rbees-ctl (rbees)
- pool-ctl (llorch-pool) → rbees-pool
- llorch-candled → rbees-workerd
- orchestrator-core → rbees-orchestrator-core
- pool-core → rbees-pool-core

Updated all references in:
- Cargo.toml files (workspace, dependencies)
- Source code (imports, binary names)
- Documentation (README, specs, plans)
- Scripts (bash, CI/CD)
- Configuration files
- Tests (unit, integration, BDD)

Brand: rbees - Your distributed swarm
"
```

---

## Automated Migration Script

**File:** `scripts/migrate-to-rbees.sh`

```bash
#!/bin/bash
set -e

echo "🐝 Migrating to rbees brand..."

# Phase 1: Rename directories
echo "📁 Renaming directories..."
[ -d "bin/orchestratord" ] && git mv bin/orchestratord bin/rbees-orcd
[ -d "bin/llorch-ctl" ] && git mv bin/llorch-ctl bin/rbees-ctl
[ -d "bin/pool-ctl" ] && git mv bin/pool-ctl bin/rbees-pool
[ -d "bin/llorch-candled" ] && git mv bin/llorch-candled bin/rbees-workerd
[ -d "libs/orchestrator-core" ] && git mv libs/orchestrator-core libs/rbees-orchestrator-core
[ -d "libs/pool-core" ] && git mv libs/pool-core libs/rbees-pool-core

# Phase 2: Update Cargo.toml files
echo "📦 Updating Cargo.toml files..."
find . -name "Cargo.toml" -type f -not -path "*/target/*" -not -path "*/.git/*" | while read file; do
  sed -i 's/orchestrator-core/rbees-orchestrator-core/g' "$file"
  sed -i 's/pool-core/rbees-pool-core/g' "$file"
  sed -i 's/name = "llorch-candled"/name = "rbees-workerd"/g' "$file"
  sed -i 's/name = "llorch-ctl"/name = "rbees-ctl"/g' "$file"
  sed -i 's/name = "pool-ctl"/name = "rbees-pool"/g' "$file"
  sed -i 's/name = "orchestratord"/name = "rbees-orcd"/g' "$file"
done

# Phase 3: Update Rust source files
echo "🦀 Updating Rust source files..."
find . -name "*.rs" -type f -not -path "*/target/*" -not -path "*/.git/*" | while read file; do
  sed -i 's/use orchestrator_core/use rbees_orchestrator_core/g' "$file"
  sed -i 's/use pool_core/use rbees_pool_core/g' "$file"
  sed -i 's/extern crate orchestrator_core/extern crate rbees_orchestrator_core/g' "$file"
  sed -i 's/extern crate pool_core/extern crate rbees_pool_core/g' "$file"
done

# Phase 4: Update documentation
echo "📚 Updating documentation..."
find . -name "*.md" -type f -not -path "*/target/*" -not -path "*/.git/*" | while read file; do
  sed -i 's/llorch-candled/rbees-workerd/g' "$file"
  sed -i 's/llorch-pool/rbees-pool/g' "$file"
  sed -i 's/orchestratord/rbees-orcd/g' "$file"
  sed -i 's/pool-ctl/rbees-pool/g' "$file"
  sed -i 's/llorch-ctl/rbees-ctl/g' "$file"
  # Be careful with standalone "llorch" - might need manual review
  sed -i 's/`llorch`/`rbees`/g' "$file"
  sed -i 's/ llorch / rbees /g' "$file"
done

# Phase 5: Update scripts
echo "📜 Updating scripts..."
find scripts/ -type f | while read file; do
  sed -i 's/llorch-candled/rbees-workerd/g' "$file"
  sed -i 's/llorch-pool/rbees-pool/g' "$file"
  sed -i 's/orchestratord/rbees-orcd/g' "$file"
  sed -i 's/pool-ctl/rbees-pool/g' "$file"
  sed -i 's/llorch-ctl/rbees-ctl/g' "$file"
done

# Phase 6: Update config files
echo "⚙️  Updating config files..."
[ -f ".llorch.toml.example" ] && git mv .llorch.toml.example .rbees.toml.example
find . -name "*.toml" -type f -not -path "*/target/*" -not -path "*/.git/*" -not -name "Cargo.toml" | while read file; do
  sed -i 's/llorch-candled/rbees-workerd/g' "$file"
  sed -i 's/llorch-pool/rbees-pool/g' "$file"
  sed -i 's/orchestratord/rbees-orcd/g' "$file"
done

# Phase 7: Update CI/CD
echo "🔄 Updating CI/CD..."
find .github/ -name "*.yml" -type f | while read file; do
  sed -i 's/llorch-candled/rbees-workerd/g' "$file"
  sed -i 's/llorch-pool/rbees-pool/g' "$file"
  sed -i 's/orchestratord/rbees-orcd/g' "$file"
  sed -i 's/pool-ctl/rbees-pool/g' "$file"
  sed -i 's/llorch-ctl/rbees-ctl/g' "$file"
done

echo "✅ Migration complete!"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Build: cargo build --release"
echo "3. Test: cargo test --workspace"
echo "4. Commit: git commit -m 'Rebrand to rbees'"
echo ""
echo "🐝 rbees: Your distributed swarm"
```

---

## Manual Review Checklist

After running automated migration, manually review:

- [ ] All Cargo.toml files have correct package names
- [ ] All binary names in Cargo.toml are correct
- [ ] All Rust imports use new crate names
- [ ] All documentation examples use new binary names
- [ ] All scripts use new binary names
- [ ] All CI/CD workflows use new binary names
- [ ] All test files use new binary names
- [ ] Root Cargo.toml workspace members are updated
- [ ] Config files use new binary names
- [ ] No remaining references to old names (except intentional)

---

## Rollback Plan

If migration fails:

```bash
# Discard all changes
git reset --hard HEAD

# Or if already committed
git revert HEAD
```

---

## Post-Migration Tasks

After successful migration:

1. **Update documentation website** (if exists)
2. **Update package registries** (crates.io, if published)
3. **Update GitHub repository description**
4. **Update social media handles** (if applicable)
5. **Notify users** of the rebrand
6. **Update any external references** (blog posts, tutorials, etc.)

---

## Estimated Timeline

- **Phase 1-2:** 15 minutes (rename directories, update Cargo.toml)
- **Phase 3:** 10 minutes (update dependencies)
- **Phase 4:** 15 minutes (update scripts)
- **Phase 5:** 30 minutes (update documentation)
- **Phase 6:** 10 minutes (update configuration)
- **Phase 7:** 15 minutes (update tests)
- **Phase 8:** 10 minutes (update build config)
- **Phase 9:** 20 minutes (verification)
- **Phase 10:** 5 minutes (commit)

**Total:** ~2.5 hours (with manual review)

**With automation script:** ~30 minutes (script + manual review)

---

**Status:** Ready to Execute  
**Next Step:** Run `scripts/migrate-to-rbees.sh` or execute phases manually

---

**rbees: Your distributed swarm, consistently named.** 🐝
