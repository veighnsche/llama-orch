# TEAM-277 Instructions - Part 2: Phase 1 & 2 Details

**Previous:** Part 1 (Overview)  
**This Part:** Detailed steps for Phase 1 (Config) and Phase 2 (Operations)

---

## Phase 1: Add Config Support (Detailed Steps)

### Step 1.1: Create declarative.rs

**File:** `bin/99_shared_crates/rbee-config/src/declarative.rs`

Create new file with config structs. See `.docs/DECLARATIVE_MIGRATION_PLAN.md` lines 250-350 for full code.

**Key structs:**
- `HivesConfig` - Top-level config
- `HiveConfig` - Single hive config
- `WorkerConfig` - Worker config

**Key methods:**
- `HivesConfig::load()` - Load from `~/.config/rbee/hives.conf`
- `HivesConfig::validate()` - Validate config

### Step 1.2: Export from lib.rs

**File:** `bin/99_shared_crates/rbee-config/src/lib.rs`

Add:
```rust
// TEAM-277: Declarative configuration
pub mod declarative;
pub use declarative::{HivesConfig, HiveConfig, WorkerConfig};
```

### Step 1.3: Add dependencies

**File:** `bin/99_shared_crates/rbee-config/Cargo.toml`

Add:
```toml
[dependencies]
toml = "0.8"
dirs = "5.0"
```

### Step 1.4: Test

Create test config:
```bash
mkdir -p ~/.config/rbee
cat > ~/.config/rbee/hives.conf << 'EOF'
[[hive]]
alias = "test-hive"
hostname = "localhost"
ssh_user = "vince"
workers = [
    { type = "vllm", version = "latest" },
]
EOF
```

Verify:
```bash
cargo check -p rbee-config
cargo test -p rbee-config
```

---

## Phase 2: Add Package Operations (Detailed Steps)

### Step 2.1: Add to Operation enum

**File:** `bin/99_shared_crates/rbee-operations/src/lib.rs`

**Location:** Around line 54

Add 6 new operations. See `.docs/PACKAGE_MANAGER_OPERATIONS.md` lines 50-150 for full code.

**Operations to add:**
- `PackageSync`
- `PackageStatus`
- `PackageInstall`
- `PackageUninstall`
- `PackageValidate`
- `PackageMigrate`

### Step 2.2: Add to Operation::name()

**Location:** Around line 148

Add:
```rust
Operation::PackageSync { .. } => "package_sync",
Operation::PackageStatus { .. } => "package_status",
Operation::PackageInstall { .. } => "package_install",
Operation::PackageUninstall { .. } => "package_uninstall",
Operation::PackageValidate { .. } => "package_validate",
Operation::PackageMigrate { .. } => "package_migrate",
```

### Step 2.3: Update should_forward_to_hive()

**Location:** Around line 305

**Important:** Package operations are NOT forwarded to hive via `should_forward_to_hive()`.
They are handled directly in queen-rbee's job_router.rs (see Phase 3, Step 3.6).

Update the doc comment in `should_forward_to_hive()` to mention:
- Package operations are handled by queen (orchestration)
- Worker/Model operations are forwarded to hive (execution)

### Step 2.4: Verify

```bash
cargo check -p rbee-operations
cargo test -p rbee-operations
```

✅ **Phase 2 complete when all operations compile**

---

**Continue to Part 3 for Phase 3 (Package Manager Implementation)**
