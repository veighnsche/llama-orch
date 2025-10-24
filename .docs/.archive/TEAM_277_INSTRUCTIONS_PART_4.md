# TEAM-277 Instructions - Part 4: Phase 4, 5, 6

**Previous:** Part 3 (Phase 3)  
**This Part:** Phase 4 (Simplify Hive), Phase 5 (CLI), Phase 6 (Cleanup)

---

## Phase 4: Simplify Hive (8-12 hours)

### Goal
Remove worker installation logic from rbee-hive. Hive only manages processes now.

### Step 4.1: Worker Installation is Already Stubbed

**Note:** `install.rs` and `uninstall.rs` are already stubs (TEAM-276).
They delegate to worker-catalog and don't perform actual installation.

**Action:** Update documentation in these files to clarify that:
- Queen will handle worker installation via SSH (new TEAM-277 pattern)
- These stubs remain for API consistency
- No deletion needed - they're part of the public API

### Step 4.2: Update worker-lifecycle Documentation

**File:** `bin/25_rbee_hive_crates/worker-lifecycle/src/lib.rs`

**Keep all modules** (install/uninstall are API stubs):
```rust
pub mod install;    // KEEP - API stub for consistency
pub mod uninstall;  // KEEP - API stub for consistency
```

Update doc comment to clarify:
```rust
//! # TEAM-277 Architecture Update
//!
//! Worker installation is now handled by queen-rbee via SSH.
//! Hive only manages worker processes (start/stop/list/get).
//! install/uninstall modules are API stubs for consistency.

pub mod start;      // Process management
pub mod stop;       // Process management  
pub mod list;       // Process queries
pub mod get;        // Process queries
pub mod install;    // API stub - delegates to worker-catalog
pub mod uninstall;  // API stub - delegates to worker-catalog
```

### Step 4.3: Update Hive job_router.rs

**File:** `bin/20_rbee_hive/src/job_router.rs`

Remove these match arms:
```rust
Operation::WorkerDownload { .. } => { ... }     // DELETE
Operation::WorkerBuild { .. } => { ... }        // DELETE
Operation::WorkerBinaryDelete { .. } => { ... } // DELETE
```

Keep these:
```rust
Operation::WorkerSpawn { .. } => { ... }        // KEEP
Operation::WorkerProcessList { .. } => { ... }  // KEEP
```

### Step 4.4: Make Worker Catalog Read-Only

**File:** `bin/20_rbee_hive/src/worker_catalog.rs`

Hive scans `~/.local/share/rbee/workers/` but never writes to it.

Update doc comments to clarify:
```rust
//! Worker catalog - READ ONLY
//!
//! TEAM-277: Hive discovers workers installed by queen
//! Hive never installs workers itself
```

### Step 4.5: Verify

```bash
cargo check -p rbee-hive
cargo check -p worker-lifecycle
```

✅ **Phase 4 complete when hive compiles without worker install**

---

## Phase 5: Update CLI (8-12 hours)

### Goal
Add package manager commands to rbee-keeper CLI.

### Step 5.1: Create Command Files

```bash
touch bin/00_rbee_keeper/src/commands/sync.rs
touch bin/00_rbee_keeper/src/commands/package_status.rs
touch bin/00_rbee_keeper/src/commands/validate.rs
touch bin/00_rbee_keeper/src/commands/migrate.rs
```

### Step 5.2: Implement sync.rs

**File:** `bin/00_rbee_keeper/src/commands/sync.rs`

```rust
//! rbee sync command
//! TEAM-277

use anyhow::Result;
use rbee_operations::Operation;
use crate::job_client::submit_and_stream_job;
use crate::config::Config;

pub async fn sync(
    dry_run: bool,
    remove_extra: bool,
    force: bool,
    hive_alias: Option<String>,
) -> Result<()> {
    let config = Config::load()?;
    let queen_url = config.queen_url();
    
    let operation = Operation::PackageSync {
        config_path: None,
        dry_run,
        remove_extra,
        force,
    };
    
    submit_and_stream_job(&queen_url, operation).await?;
    
    Ok(())
}
```

**Reference:** See `bin/00_rbee_keeper/src/commands/` for existing command patterns.

### Step 5.3: Update main.rs

**File:** `bin/00_rbee_keeper/src/main.rs`

Add subcommands:
```rust
#[derive(Subcommand)]
enum Commands {
    // ... existing commands ...
    
    /// Sync all hives to match config
    Sync {
        #[arg(long)]
        dry_run: bool,
        
        #[arg(long)]
        remove_extra: bool,
        
        #[arg(long)]
        force: bool,
        
        #[arg(long)]
        hive: Option<String>,
    },
    
    /// Check package status
    Status {
        #[arg(long)]
        verbose: bool,
    },
    
    /// Validate config
    Validate {
        #[arg(long)]
        config: Option<String>,
    },
    
    /// Generate config from current state
    Migrate {
        #[arg(long)]
        output: String,
    },
}
```

Add handlers:
```rust
match cli.command {
    Commands::Sync { dry_run, remove_extra, force, hive } => {
        commands::sync::sync(dry_run, remove_extra, force, hive).await?
    }
    
    Commands::Status { verbose } => {
        commands::package_status::status(verbose).await?
    }
    
    // ... more handlers
}
```

### Step 5.4: Verify

```bash
cargo check -p rbee-keeper
cargo build -p rbee-keeper

# Test commands
./target/debug/rbee sync --dry-run
./target/debug/rbee status
```

✅ **Phase 5 complete when CLI commands work**

---

## Phase 6: Remove Old Operations (4-6 hours)

### Goal
AGGRESSIVELY remove deprecated operations. No backwards compatibility.

**v0.1.0 = DELETE EVERYTHING OLD!**
- Delete old operations
- Delete old CLI commands
- Delete old handlers
- No shims, no compatibility layers
- Clean slate

### Step 6.1: AGGRESSIVELY Remove from Operation enum

**File:** `bin/99_shared_crates/rbee-operations/src/lib.rs`

**DELETE WITHOUT MERCY - v0.1.0 allows breaking changes!**

Delete these variants:
```rust
Operation::HiveInstall { .. }        // DELETE
Operation::HiveUninstall { .. }      // DELETE
Operation::WorkerDownload { .. }     // DELETE
Operation::WorkerBuild { .. }        // DELETE
Operation::WorkerBinaryList { .. }   // DELETE
Operation::WorkerBinaryGet { .. }    // DELETE
Operation::WorkerBinaryDelete { .. } // DELETE
```

### Step 6.2: Remove from Operation::name()

Delete corresponding cases in `name()` method.

### Step 6.3: Remove Old CLI Commands

**File:** `bin/00_rbee_keeper/src/main.rs`

Delete:
```rust
Commands::InstallHive { .. }    // DELETE
Commands::UninstallHive { .. }  // DELETE
Commands::InstallWorker { .. }  // DELETE
```

### Step 6.4: Update Documentation

Update `bin/ADDING_NEW_OPERATIONS.md` to reflect new operations.

### Step 6.5: Verify

```bash
cargo check --workspace
cargo test --workspace
```

✅ **Phase 6 complete when old operations removed**

---

## Final Verification

### Test End-to-End

```bash
# 1. Create config
cat > ~/.config/rbee/hives.conf << 'EOF'
[[hive]]
alias = "test-hive"
hostname = "localhost"
ssh_user = "vince"
workers = [
    { type = "vllm", version = "latest" },
]
EOF

# 2. Validate
rbee validate

# 3. Dry run
rbee sync --dry-run

# 4. Install
rbee sync

# 5. Check status
rbee status
```

### Success Criteria

✅ All phases complete  
✅ All tests pass  
✅ `rbee sync` works  
✅ Concurrent installation works  
✅ Old operations removed  

---

## Summary

**You've completed:**
1. ✅ Config support
2. ✅ Package operations
3. ✅ Package manager in queen
4. ✅ Simplified hive
5. ✅ Updated CLI
6. ✅ Removed old operations

**Result:** Declarative lifecycle management! 🎉

**Handoff:** Write summary in `bin/.plan/TEAM_277_HANDOFF.md`
