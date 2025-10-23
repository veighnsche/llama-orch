# TEAM-277 Declarative Lifecycle Migration - Complete Checklist

**Mission:** Migrate from imperative to declarative lifecycle management  
**Estimated Effort:** 64-90 hours (2-3 weeks)  
**Status:** ⏳ NOT STARTED

## 🔥 v0.1.0 = BREAK EVERYTHING!

**This is v0.1.0 - breaking changes are REQUIRED.**

- ✅ Delete old code aggressively
- ✅ Remove backwards compatibility
- ✅ No shims, no compatibility layers
- ❌ Don't be "careful" - be DESTRUCTIVE
- ❌ Don't preserve old patterns

---

## 📋 How to Use This Checklist

1. Check off items as you complete them: `- [ ]` → `- [x]`
2. Update status markers: `⏳ TODO` → `🔄 IN PROGRESS` → `✅ DONE`
3. Track blockers in the "Blockers" section
4. Update cumulative progress at the end

---

## Phase 1: Add Config Support (8-12 hours) ✅ DONE

**Goal:** Add `hives.conf` parsing and REPLACE old config patterns

**v0.1.0 = BREAK THINGS!** Delete old code, no backwards compatibility.

### Step 1.1: Create declarative.rs
- [x] Create file `bin/99_shared_crates/rbee-config/src/declarative.rs`
- [x] Add `HivesConfig` struct (top-level config)
- [x] Add `HiveConfig` struct (single hive config)
- [x] Add `WorkerConfig` struct (worker config)
- [x] Implement `HivesConfig::load()` - Load from `~/.config/rbee/hives.conf`
- [x] Implement `HivesConfig::load_from(path)` - Load from custom path
- [x] Implement `HivesConfig::validate()` - Validate config
- [x] Add error handling for missing/invalid config

### Step 1.2: Export from lib.rs
- [x] Open `bin/99_shared_crates/rbee-config/src/lib.rs`
- [x] Add `pub mod declarative;`
- [x] Add `pub use declarative::{HivesConfig, HiveConfig, WorkerConfig};`

### Step 1.3: Add dependencies
- [x] Open `bin/99_shared_crates/rbee-config/Cargo.toml`
- [x] Add `toml = "0.8"` to `[dependencies]` (already present)
- [x] Add `dirs = "5.0"` to `[dependencies]`

### Step 1.4: Test
- [x] Create test config at `~/.config/rbee/hives.conf`
- [x] Add test hive with localhost + workers
- [x] Run `cargo check -p rbee-config`
- [x] Run `cargo test -p rbee-config`
- [x] Verify config loads correctly

**✅ Phase 1 Complete When:**
- ✅ All tests pass (8/8 declarative tests passing)
- ✅ Config parsing works
- ✅ No compilation errors

---

## Phase 2: Add Package Operations (12-16 hours) ⏳

**Goal:** Add new operations to Operation enum

### Step 2.1: Add to Operation enum
- [ ] Open `bin/99_shared_crates/rbee-operations/src/lib.rs`
- [ ] Find Operation enum (around line 54)
- [ ] Add `PackageSync` variant with fields:
  - [ ] `config_path: Option<String>`
  - [ ] `dry_run: bool`
  - [ ] `remove_extra: bool`
  - [ ] `force: bool`
- [ ] Add `PackageStatus` variant with fields:
  - [ ] `config_path: Option<String>`
  - [ ] `verbose: bool`
- [ ] Add `PackageInstall` variant with fields:
  - [ ] `config_path: Option<String>`
  - [ ] `force: bool`
- [ ] Add `PackageUninstall` variant with fields:
  - [ ] `config_path: Option<String>`
  - [ ] `purge: bool`
- [ ] Add `PackageValidate` variant with field:
  - [ ] `config_path: Option<String>`
- [ ] Add `PackageMigrate` variant with field:
  - [ ] `output_path: String`

### Step 2.2: Add to Operation::name()
- [ ] Find `Operation::name()` method (around line 148)
- [ ] Add `Operation::PackageSync { .. } => "package_sync"`
- [ ] Add `Operation::PackageStatus { .. } => "package_status"`
- [ ] Add `Operation::PackageInstall { .. } => "package_install"`
- [ ] Add `Operation::PackageUninstall { .. } => "package_uninstall"`
- [ ] Add `Operation::PackageValidate { .. } => "package_validate"`
- [ ] Add `Operation::PackageMigrate { .. } => "package_migrate"`

### Step 2.3: Update should_forward_to_hive()
- [ ] Find `should_forward_to_hive()` method (around line 305)
- [ ] Update doc comment to clarify:
  - [ ] Package operations handled by queen (orchestration)
  - [ ] Worker/Model operations forwarded to hive (execution)
- [ ] Verify package operations are NOT in the matches! list

### Step 2.4: Verify
- [ ] Run `cargo check -p rbee-operations`
- [ ] Run `cargo test -p rbee-operations`
- [ ] Verify all operations compile

**✅ Phase 2 Complete When:**
- All operations compile
- Tests pass
- Documentation updated

---

## Phase 3: Package Manager in Queen (24-32 hours) ⏳

**Goal:** Implement package manager logic in queen-rbee

### Step 3.1: Create Module Structure
- [ ] Create directory `bin/10_queen_rbee/src/package_manager`
- [ ] Create `bin/10_queen_rbee/src/package_manager/mod.rs`
- [ ] Create `bin/10_queen_rbee/src/package_manager/sync.rs`
- [ ] Create `bin/10_queen_rbee/src/package_manager/diff.rs`
- [ ] Create `bin/10_queen_rbee/src/package_manager/install.rs`
- [ ] Create `bin/10_queen_rbee/src/package_manager/status.rs`
- [ ] Create `bin/10_queen_rbee/src/package_manager/validate.rs`
- [ ] Create `bin/10_queen_rbee/src/package_manager/migrate.rs`

### Step 3.2: Implement mod.rs
- [ ] Add module declarations for all 6 modules
- [ ] Re-export main functions:
  - [ ] `sync_all_hives`, `sync_single_hive`
  - [ ] `check_package_status`
  - [ ] `install_all`, `install_hive_binary`, `install_worker_binary`

### Step 3.3: Implement install.rs (Core Logic)
- [ ] Add import: `use queen_rbee_hive_lifecycle::ssh_helper::SshClient;`
- [ ] Implement `install_hive_binary()`:
  - [ ] Connect via SSH
  - [ ] Download hive binary
  - [ ] Set permissions
  - [ ] Verify installation
- [ ] Implement `install_worker_binary()` (NEW!):
  - [ ] Connect via SSH (reuse SshClient)
  - [ ] Download worker to `~/.local/share/rbee/workers/`
  - [ ] Set permissions
  - [ ] Verify installation
- [ ] Implement `install_all()`:
  - [ ] Use `tokio::spawn` for concurrent installation
  - [ ] Install hive first, then workers
  - [ ] Collect results
- [ ] Add narration with `.job_id()` for SSE routing

### Step 3.4: Implement sync.rs (Orchestration)
- [ ] Implement `sync_all_hives()`:
  - [ ] Query actual state
  - [ ] Compute diff (desired vs actual)
  - [ ] If dry_run, return early with report
  - [ ] Apply changes concurrently using `tokio::spawn`
  - [ ] Return sync report
- [ ] Implement `sync_single_hive()`:
  - [ ] Install hive if needed
  - [ ] Install workers if needed
  - [ ] Start hive if `auto_start` enabled
  - [ ] Return result

### Step 3.5: Implement diff.rs (State Comparison)
- [ ] Create `StateDiff` struct with fields:
  - [ ] `hives_to_install: Vec<HiveConfig>`
  - [ ] `hives_already_installed: Vec<HiveConfig>`
  - [ ] `workers_to_install: Vec<(String, Vec<String>)>`
  - [ ] `workers_already_installed: Vec<(String, Vec<String>)>`
  - [ ] `hives_to_remove: Vec<String>` (if `remove_extra`)
- [ ] Implement `compute_diff()`:
  - [ ] Compare desired (config) vs actual (installed)
  - [ ] Return StateDiff

### Step 3.6: Wire into job_router.rs
- [ ] Open `bin/10_queen_rbee/src/job_router.rs`
- [ ] Add import: `use rbee_config::declarative::HivesConfig;`
- [ ] Add match arm for `PackageSync`:
  - [ ] Load config with `HivesConfig::load()` or `load_from()`
  - [ ] Create `SyncOptions` struct
  - [ ] Call `package_manager::sync_all_hives()`
- [ ] Add match arm for `PackageStatus`:
  - [ ] Load config
  - [ ] Call `package_manager::check_package_status()`
- [ ] Add match arms for other package operations:
  - [ ] `PackageInstall`
  - [ ] `PackageUninstall`
  - [ ] `PackageValidate`
  - [ ] `PackageMigrate`

### Step 3.7: Add to lib.rs
- [ ] Open `bin/10_queen_rbee/src/lib.rs`
- [ ] Add `pub mod package_manager;`

### Step 3.8: Verify
- [ ] Run `cargo check -p queen-rbee`
- [ ] Fix any compilation errors
- [ ] Test sync operation manually

**✅ Phase 3 Complete When:**
- Sync works end-to-end
- Concurrent installation works
- All package operations compile

---

## Phase 4: Simplify Hive (8-12 hours) ⏳

**Goal:** Remove worker installation logic from rbee-hive

### Step 4.1: Worker Installation is Already Stubbed
- [ ] Review `bin/25_rbee_hive_crates/worker-lifecycle/src/install.rs`
- [ ] Verify it's a stub that delegates to worker-catalog
- [ ] Update doc comment to mention TEAM-277 architecture change
- [ ] Review `bin/25_rbee_hive_crates/worker-lifecycle/src/uninstall.rs`
- [ ] Update doc comment similarly

### Step 4.2: Update worker-lifecycle Documentation
- [ ] Open `bin/25_rbee_hive_crates/worker-lifecycle/src/lib.rs`
- [ ] Add TEAM-277 architecture update doc comment
- [ ] Clarify that install/uninstall are API stubs
- [ ] Clarify that queen handles installation via SSH

### Step 4.3: Update Hive job_router.rs
- [ ] Open `bin/20_rbee_hive/src/job_router.rs`
- [ ] Remove match arm for `WorkerDownload` (if exists)
- [ ] Remove match arm for `WorkerBuild` (if exists)
- [ ] Remove match arm for `WorkerBinaryDelete` (if exists)
- [ ] Keep match arms for:
  - [ ] `WorkerSpawn` (process management)
  - [ ] `WorkerProcessList` (process queries)
  - [ ] `WorkerProcessGet` (process queries)
  - [ ] `WorkerProcessDelete` (process management)

### Step 4.4: Make Worker Catalog Read-Only
- [ ] Open `bin/20_rbee_hive/src/worker_catalog.rs`
- [ ] Update doc comment to clarify:
  - [ ] Hive discovers workers installed by queen
  - [ ] Hive never installs workers itself
  - [ ] Catalog is READ ONLY

### Step 4.5: Verify
- [ ] Run `cargo check -p rbee-hive`
- [ ] Run `cargo check -p worker-lifecycle`
- [ ] Verify hive compiles without worker install logic

**✅ Phase 4 Complete When:**
- Hive compiles successfully
- Worker catalog is read-only
- Documentation updated

---

## Phase 5: Update CLI (8-12 hours) ⏳

**Goal:** Add package manager commands to rbee-keeper CLI

### Step 5.1: Create Command Files
- [ ] Create `bin/00_rbee_keeper/src/handlers/sync.rs`
- [ ] Create `bin/00_rbee_keeper/src/handlers/package_status.rs`
- [ ] Create `bin/00_rbee_keeper/src/handlers/validate.rs`
- [ ] Create `bin/00_rbee_keeper/src/handlers/migrate.rs`

### Step 5.2: Implement sync.rs
- [ ] Add imports:
  - [ ] `use crate::job_client::submit_and_stream_job;`
  - [ ] `use crate::config::Config;`
  - [ ] `use rbee_operations::Operation;`
- [ ] Implement `sync()` function:
  - [ ] Load config with `Config::load()`
  - [ ] Get queen_url
  - [ ] Create `Operation::PackageSync`
  - [ ] Call `submit_and_stream_job()`

### Step 5.3: Implement other handlers
- [ ] Implement `package_status.rs` (similar pattern)
- [ ] Implement `validate.rs` (similar pattern)
- [ ] Implement `migrate.rs` (similar pattern)

### Step 5.4: Update CLI enum
- [ ] Open `bin/00_rbee_keeper/src/cli/mod.rs` (or wherever Commands enum is)
- [ ] Add `Sync` command with args:
  - [ ] `dry_run: bool`
  - [ ] `remove_extra: bool`
  - [ ] `force: bool`
  - [ ] `hive: Option<String>`
- [ ] Add `Status` command with args:
  - [ ] `verbose: bool`
- [ ] Add `Validate` command with args:
  - [ ] `config: Option<String>`
- [ ] Add `Migrate` command with args:
  - [ ] `output: String`

### Step 5.5: Update main.rs
- [ ] Open `bin/00_rbee_keeper/src/main.rs`
- [ ] Add handler imports
- [ ] Add match arms for new commands:
  - [ ] `Commands::Sync { .. } => handlers::sync::sync(...).await?`
  - [ ] `Commands::Status { .. } => handlers::package_status::status(...).await?`
  - [ ] `Commands::Validate { .. } => handlers::validate::validate(...).await?`
  - [ ] `Commands::Migrate { .. } => handlers::migrate::migrate(...).await?`

### Step 5.6: Verify
- [ ] Run `cargo check -p rbee-keeper`
- [ ] Run `cargo build -p rbee-keeper`
- [ ] Test commands:
  - [ ] `./target/debug/rbee sync --dry-run`
  - [ ] `./target/debug/rbee status`
  - [ ] `./target/debug/rbee validate`

**✅ Phase 5 Complete When:**
- All CLI commands work
- Help text is correct
- Commands submit to queen successfully

---

## Phase 6: Remove Old Operations (4-6 hours) ⏳

**Goal:** AGGRESSIVELY remove deprecated operations

**v0.1.0 = DELETE EVERYTHING OLD!** No shims, no compatibility, clean slate.

### Step 6.1: Remove from Operation enum
- [ ] Open `bin/99_shared_crates/rbee-operations/src/lib.rs`
- [ ] Delete `HiveInstall` variant
- [ ] Delete `HiveUninstall` variant
- [ ] Delete `WorkerDownload` variant (if exists)
- [ ] Delete `WorkerBuild` variant (if exists)
- [ ] Delete `WorkerBinaryList` variant (if exists)
- [ ] Delete `WorkerBinaryGet` variant (if exists)
- [ ] Delete `WorkerBinaryDelete` variant (if exists)

### Step 6.2: Remove from Operation::name()
- [ ] Delete corresponding cases in `name()` method
- [ ] Delete from constants module (if present)

### Step 6.3: Remove Old CLI Commands
- [ ] Open `bin/00_rbee_keeper/src/cli/mod.rs`
- [ ] Delete `InstallHive` command (if exists)
- [ ] Delete `UninstallHive` command (if exists)
- [ ] Delete `InstallWorker` command (if exists)
- [ ] Open `bin/00_rbee_keeper/src/main.rs`
- [ ] Remove corresponding match arms

### Step 6.4: Remove from job_router.rs
- [ ] Open `bin/10_queen_rbee/src/job_router.rs`
- [ ] Remove match arms for deleted operations
- [ ] Clean up imports

### Step 6.5: Update Documentation
- [ ] Update `bin/ADDING_NEW_OPERATIONS.md`
- [ ] Mention package operations
- [ ] Update examples

### Step 6.6: Verify
- [ ] Run `cargo check --workspace`
- [ ] Run `cargo test --workspace`
- [ ] Fix any compilation errors
- [ ] Verify old operations are gone

**✅ Phase 6 Complete When:**
- Old operations removed
- All tests pass
- Documentation updated

---

## Final Verification ⏳

### End-to-End Test
- [ ] Create test config at `~/.config/rbee/hives.conf`:
  ```toml
  [[hive]]
  alias = "test-hive"
  hostname = "localhost"
  ssh_user = "vince"
  workers = [
      { type = "vllm", version = "latest" },
  ]
  ```
- [ ] Run `rbee validate`
- [ ] Run `rbee sync --dry-run`
- [ ] Run `rbee sync`
- [ ] Run `rbee status`
- [ ] Verify hive + workers installed concurrently

### Success Criteria
- [ ] All phases complete
- [ ] All tests pass
- [ ] `rbee sync` works end-to-end
- [ ] Concurrent installation works (3-10x faster)
- [ ] Config file is source of truth
- [ ] Old operations removed
- [ ] Documentation complete

---

## Handoff ⏳

### Write Handoff Document
- [ ] Create `bin/.plan/TEAM_277_HANDOFF.md`
- [ ] Include what you built
- [ ] Include what works
- [ ] Include what's next
- [ ] Include code examples
- [ ] Include verification results
- [ ] Keep to 2 pages maximum

---

## Blockers & Issues

**Track blockers here:**
- None yet

---

## Cumulative Progress

| Phase | Status | LOC Added | LOC Removed | Duration |
|-------|--------|-----------|-------------|----------|
| Phase 1: Config | ⏳ TODO | 0 | 0 | 0h |
| Phase 2: Operations | ⏳ TODO | 0 | 0 | 0h |
| Phase 3: Package Manager | ⏳ TODO | 0 | 0 | 0h |
| Phase 4: Simplify Hive | ⏳ TODO | 0 | 0 | 0h |
| Phase 5: CLI | ⏳ TODO | 0 | 0 | 0h |
| Phase 6: Cleanup | ⏳ TODO | 0 | 0 | 0h |
| **Total** | **⏳ TODO** | **0** | **0** | **0h / 64-90h** |

---

## Quick Reference

### Key Files to Create
- `bin/99_shared_crates/rbee-config/src/declarative.rs`
- `bin/10_queen_rbee/src/package_manager/mod.rs`
- `bin/10_queen_rbee/src/package_manager/sync.rs`
- `bin/10_queen_rbee/src/package_manager/diff.rs`
- `bin/10_queen_rbee/src/package_manager/install.rs`
- `bin/10_queen_rbee/src/package_manager/status.rs`
- `bin/10_queen_rbee/src/package_manager/validate.rs`
- `bin/10_queen_rbee/src/package_manager/migrate.rs`
- `bin/00_rbee_keeper/src/handlers/sync.rs`
- `bin/00_rbee_keeper/src/handlers/package_status.rs`
- `bin/00_rbee_keeper/src/handlers/validate.rs`
- `bin/00_rbee_keeper/src/handlers/migrate.rs`

### Key Files to Modify
- `bin/99_shared_crates/rbee-config/src/lib.rs`
- `bin/99_shared_crates/rbee-config/Cargo.toml`
- `bin/99_shared_crates/rbee-operations/src/lib.rs`
- `bin/10_queen_rbee/src/job_router.rs`
- `bin/10_queen_rbee/src/lib.rs`
- `bin/20_rbee_hive/src/job_router.rs`
- `bin/25_rbee_hive_crates/worker-lifecycle/src/lib.rs`
- `bin/00_rbee_keeper/src/cli/mod.rs`
- `bin/00_rbee_keeper/src/main.rs`
- `bin/ADDING_NEW_OPERATIONS.md`

### Key Patterns to Follow
- Use `tokio::spawn` for concurrent operations
- Use `SshClient` from hive-lifecycle for SSH
- Use `submit_and_stream_job()` from job_client
- Add `.job_id()` to all narration for SSE routing
- Add TEAM-277 signatures to all code

---

**Remember:** This is a major architectural improvement. Take your time, test thoroughly, and document everything!
