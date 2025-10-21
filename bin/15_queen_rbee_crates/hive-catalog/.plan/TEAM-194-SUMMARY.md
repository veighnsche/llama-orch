# TEAM-194 SUMMARY: Phase 2 - Replace SQLite in job_router.rs

**Team:** TEAM-194  
**Duration:** 6-8 hours (IN PROGRESS)  
**Mission:** Replace all SQLite-based hive catalog operations in `job_router.rs` with file-based config lookups using the new `rbee-config` crate.

---

## ✅ COMPLETED WORK

### 1. Dependencies Updated ✅
- **File:** `bin/10_queen_rbee/Cargo.toml`
- **Change:** Replaced `queen-rbee-hive-catalog` with `rbee-config`
- **Status:** COMPLETE

### 2. AppState Refactored ✅
- **File:** `bin/10_queen_rbee/src/main.rs`
- **Changes:**
  - Replaced `HiveCatalog` with `RbeeConfig`
  - Updated `create_router()` signature
  - Changed initialization to load from `~/.config/rbee/`
  - Added `--config-dir` CLI argument
- **Status:** COMPLETE

### 3. HTTP Module Updated ✅
- **File:** `bin/10_queen_rbee/src/http/jobs.rs`
- **Changes:**
  - Updated `SchedulerState` to use `config: Arc<RbeeConfig>`
  - Updated `From<SchedulerState>` impl for `JobState`
- **Status:** COMPLETE

### 4. JobState Refactored ✅
- **File:** `bin/10_queen_rbee/src/job_router.rs`
- **Changes:**
  - Updated `JobState` struct to use `config: Arc<RbeeConfig>`
  - Updated `route_operation()` signature
  - Updated `execute_job()` to pass config
  - Added `Narration` import
  - Added narration constants (`ACTOR_QUEEN_ROUTER`, `ACTION_ROUTE_JOB`)
- **Status:** COMPLETE

---

## 🚧 REMAINING WORK

### Critical: Operation Enum Must Be Updated FIRST

**Before refactoring handlers, the Operation enum MUST be simplified:**

**File:** `bin/99_shared_crates/rbee-operations/src/lib.rs`

**Current (WRONG):**
```rust
Operation::HiveInstall {
    hive_id: String,
    ssh_host: Option<String>,
    ssh_port: Option<u16>,
    ssh_user: Option<String>,
    port: u16,
    binary_path: Option<String>,
}

Operation::SshTest {
    ssh_host: String,
    ssh_port: u16,
    ssh_user: String,
}
```

**Target (CORRECT):**
```rust
Operation::HiveInstall {
    alias: String, // Must exist in hives.conf
}

Operation::HiveUninstall {
    alias: String,
}

Operation::HiveStart {
    alias: String,
}

Operation::HiveStop {
    alias: String,
}

Operation::HiveList {}

Operation::SshTest {
    alias: String, // Test SSH using config from hives.conf
}
```

**Why this matters:** The Operation enum defines the contract between rbee-keeper (CLI) and queen-rbee (server). It must be updated BEFORE refactoring handlers, otherwise the handlers will fail to compile.

---

### Handler Refactoring (BLOCKED until Operation enum updated)

Once Operation enum is updated, these handlers need refactoring:

#### 1. HiveInstall Handler
**Current:** Lines 205-371 in `job_router.rs`  
**Needs:**
- Remove `use queen_rbee_hive_catalog::HiveRecord;` (line 207)
- Replace `state.hive_catalog.hive_exists()` with `state.config.capabilities.get(alias)`
- Remove SQLite registration (lines 328-353)
- Follow spec from PHASE_2_TEAM_189.md lines 193-336

#### 2. HiveUninstall Handler
**Current:** Lines 372-478 in `job_router.rs`  
**Needs:**
- Replace `state.hive_catalog.get_hive()` with `state.config.hives.get(alias)`
- Remove `state.hive_catalog.remove_hive()` call
- Follow spec from PHASE_2_TEAM_189.md lines 338-380

#### 3. HiveList Handler
**Current:** Lines 697-737 in `job_router.rs`  
**Needs:**
- Replace `state.hive_catalog.list_hives()` with `state.config.hives.all()`
- Check `state.config.capabilities.get(alias)` for running status
- Follow spec from PHASE_2_TEAM_189.md lines 382-424

#### 4. SshTest Handler
**Current:** Lines 181-204 in `job_router.rs`  
**Needs:**
- Replace direct SSH parameters with alias lookup
- Get SSH details from `state.config.hives.get(alias)`
- Follow spec from PHASE_2_TEAM_189.md lines 426-464

#### 5. HiveStart Handler
**Current:** Lines 513-593 in `job_router.rs`  
**Needs:**
- Replace `state.hive_catalog.get_hive()` with `state.config.hives.get(alias)`
- Check `state.config.capabilities.get(alias)` for already-running check

#### 6. HiveStop Handler
**Current:** Lines 596-696 in `job_router.rs`  
**Needs:**
- Replace `state.hive_catalog.get_hive()` with `state.config.hives.get(alias)`

#### 7. HiveStatus Handler
**Current:** Lines 747-840 in `job_router.rs`  
**Needs:**
- Replace `state.hive_catalog.get_hive()` with `state.config.hives.get(alias)`

---

### CLI Argument Parsing (rbee-keeper)

**File:** `bin/00_rbee_keeper/src/main.rs`

**Current (lines 155-246):**
```rust
HiveAction::Install {
    id: String,
    ssh_host: Option<String>,
    ssh_port: Option<u16>,
    ssh_user: Option<String>,
    port: u16,
    binary_path: Option<String>,
}

HiveAction::SshTest {
    ssh_host: String,
    ssh_port: u16,
    ssh_user: String,
}
```

**Target:**
```rust
HiveAction::Install {
    #[arg(short = 'h', long = "host")]
    alias: String,
}

HiveAction::Uninstall {
    #[arg(short = 'h', long = "host")]
    alias: String,
}

HiveAction::Start {
    #[arg(short = 'h', long = "host")]
    alias: String,
}

HiveAction::Stop {
    #[arg(short = 'h', long = "host")]
    alias: String,
}

HiveAction::SshTest {
    #[arg(short = 'h', long = "host")]
    alias: String,
}
```

**Also update:** Lines 350-490 in `main.rs` where these actions are converted to Operations

---

## 📋 EXECUTION PLAN FOR NEXT TEAM

### Step 1: Update Operation Enum (CRITICAL - DO THIS FIRST)
1. Edit `bin/99_shared_crates/rbee-operations/src/lib.rs`
2. Simplify all hive operations to use `alias: String` only
3. Update tests at bottom of file
4. Run `cargo check --package rbee-operations`

### Step 2: Update CLI Arguments
1. Edit `bin/00_rbee_keeper/src/main.rs`
2. Update `HiveAction` enum (lines 155-246)
3. Update action-to-operation conversion (lines 350-490)
4. Run `cargo check --bin rbee-keeper`

### Step 3: Refactor Handlers (One at a Time)
1. HiveInstall (most complex - ~150 lines)
2. HiveUninstall (~40 lines)
3. HiveList (~30 lines)
4. SshTest (~20 lines)
5. HiveStart (~80 lines)
6. HiveStop (~100 lines)
7. HiveStatus (~90 lines)

### Step 4: Verification
```bash
# Build
cargo build --bin queen-rbee
cargo build --bin rbee-keeper

# Check
cargo clippy --bin queen-rbee
cargo clippy --bin rbee-keeper

# Test (if tests exist)
cargo test --bin queen-rbee
```

---

## 🎯 ACCEPTANCE CRITERIA

- [ ] All SQLite calls removed from `job_router.rs`
- [ ] All hive operations use `state.config.hives.get(alias)`
- [ ] CLI uses `-h <alias>` instead of `--id <id> --ssh-host ...`
- [ ] Operation enum simplified (only alias field)
- [ ] Code compiles without errors
- [ ] Narration messages updated with new flow
- [ ] Error messages guide users to edit `hives.conf`

---

## 📊 PROGRESS

**Completed:** 4/9 tasks (44%)  
**Remaining:** 5/9 tasks (56%)  
**Blockers:** Operation enum must be updated before handler refactoring can proceed

---

## 🔗 REFERENCES

- **Phase 2 Spec:** `bin/15_queen_rbee_crates/hive-catalog/.plan/PHASE_2_TEAM_189.md`
- **rbee-config API:** `bin/15_queen_rbee_crates/rbee-config/src/lib.rs`
- **HivesConfig API:** `bin/15_queen_rbee_crates/rbee-config/src/hives_config.rs`
- **Engineering Rules:** `.windsurf/rules/engineering-rules.md`

---

**Created by:** TEAM-194  
**Status:** 🚧 IN PROGRESS - 44% complete  
**Next Team:** TEAM-195 (or continue as TEAM-194)  
**Estimated Remaining Time:** 3-4 hours
