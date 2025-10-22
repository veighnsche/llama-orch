# TEAM-215: Phase 6 - Integration

**Assigned to:** TEAM-215  
**Depends on:** TEAM-210, TEAM-211, TEAM-212, TEAM-213, TEAM-214  
**Blocks:** TEAM-209 (Peer Review)  
**Estimated LOC:** ~50 lines added, ~724 lines removed

---

## Mission

Wire up `job_router.rs` to use the new `hive-lifecycle` crate:
- Replace inline implementations with crate function calls
- Remove old code (~724 LOC)
- Update imports
- Ensure job_id propagation for SSE routing

---

## Current State

**File:** `bin/10_queen_rbee/src/job_router.rs` (1,115 LOC)

Contains:
- Job routing logic (lines 1-251, 1012-1115) - ~350 LOC
- Hive operations (lines 252-1011) - ~760 LOC (including validation helper)

---

## Target State

**File:** `bin/10_queen_rbee/src/job_router.rs` (~350 LOC)

Contains:
- Job routing logic only
- Thin wrappers that call hive-lifecycle functions
- Proper error handling and job_id propagation

---

## Deliverables

### 1. Update Cargo.toml

**File:** `bin/10_queen_rbee/Cargo.toml`

Add dependency:
```toml
[dependencies]
queen-rbee-hive-lifecycle = { path = "../15_queen_rbee_crates/hive-lifecycle" }
```

### 2. Update Imports in job_router.rs

**File:** `bin/10_queen_rbee/src/job_router.rs`

Replace:
```rust
use queen_rbee_hive_lifecycle::{execute_ssh_test, SshTestRequest};
```

With:
```rust
use queen_rbee_hive_lifecycle::{
    execute_ssh_test, SshTestRequest,
    execute_hive_install, HiveInstallRequest,
    execute_hive_uninstall, HiveUninstallRequest,
    execute_hive_start, HiveStartRequest,
    execute_hive_stop, HiveStopRequest,
    execute_hive_list, HiveListRequest,
    execute_hive_get, HiveGetRequest,
    execute_hive_status, HiveStatusRequest,
    execute_hive_refresh_capabilities, HiveRefreshCapabilitiesRequest,
};
```

### 3. Remove Old Code

**Delete these sections from job_router.rs:**
- Lines 98-160: `validate_hive_exists()` function (moved to hive-lifecycle)
- Lines 280-1011: All hive operation implementations (moved to hive-lifecycle)

### 4. Replace with Thin Wrappers

**File:** `bin/10_queen_rbee/src/job_router.rs`

Replace each operation with a thin wrapper:

```rust
// TEAM-215: Migrated hive operations to hive-lifecycle crate

match operation {
    // System-wide operations
    Operation::Status => {
        // Keep existing Status implementation (not part of hive-lifecycle)
        // ... existing code ...
    }

    // Hive operations (delegated to hive-lifecycle crate)
    Operation::SshTest { alias } => {
        // TEAM-215: Delegate to hive-lifecycle crate
        let hive_config = state.config.hives.get(&alias).ok_or_else(|| {
            anyhow::anyhow!("Hive '{}' not found in config", alias)
        })?;

        let request = SshTestRequest {
            ssh_host: hive_config.hostname.clone(),
            ssh_port: hive_config.ssh_port,
            ssh_user: hive_config.ssh_user.clone(),
        };

        let response = execute_ssh_test(request).await?;

        if !response.success {
            return Err(anyhow::anyhow!(
                "SSH connection failed: {}",
                response.error.unwrap_or_else(|| "Unknown error".to_string())
            ));
        }

        NARRATE
            .action("ssh_test_ok")
            .job_id(&job_id)
            .context(response.test_output.unwrap_or_default())
            .human("✅ SSH test successful: {}")
            .emit();
    }

    Operation::HiveInstall { alias } => {
        // TEAM-215: Delegate to hive-lifecycle crate
        let request = HiveInstallRequest { alias };
        execute_hive_install(request, state.config.clone(), &job_id).await?;
    }

    Operation::HiveUninstall { alias } => {
        // TEAM-215: Delegate to hive-lifecycle crate
        let request = HiveUninstallRequest { alias };
        execute_hive_uninstall(request, state.config.clone(), &job_id).await?;
    }

    Operation::HiveStart { alias } => {
        // TEAM-215: Delegate to hive-lifecycle crate
        let request = HiveStartRequest {
            alias,
            job_id: job_id.clone(),
        };
        execute_hive_start(request, state.config.clone()).await?;
    }

    Operation::HiveStop { alias } => {
        // TEAM-215: Delegate to hive-lifecycle crate
        let request = HiveStopRequest {
            alias,
            job_id: job_id.clone(),
        };
        execute_hive_stop(request, state.config.clone()).await?;
    }

    Operation::HiveList => {
        // TEAM-215: Delegate to hive-lifecycle crate
        let request = HiveListRequest {};
        execute_hive_list(request, state.config.clone(), &job_id).await?;
    }

    Operation::HiveGet { alias } => {
        // TEAM-215: Delegate to hive-lifecycle crate
        let request = HiveGetRequest { alias };
        execute_hive_get(request, state.config.clone(), &job_id).await?;
    }

    Operation::HiveStatus { alias } => {
        // TEAM-215: Delegate to hive-lifecycle crate
        let request = HiveStatusRequest {
            alias,
            job_id: job_id.clone(),
        };
        execute_hive_status(request, state.config.clone(), &job_id).await?;
    }

    Operation::HiveRefreshCapabilities { alias } => {
        // TEAM-215: Delegate to hive-lifecycle crate
        let request = HiveRefreshCapabilitiesRequest {
            alias,
            job_id: job_id.clone(),
        };
        execute_hive_refresh_capabilities(request, state.config.clone()).await?;
    }

    // Worker operations (keep existing TODOs)
    Operation::WorkerSpawn { .. } => {
        // ... existing TODO ...
    }
    // ... rest of operations ...
}
```

### 5. Remove hive_client Module (if applicable)

If `hive_client` module only contains `check_hive_health()` and `fetch_hive_capabilities()`, and these are now in `hive-lifecycle` crate, remove the module:

```rust
// TEAM-215: Remove old hive_client module (moved to hive-lifecycle)
// mod hive_client;  // DELETE THIS LINE
```

---

## Verification Checklist

### Compilation
- [ ] `cargo check -p queen-rbee` succeeds
- [ ] `cargo build -p queen-rbee` succeeds
- [ ] No unused import warnings
- [ ] No dead code warnings

### Functionality
- [ ] `./rbee hive list` works
- [ ] `./rbee hive install` works
- [ ] `./rbee hive start` works
- [ ] `./rbee hive stop` works
- [ ] `./rbee hive status` works
- [ ] `./rbee hive refresh` works
- [ ] `./rbee hive get <alias>` works
- [ ] `./rbee ssh-test <alias>` works

### SSE Routing
- [ ] All narration events appear in SSE stream
- [ ] No events lost (check for job_id in all narration)
- [ ] Timeout countdown visible in SSE stream

### Error Messages
- [ ] Error messages match original exactly
- [ ] Helpful error messages for missing hives
- [ ] Helpful error messages for missing binaries

### LOC Reduction
- [ ] job_router.rs reduced from 1,115 LOC to ~350 LOC
- [ ] ~65% reduction achieved
- [ ] Code is more maintainable

---

## Testing Commands

```bash
# Build everything
cargo build --bin rbee-keeper --bin queen-rbee --bin rbee-hive

# Test hive operations
./rbee hive list
./rbee hive install
./rbee hive start
./rbee hive status
./rbee hive refresh
./rbee hive stop

# Test with SSE stream (check for narration)
./rbee hive start --verbose

# Check LOC reduction
wc -l bin/10_queen_rbee/src/job_router.rs
```

---

## Acceptance Criteria

- [ ] Cargo.toml updated with hive-lifecycle dependency
- [ ] All imports updated
- [ ] Old code removed (~724 LOC)
- [ ] Thin wrappers implemented (~50 LOC)
- [ ] All operations work identically
- [ ] SSE narration flows correctly
- [ ] Error messages preserved
- [ ] job_router.rs reduced to ~350 LOC
- [ ] No TODO markers in TEAM-215 code
- [ ] All code has TEAM-215 signatures

---

## Handoff to TEAM-209 (Peer Review)

**What's Ready:**
- ✅ All hive operations migrated to hive-lifecycle crate
- ✅ job_router.rs reduced by ~65%
- ✅ Clean separation of concerns
- ✅ All functionality preserved

**Next Steps for TEAM-209:**
1. Read `07_PHASE_7_PEER_REVIEW.md`
2. Perform critical peer review
3. Verify all acceptance criteria
4. Test all operations
5. Check SSE routing
6. Validate error messages

---

## Notes

- Preserve exact error messages from original
- Ensure job_id propagation for SSE routing
- Test all operations thoroughly
- Verify LOC reduction achieved
- Check for any unused imports or dead code
