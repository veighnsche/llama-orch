# TEAM-213: Phase 4 - Install/Uninstall

**Assigned to:** TEAM-213  
**Depends on:** TEAM-210 (Foundation)  
**Blocks:** TEAM-215 (Integration)  
**Estimated LOC:** ~220 lines

---

## Mission

Implement hive installation and uninstallation operations:
- HiveInstall - Binary path resolution, localhost vs remote detection
- HiveUninstall - Cleanup with capabilities cache removal

---

## Source Code Reference

**From:** `job_router.rs` lines 280-484

### HiveInstall (lines 280-401) - 121 LOC
Key steps:
1. Determine if localhost or remote
2. For localhost: Find or verify binary path
3. Display success message with instructions
4. (Remote SSH installation not yet implemented)

### HiveUninstall (lines 402-484) - 82 LOC
Key steps:
1. Validate hive exists
2. Remove from capabilities cache if present
3. Display success message
4. (Pre-flight check for running hive documented but not enforced)

---

## Deliverables

### 1. Implement HiveInstall

**File:** `src/install.rs`
```rust
// TEAM-213: Install hive configuration

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveInstallRequest, HiveInstallResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Install hive configuration
///
/// COPIED FROM: job_router.rs lines 280-401
///
/// Steps:
/// 1. Validate hive exists in config
/// 2. Determine if localhost or remote
/// 3. For localhost: Find or verify binary path
/// 4. Display success message
///
/// NOTE: Remote SSH installation not yet implemented
///
/// # Arguments
/// * `request` - Install request with alias
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveInstallResponse)` - Success with binary path
/// * `Err` - Configuration error or binary not found
pub async fn execute_hive_install(
    request: HiveInstallRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveInstallResponse> {
    let alias = &request.alias;
    let hive_config = validate_hive_exists(&config, alias)?;

    NARRATE
        .action("hive_install")
        .job_id(job_id)
        .context(alias)
        .human("🔧 Installing hive '{}'")
        .emit();

    // STEP 1: Determine if this is localhost or remote installation
    let is_remote = hive_config.hostname != "127.0.0.1" && hive_config.hostname != "localhost";

    if is_remote {
        // REMOTE INSTALLATION
        let host = &hive_config.hostname;
        let ssh_port = hive_config.ssh_port;
        let user = &hive_config.ssh_user;

        NARRATE
            .action("hive_mode")
            .job_id(job_id)
            .context(format!("{}@{}:{}", user, host, ssh_port))
            .human("🌐 Remote installation: {}")
            .emit();

        // TODO: Implement remote SSH installation
        NARRATE
            .action("hive_not_impl")
            .job_id(job_id)
            .human(
                "❌ Remote SSH installation not yet implemented.\n\
                   \n\
                   Currently only localhost installation is supported.",
            )
            .emit();
        return Err(anyhow::anyhow!("Remote installation not yet implemented"));
    } else {
        // LOCALHOST INSTALLATION
        NARRATE
            .action("hive_mode")
            .job_id(job_id)
            .human("🏠 Localhost installation")
            .emit();

        // STEP 2: Find or build the rbee-hive binary
        let binary = if let Some(provided_path) = &hive_config.binary_path {
            NARRATE
                .action("hive_binary")
                .job_id(job_id)
                .context(provided_path)
                .human("📁 Using provided binary path: {}")
                .emit();

            // Verify binary exists
            let path = std::path::Path::new(provided_path);
            if !path.exists() {
                NARRATE
                    .action("hive_bin_err")
                    .job_id(job_id)
                    .context(provided_path)
                    .human("❌ Binary not found at: {}")
                    .emit();
                return Err(anyhow::anyhow!("Binary not found: {}", provided_path));
            }

            NARRATE
                .action("hive_binary")
                .job_id(job_id)
                .human("✅ Binary found")
                .emit();

            provided_path.clone()
        } else {
            // Find binary in target directory
            NARRATE
                .action("hive_binary")
                .job_id(job_id)
                .human("🔍 Looking for rbee-hive binary in target/debug...")
                .emit();

            let debug_path = std::path::PathBuf::from("target/debug/rbee-hive");
            let release_path = std::path::PathBuf::from("target/release/rbee-hive");

            if debug_path.exists() {
                NARRATE
                    .action("hive_binary")
                    .job_id(job_id)
                    .context(debug_path.display().to_string())
                    .human("✅ Found binary at: {}")
                    .emit();
                debug_path.display().to_string()
            } else if release_path.exists() {
                NARRATE
                    .action("hive_binary")
                    .job_id(job_id)
                    .context(release_path.display().to_string())
                    .human("✅ Found binary at: {}")
                    .emit();
                release_path.display().to_string()
            } else {
                NARRATE
                    .action("hive_bin_err")
                    .job_id(job_id)
                    .human(
                        "❌ rbee-hive binary not found.\n\
                         \n\
                         Please build it first:\n\
                         \n\
                           cargo build --bin rbee-hive\n\
                         \n\
                         Or provide a binary path:\n\
                         \n\
                           ./rbee hive install --binary-path /path/to/rbee-hive",
                    )
                    .emit();
                return Err(anyhow::anyhow!(
                    "rbee-hive binary not found. Build it with: cargo build --bin rbee-hive"
                ));
            }
        };

        NARRATE
            .action("hive_complete")
            .job_id(job_id)
            .context(alias)
            .context(hive_config.hive_port.to_string())
            .context(&binary)
            .human(
                "✅ Hive '{0}' configured successfully!\n\
                 \n\
                 Configuration:\n\
                 - Host: localhost\n\
                 - Port: {1}\n\
                 - Binary: {2}\n\
                 \n\
                 To start the hive:\n\
                 \n\
                   ./rbee hive start --host {0}",
            )
            .emit();

        Ok(HiveInstallResponse {
            success: true,
            message: format!("Hive '{}' configured successfully", alias),
            binary_path: Some(binary),
        })
    }
}
```

### 2. Implement HiveUninstall

**File:** `src/uninstall.rs`
```rust
// TEAM-213: Uninstall hive configuration

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::sync::Arc;

use crate::types::{HiveUninstallRequest, HiveUninstallResponse};
use crate::validation::validate_hive_exists;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-life");

/// Uninstall hive configuration
///
/// COPIED FROM: job_router.rs lines 402-484
///
/// Steps:
/// 1. Validate hive exists
/// 2. Remove from capabilities cache if present
/// 3. Display success message
///
/// NOTE: Pre-flight check (hive must be stopped) is documented but not enforced.
/// User must manually stop the hive first: ./rbee hive stop
///
/// # Arguments
/// * `request` - Uninstall request with alias
/// * `config` - RbeeConfig with hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(HiveUninstallResponse)` - Success message
/// * `Err` - Configuration error
pub async fn execute_hive_uninstall(
    request: HiveUninstallRequest,
    config: Arc<RbeeConfig>,
    job_id: &str,
) -> Result<HiveUninstallResponse> {
    let alias = &request.alias;
    let _hive_config = validate_hive_exists(&config, alias)?;

    NARRATE
        .action("hive_uninstall")
        .job_id(job_id)
        .context(alias)
        .human("🗑️  Uninstalling hive '{}'")
        .emit();

    // TEAM-196: Remove from capabilities cache
    if config.capabilities.contains(alias) {
        NARRATE
            .action("hive_cache_cleanup")
            .job_id(job_id)
            .human("🗑️  Removing from capabilities cache...")
            .emit();

        let mut config_mut = (**config).clone();
        config_mut.capabilities.remove(alias);
        if let Err(e) = config_mut.capabilities.save() {
            NARRATE
                .action("hive_cache_error")
                .job_id(job_id)
                .context(e.to_string())
                .human("⚠️  Failed to save capabilities cache: {}")
                .emit();
        } else {
            NARRATE
                .action("hive_cache_removed")
                .job_id(job_id)
                .human("✅ Removed from capabilities cache")
                .emit();
        }
    }

    NARRATE
        .action("hive_complete")
        .job_id(job_id)
        .context(alias)
        .human(
            "✅ Hive '{}' uninstalled successfully.\n\
             \n\
             To remove from config, edit ~/.config/rbee/hives.conf",
        )
        .emit();

    // TEAM-189: Documented pre-flight check requirements
    //
    // CURRENT IMPLEMENTATION:
    // 1. Check if hive exists in catalog → error if not found
    // 2. Check if hive is running (health endpoint) → error if running
    // 3. Remove from catalog
    //
    // IMPORTANT: User must manually stop the hive first:
    //   ./rbee hive stop
    // This ensures clean shutdown of hive and all child workers.
    //
    // FUTURE ENHANCEMENTS (for TEAM-190+):
    //
    // CATALOG-ONLY MODE (catalog_only=true):
    // - Used for unreachable remote hives (network issues, host down)
    // - Simply remove HiveRecord from catalog
    // - No SSH connection or health check
    // - Warn user about orphaned processes on remote host
    //
    // ADDITIONAL CLEANUP OPTIONS (flags):
    // - --remove-workers: Delete worker binaries (requires hive stopped)
    // - --remove-models: Delete model files (requires hive stopped)
    // - --remove-binary: Delete hive binary itself
    //
    // LOCALHOST FULL CLEANUP:
    // 1. Verify hive stopped (health check fails)
    // 2. Verify no worker processes running (pgrep llm-worker)
    // 3. Optional: Remove worker binaries if --remove-workers
    // 4. Optional: Remove models if --remove-models
    // 5. Optional: Remove hive binary if --remove-binary
    // 6. Remove from catalog
    //
    // REMOTE SSH FULL CLEANUP:
    // 1. Run SshTest to verify connectivity
    // 2. Verify hive stopped (SSH: curl health or pgrep)
    // 3. Verify no worker processes (SSH: pgrep llm-worker)
    // 4. Optional: Remove files via SSH (rm commands)
    // 5. Remove from catalog

    Ok(HiveUninstallResponse {
        success: true,
        message: format!("Hive '{}' uninstalled successfully", alias),
    })
}
```

### 3. Update lib.rs Exports

```rust
// TEAM-213: Export install/uninstall operations
pub use install::execute_hive_install;
pub use uninstall::execute_hive_uninstall;
```

---

## Acceptance Criteria

- [ ] `src/install.rs` implemented (localhost + remote detection)
- [ ] `src/uninstall.rs` implemented (with cache cleanup)
- [ ] Binary path resolution working (provided vs auto-detect)
- [ ] Capabilities cache cleanup working
- [ ] All narration includes `.job_id(job_id)` for SSE routing
- [ ] Error messages match original exactly
- [ ] Crate compiles: `cargo check -p queen-rbee-hive-lifecycle`
- [ ] No TODO markers in TEAM-213 code
- [ ] All code has TEAM-213 signatures

---

## Notes

- Install operation does NOT start the hive (user must run `./rbee hive start`)
- Uninstall operation assumes hive is already stopped (pre-flight check documented)
- Remote SSH installation not yet implemented (returns error)
- Preserve exact error messages and instructions from job_router.rs
- All narration MUST include `.job_id(job_id)` for SSE routing
