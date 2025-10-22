# Hive Installation & Device Detection Proposal

## Vision

**User runs `rbee hive install` - Queen handles everything.**

- Deprecates manual build commands
- Supports localhost AND remote SSH installations
- Automatic device detection on first heartbeat
- Clean install/uninstall workflow

---

## Architecture

### User Command: `rbee hive install`

**Replaces all manual build/start commands!**

```bash
# Localhost installation
rbee hive install --id localhost

# Remote SSH installation
rbee hive install --id hive-prod-01 --ssh admin@192.168.1.100:22
```

### Flow Overview

```
User → rbee hive install
     ↓
Queen → Check if localhost or SSH
     ↓
[LOCALHOST]              [REMOTE SSH]
Git already exists       SSH into machine
Build on localhost       Clone git repo
Start hive process       Build rbee-hive
     ↓                   Start hive process
     └───────────────────────┘
                ↓
        Hive sends heartbeat
                ↓
        Queen receives heartbeat
                ↓
    Check: Does hive have capabilities in catalog?
                ↓
        [NO] → Call hive's device detection API
             → Hive detects its own devices
             → Hive returns capabilities
             → Queen stores in catalog (ONLY TIME!)
                ↓
        [YES] → Do nothing (already configured)
```

---

## Implementation

### 1. `execute_hive_install()` - Main Entry Point

```rust
/// Execute hive installation
///
/// TEAM-186: User-driven hive installation
///
/// **Replaces:** Manual build/start commands
///
/// **Flow:**
/// 1. Add hive to catalog (user configuration)
/// 2. If localhost → Build locally
/// 3. If SSH → SSH into machine, clone repo, build
/// 4. Start hive process
/// 5. Wait for heartbeat (async)
/// 6. On first heartbeat → Trigger device detection
///
/// # Arguments
/// * `catalog` - Hive catalog for configuration
/// * `request` - Installation request (hive_id, ssh_config)
///
/// # Returns
/// * `Ok(HiveInstallResponse)` - Installation initiated
/// * `Err` - Installation failed
pub async fn execute_hive_install(
    catalog: Arc<HiveCatalog>,
    request: HiveInstallRequest,
) -> Result<HiveInstallResponse> {
    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_INSTALL, &request.hive_id)
        .human(format!("🐝 Installing hive: {}", request.hive_id))
        .emit();

    // Step 1: Add hive to catalog (user configuration)
    let hive = HiveRecord {
        id: request.hive_id.clone(),
        host: request.ssh_config.as_ref()
            .map(|s| s.host.clone())
            .unwrap_or_else(|| "127.0.0.1".to_string()),
        port: request.port.unwrap_or(8600),
        ssh_host: request.ssh_config.as_ref().map(|s| s.host.clone()),
        ssh_port: request.ssh_config.as_ref().map(|s| s.port),
        ssh_user: request.ssh_config.as_ref().map(|s| s.user.clone()),
        devices: None, // Will be detected on first heartbeat
        created_at_ms: chrono::Utc::now().timestamp_millis(),
        updated_at_ms: chrono::Utc::now().timestamp_millis(),
    };

    catalog.add_hive(hive.clone()).await
        .context("Failed to add hive to catalog")?;

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_INSTALL, &hive.id)
        .human(format!("✅ Hive '{}' added to catalog", hive.id))
        .emit();

    // Step 2: Build and start hive
    if request.ssh_config.is_some() {
        // Remote SSH installation
        install_hive_remote(&hive, &request.ssh_config.unwrap()).await?;
    } else {
        // Localhost installation
        install_hive_localhost(&hive).await?;
    }

    Ok(HiveInstallResponse {
        hive_id: hive.id,
        hive_url: format!("http://{}:{}", hive.host, hive.port),
        status: "installed".to_string(),
    })
}
```

### 2. `install_hive_localhost()` - Local Installation

```rust
/// Install hive on localhost
///
/// **Assumptions:**
/// - Git repo already cloned
/// - Just need to build and start
///
/// **Flow:**
/// 1. Build rbee-hive binary
/// 2. Capture build warnings
/// 3. Start hive process
async fn install_hive_localhost(hive: &HiveRecord) -> Result<()> {
    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_BUILD, "localhost")
        .human("🔨 Building rbee-hive on localhost...")
        .emit();

    // Build rbee-hive
    let output = Command::new("cargo")
        .args(&["build", "--bin", "rbee-hive"])
        .output()
        .await
        .context("Failed to run cargo build")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("Build failed:\n{}", stderr));
    }

    // Check for warnings
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    if combined.contains("warning:") {
        let count = combined.matches("warning:").count();
        eprintln!("⚠️  BUILD WARNINGS: {} warnings detected", count);
        for line in combined.lines() {
            if line.contains("warning:") {
                eprintln!("  {}", line);
            }
        }
    }

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_BUILD, "success")
        .human("✅ Build complete")
        .emit();

    // Start hive process
    start_hive_process(hive).await?;

    Ok(())
}
```

### 3. `install_hive_remote()` - SSH Installation

```rust
/// Install hive on remote machine via SSH
///
/// **Flow:**
/// 1. SSH into machine
/// 2. Clone git repo (if not exists)
/// 3. Build rbee-hive
/// 4. Start hive process
async fn install_hive_remote(
    hive: &HiveRecord,
    ssh_config: &SshConfig,
) -> Result<()> {
    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_INSTALL, "remote")
        .human(format!("🔐 SSH into {}@{}", ssh_config.user, ssh_config.host))
        .emit();

    // SSH command builder
    let ssh_host = format!("{}@{}", ssh_config.user, ssh_config.host);

    // Step 1: Clone repo (if not exists)
    let clone_cmd = format!(
        "if [ ! -d llama-orch ]; then \
         git clone https://github.com/your-org/llama-orch.git; \
         fi"
    );

    Command::new("ssh")
        .args(&["-p", &ssh_config.port.to_string(), &ssh_host, &clone_cmd])
        .output()
        .await
        .context("Failed to clone repo")?;

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_INSTALL, "clone")
        .human("✅ Repo cloned/verified")
        .emit();

    // Step 2: Build rbee-hive
    let build_cmd = "cd llama-orch && cargo build --bin rbee-hive";

    let output = Command::new("ssh")
        .args(&["-p", &ssh_config.port.to_string(), &ssh_host, build_cmd])
        .output()
        .await
        .context("Failed to build rbee-hive")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("Remote build failed:\n{}", stderr));
    }

    // Check for warnings
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    if combined.contains("warning:") {
        let count = combined.matches("warning:").count();
        eprintln!("⚠️  REMOTE BUILD WARNINGS: {} warnings", count);
    }

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_BUILD, "success")
        .human("✅ Remote build complete")
        .emit();

    // Step 3: Start hive process
    start_hive_process_remote(hive, ssh_config).await?;

    Ok(())
}
```

### 4. Device Detection on First Heartbeat

**In `heartbeat.rs` (queen-rbee):**

```rust
/// Handle new hive discovery workflow
///
/// TEAM-186: Automatic device detection on first heartbeat
///
/// **Flow:**
/// 1. Check if hive has capabilities in catalog
/// 2. If NO → Call hive's device detection API
/// 3. Hive detects its own devices (GPU, CPU)
/// 4. Queen stores capabilities in catalog
/// 5. This is the ONLY time catalog is auto-updated!
async fn handle_new_hive_discovery(
    state: &HeartbeatState,
    payload: &HiveHeartbeatPayload,
) -> Result<(), (StatusCode, String)> {
    // Check if hive exists in catalog
    let hive = state.catalog
        .get_hive(&payload.hive_id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    match hive {
        Some(h) if h.devices.is_some() => {
            // Hive already has capabilities → Do nothing
            eprintln!("✅ Hive '{}' already has capabilities", payload.hive_id);
            Ok(())
        }
        Some(h) => {
            // Hive exists but NO capabilities → Detect now!
            eprintln!("🔍 Detecting devices for hive '{}'...", payload.hive_id);

            // Call hive's device detection API
            let hive_url = format!("http://{}:{}", h.host, h.port);
            let capabilities = call_hive_device_detection(&hive_url).await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            // Store in catalog (ONLY TIME catalog is auto-updated!)
            state.catalog
                .update_devices(&payload.hive_id, capabilities)
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            eprintln!("✅ Device capabilities stored for '{}'", payload.hive_id);
            Ok(())
        }
        None => {
            // Hive not in catalog → Reject!
            Err((
                StatusCode::NOT_FOUND,
                format!("Hive '{}' not registered. Run 'rbee hive install' first.", payload.hive_id)
            ))
        }
    }
}

/// Call hive's device detection API
///
/// **Hive does its own detection!**
/// Queen cannot assume same machine.
async fn call_hive_device_detection(hive_url: &str) -> Result<DeviceCapabilities> {
    let url = format!("{}/v1/devices/detect", hive_url);
    
    let response = reqwest::get(&url)
        .await
        .context("Failed to call hive device detection")?;

    if !response.status().is_success() {
        return Err(anyhow!("Device detection failed: {}", response.status()));
    }

    let capabilities = response.json::<DeviceCapabilities>()
        .await
        .context("Failed to parse device capabilities")?;

    Ok(capabilities)
}
```

### 5. `execute_hive_uninstall()` - Cleanup

```rust
/// Execute hive uninstallation
///
/// TEAM-186: Clean uninstall workflow
///
/// **Flow:**
/// 1. Stop hive process (if running)
/// 2. Remove from catalog
/// 3. Optionally: Remove from registry (RAM)
///
/// # Arguments
/// * `catalog` - Hive catalog
/// * `hive_id` - Hive to uninstall
pub async fn execute_hive_uninstall(
    catalog: Arc<HiveCatalog>,
    hive_id: &str,
) -> Result<()> {
    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_UNINSTALL, hive_id)
        .human(format!("🗑️  Uninstalling hive: {}", hive_id))
        .emit();

    // Step 1: Get hive info
    let hive = catalog.get_hive(hive_id).await?
        .ok_or_else(|| anyhow!("Hive '{}' not found", hive_id))?;

    // Step 2: Stop hive process (best effort)
    stop_hive_process(&hive).await.ok();

    // Step 3: Remove from catalog
    catalog.remove_hive(hive_id).await?;

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_UNINSTALL, hive_id)
        .human(format!("✅ Hive '{}' uninstalled", hive_id))
        .emit();

    Ok(())
}
```

---

## Benefits

### ✅ User-Driven Installation
- Single command: `rbee hive install`
- Deprecates all manual build/start commands
- Clear, simple workflow

### ✅ Localhost AND Remote Support
- Localhost: Assumes git cloned, just build
- Remote SSH: Clone + build + start
- Same command for both!

### ✅ Build Warnings Visible
- Warnings captured during install
- Displayed to user immediately
- Can't be ignored!

### ✅ Automatic Device Detection
- Happens on first heartbeat
- Hive detects its own devices (correct!)
- Queen stores result in catalog
- **ONLY TIME** catalog is auto-updated

### ✅ Clean Separation
- **Catalog** = User configuration (manual)
- **Registry** = Runtime state (automatic)
- **Device detection** = One-time on first heartbeat

### ✅ Uninstall Support
- Clean removal workflow
- Stops process + removes from catalog
- No orphaned entries

---

## Key Architectural Decisions

### 1. **Hive Does Its Own Device Detection** ✅
- Queen CANNOT assume same machine
- Hive has `/v1/devices/detect` API
- Queen calls hive's API, stores result

### 2. **Catalog Auto-Update ONLY for Devices** ✅
- User adds hive → Manual (via install command)
- Hive sends heartbeat → Registry updated (RAM)
- Hive has no devices → Queen calls detection, stores ONCE
- **This is the ONLY exception to "catalog is user-managed"**

### 3. **Install Command Replaces Everything** ✅
- No more `cargo build --bin rbee-hive`
- No more manual process spawning
- Just: `rbee hive install --id localhost`

---

## Command Examples

### Localhost Installation
```bash
# Install hive on localhost
rbee hive install --id localhost

# Output:
# 🐝 Installing hive: localhost
# ✅ Hive 'localhost' added to catalog
# 🔨 Building rbee-hive on localhost...
# ⚠️  BUILD WARNINGS: 2 warnings detected
#   warning: unused variable: `foo`
#   warning: field is never read: `bar`
# ✅ Build complete
# ✅ Hive spawn initiated: http://127.0.0.1:8600
# (Waiting for heartbeat...)
# 🔍 Detecting devices for hive 'localhost'...
# ✅ Device capabilities stored for 'localhost'
```

### Remote SSH Installation
```bash
# Install hive on remote machine
rbee hive install --id hive-prod-01 --ssh admin@192.168.1.100:22

# Output:
# 🐝 Installing hive: hive-prod-01
# ✅ Hive 'hive-prod-01' added to catalog
# 🔐 SSH into admin@192.168.1.100
# ✅ Repo cloned/verified
# 🔨 Building rbee-hive remotely...
# ⚠️  REMOTE BUILD WARNINGS: 1 warning
# ✅ Remote build complete
# ✅ Hive spawn initiated: http://192.168.1.100:8600
# (Waiting for heartbeat...)
# 🔍 Detecting devices for hive 'hive-prod-01'...
# ✅ Device capabilities stored for 'hive-prod-01'
```

### Uninstall
```bash
# Uninstall hive
rbee hive uninstall --id localhost

# Output:
# 🗑️  Uninstalling hive: localhost
# ✅ Hive 'localhost' uninstalled
```

---

## Summary

**The Correct Architecture:**

1. **User runs `rbee hive install`** (deprecates all manual commands)
2. **Queen adds to catalog** (user configuration)
3. **Queen builds hive** (localhost or SSH)
4. **Queen starts hive** (process spawning)
5. **Hive sends heartbeat** → Registry updated (RAM)
6. **Queen checks: Does hive have capabilities?**
   - NO → Call hive's `/v1/devices/detect` API
   - Hive detects its own devices
   - Queen stores in catalog (**ONLY TIME auto-update!**)
   - YES → Do nothing

**Clean separation:**
- Catalog = User configuration (manual, except device detection)
- Registry = Runtime state (automatic)
- Device detection = Hive's responsibility (Queen just stores result)

**No more assumptions about same machine!** 🎯
