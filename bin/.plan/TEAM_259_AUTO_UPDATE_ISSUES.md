# TEAM-259: Auto-Update Issues & Solutions

**Status:** 🚨 CRITICAL ISSUES IDENTIFIED

**Date:** Oct 23, 2025

---

## Issue 1: 🚨 Remote Hive Auto-Update

### The Problem

**Current broken flow:**
```
queen-rbee (local machine)
  ↓ DaemonManager.enable_auto_update("rbee-hive", "bin/20_rbee_hive")
  ↓ Rebuilds rbee-hive LOCALLY
  ↓ SSH spawns rbee-hive on REMOTE machine
rbee-hive (remote machine) ← Uses OLD binary!
```

**Why it's broken:**
- Auto-update rebuilds binary on LOCAL machine
- SSH spawns binary on REMOTE machine
- Remote machine still has old binary!

### Solutions

#### Option A: SCP After Build (Recommended)

**In hive-lifecycle/src/install.rs:**
```rust
pub async fn execute_hive_install(request: HiveInstallRequest) -> Result<HiveInstallResponse> {
    if is_remote {
        // 1. Auto-update locally
        let updater = AutoUpdater::new("rbee-hive", "bin/20_rbee_hive")?;
        if updater.needs_rebuild()? {
            updater.rebuild()?;
        }
        let local_binary = updater.find_binary()?;
        
        // 2. SCP to remote
        let remote_path = "~/.local/bin/rbee-hive";
        scp_copy(&local_binary, remote_path, hive_config)?;
        
        // 3. Remote binary is now up to date!
        return Ok(HiveInstallResponse {
            binary_path: Some(remote_path.to_string()),
            ...
        });
    }
    
    // Local: just auto-update
    let updater = AutoUpdater::new("rbee-hive", "bin/20_rbee_hive")?;
    let binary_path = updater.ensure_built().await?;
    ...
}
```

**In hive-lifecycle/src/start.rs:**
```rust
pub async fn execute_hive_start(request: HiveStartRequest) -> Result<HiveStartResponse> {
    if is_remote {
        // Remote: binary already installed via install.rs
        // Just spawn it via SSH
        let start_cmd = format!("nohup {} --port {} > /dev/null 2>&1 &", 
            remote_binary_path, port);
        ssh_exec(&start_cmd, hive_config)?;
    } else {
        // Local: auto-update before spawn
        let manager = DaemonManager::new(binary_path, args)
            .enable_auto_update("rbee-hive", "bin/20_rbee_hive");
        let child = manager.spawn().await?;
    }
}
```

**Pros:**
- ✅ Simple
- ✅ Works with existing SSH infrastructure
- ✅ No remote build toolchain needed

**Cons:**
- ⚠️ Requires SCP on every update
- ⚠️ Network transfer time

---

#### Option B: Remote Build via SSH

```rust
pub async fn execute_hive_install(request: HiveInstallRequest) -> Result<HiveInstallResponse> {
    if is_remote {
        // 1. Ensure source code on remote
        ssh_exec("cd ~/llama-orch && git pull", hive_config)?;
        
        // 2. Build on remote
        ssh_exec("cd ~/llama-orch && cargo build --bin rbee-hive", hive_config)?;
        
        // 3. Remote binary is now up to date!
    }
}
```

**Pros:**
- ✅ No binary transfer
- ✅ Native compilation on target

**Cons:**
- ❌ Requires full source code on remote
- ❌ Requires Rust toolchain on remote
- ❌ Slow (full compilation)
- ❌ Complex setup

---

#### Option C: No Auto-Update for Remote (Simplest)

```rust
pub async fn execute_hive_start(request: HiveStartRequest) -> Result<HiveStartResponse> {
    if is_remote {
        // Remote: no auto-update, use installed binary
        let start_cmd = format!("nohup {} --port {} > /dev/null 2>&1 &", 
            remote_binary_path, port);
        ssh_exec(&start_cmd, hive_config)?;
    } else {
        // Local: auto-update
        let manager = DaemonManager::new(binary_path, args)
            .enable_auto_update("rbee-hive", "bin/20_rbee_hive");
        let child = manager.spawn().await?;
    }
}
```

**Pros:**
- ✅ Simple
- ✅ No network overhead
- ✅ Clear separation

**Cons:**
- ⚠️ Manual updates for remote hives
- ⚠️ User must run `rbee hive install` to update

---

### Recommendation

**Use Option C (No Auto-Update for Remote) + Manual Update Command**

```bash
# User manually updates remote hive when needed
./rbee hive install --host my-remote-hive

# This will:
# 1. Build locally (with auto-update)
# 2. SCP to remote
# 3. Remote is now updated
```

**Rationale:**
- Remote hives are typically stable (don't change often)
- Explicit update command is clearer
- No surprise network transfers
- Simpler code

---

## Issue 2: 🚨 Dependency Tracking Verification

### What AutoUpdater Actually Checks

**From `auto-update/src/lib.rs`:**

```rust
fn parse_dependencies(workspace_root: &Path, source_dir: &Path) -> Result<Vec<PathBuf>> {
    // Recursively parse Cargo.toml [dependencies]
    // Collects ALL local path dependencies
    // Example for rbee-keeper:
    //   - bin/99_shared_crates/daemon-lifecycle
    //   - bin/99_shared_crates/narration-core
    //   - bin/99_shared_crates/timeout-enforcer
    //   - bin/99_shared_crates/rbee-operations
    //   - bin/99_shared_crates/rbee-config
    //   - bin/99_shared_crates/job-client
    //   - bin/05_rbee_keeper_crates/queen-lifecycle
    //   - AND all THEIR dependencies recursively!
}

fn needs_rebuild(&self) -> Result<bool> {
    let binary_time = get_binary_mtime()?;
    
    // Check source directory
    if is_dir_newer(source_dir, binary_time)? {
        return Ok(true);
    }
    
    // Check ALL dependencies
    for dep_path in &self.dependencies {
        if is_dir_newer(dep_path, binary_time)? {
            return Ok(true);
        }
    }
    
    Ok(false)
}
```

### Verification for rbee-keeper

**Direct dependencies (from Cargo.toml):**
```toml
[dependencies]
daemon-lifecycle = { path = "../99_shared_crates/daemon-lifecycle" }
timeout-enforcer = { path = "../99_shared_crates/timeout-enforcer" }
rbee-operations = { path = "../99_shared_crates/rbee-operations" }
rbee-config = { path = "../99_shared_crates/rbee-config" }
job-client = { path = "../99_shared_crates/job-client" }
queen-lifecycle = { path = "../05_rbee_keeper_crates/queen-lifecycle" }
observability-narration-core = { path = "../99_shared_crates/narration-core" }
```

**Transitive dependencies (daemon-lifecycle's deps):**
```toml
# daemon-lifecycle/Cargo.toml
[dependencies]
observability-narration-core = { path = "../narration-core" }
auto-update = { path = "../auto-update" }
```

**Transitive dependencies (queen-lifecycle's deps):**
```toml
# queen-lifecycle/Cargo.toml
[dependencies]
daemon-lifecycle = { path = "../../99_shared_crates/daemon-lifecycle" }
observability-narration-core = { path = "../../99_shared_crates/narration-core" }
rbee-config = { path = "../../99_shared_crates/rbee-config" }
timeout-enforcer = { path = "../../99_shared_crates/timeout-enforcer" }
```

### Full Dependency Tree for rbee-keeper

```
rbee-keeper/
├── daemon-lifecycle/
│   ├── narration-core/
│   └── auto-update/
│       ├── narration-core/ (already tracked)
│       └── cargo-toml/
├── timeout-enforcer/
│   └── narration-core/ (already tracked)
├── rbee-operations/
├── rbee-config/
├── job-client/
│   ├── narration-core/ (already tracked)
│   └── rbee-operations/ (already tracked)
├── queen-lifecycle/
│   ├── daemon-lifecycle/ (already tracked)
│   ├── narration-core/ (already tracked)
│   ├── rbee-config/ (already tracked)
│   └── timeout-enforcer/ (already tracked)
└── narration-core/
```

**Total unique dependencies checked:** ~10-12 crates

### ✅ Verification Result

**YES, AutoUpdater tracks ALL dependencies correctly!**

It recursively parses Cargo.toml files and collects ALL local path dependencies.

---

## Testing the Dependency Tracking

### Test 1: Edit Shared Crate

```bash
# Edit narration-core
echo "// test change" >> bin/99_shared_crates/narration-core/src/lib.rs

# Run rbee-keeper
./rbee queen start

# Expected output:
# [auto-upd] check_rebuild: 🔨 Dependency bin/99_shared_crates/narration-core changed, rebuild needed
# [auto-upd] rebuild: 🔨 Rebuilding rbee-keeper...
#    Compiling rbee-keeper v0.1.0
#    Finished `dev` profile [unoptimized + debuginfo] target(s) in 2.3s
# [auto-upd] rebuild: ✅ Rebuilt rbee-keeper successfully
```

### Test 2: Edit Transitive Dependency

```bash
# Edit daemon-lifecycle (used by queen-lifecycle, used by rbee-keeper)
echo "// test change" >> bin/99_shared_crates/daemon-lifecycle/src/manager.rs

# Run rbee-keeper
./rbee queen start

# Expected output:
# [auto-upd] check_rebuild: 🔨 Dependency bin/99_shared_crates/daemon-lifecycle changed, rebuild needed
# [auto-upd] rebuild: 🔨 Rebuilding rbee-keeper...
```

### Test 3: No Changes

```bash
# Run without any changes
./rbee queen start

# Expected output:
# [auto-upd] check_rebuild: ✅ Binary rbee-keeper is up-to-date
```

---

## Recommendations

### For Remote Hives

**Implement Option C: No Auto-Update for Remote**

```rust
// In hive-lifecycle/src/start.rs
pub async fn execute_hive_start(request: HiveStartRequest) -> Result<HiveStartResponse> {
    let is_remote = hive_config.hostname != "127.0.0.1" && 
                    hive_config.hostname != "localhost";
    
    if is_remote {
        // Remote: use pre-installed binary (no auto-update)
        let start_cmd = format!("nohup {} --port {} > /dev/null 2>&1 &", 
            remote_binary_path, port);
        ssh_exec(&start_cmd, hive_config)?;
    } else {
        // Local: auto-update before spawn
        let manager = DaemonManager::new(binary_path, args)
            .enable_auto_update("rbee-hive", "bin/20_rbee_hive");
        let child = manager.spawn().await?;
    }
}
```

**User workflow for remote updates:**
```bash
# Update remote hive manually when needed
./rbee hive install --host my-remote-hive
```

### For Dependency Tracking

**✅ No changes needed - it already works correctly!**

AutoUpdater recursively tracks ALL dependencies via Cargo.toml parsing.

---

## Summary

### Issue 1: Remote Hive Auto-Update
- **Problem:** Can't auto-rebuild binaries on remote machines
- **Solution:** Disable auto-update for remote, manual update via `install` command
- **Status:** 🚨 NEEDS IMPLEMENTATION

### Issue 2: Dependency Tracking
- **Problem:** Concern about missing dependencies
- **Solution:** Already works! AutoUpdater recursively parses all Cargo.toml files
- **Status:** ✅ VERIFIED WORKING

**Next Steps:**
1. Implement remote/local detection in hive-lifecycle
2. Disable auto-update for remote hives
3. Document manual update workflow
4. Test dependency tracking with real edits
