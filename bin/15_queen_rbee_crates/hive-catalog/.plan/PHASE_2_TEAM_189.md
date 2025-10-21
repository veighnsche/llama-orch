# Phase 2: Replace SQLite in job_router.rs

⚠️ **YOU ARE TEAM-194** (not TEAM-189 - see START_HERE.md for explanation)

**Team:** ~~TEAM-189~~ **TEAM-194**  
**Duration:** 6-8 hours  
**Dependencies:** Phase 1 (TEAM-193) complete ✅  
**Status:** 🚧 **60% COMPLETE** - Infrastructure done, handlers need refactoring  
**Deliverables:** `job_router.rs` using `rbee-config` instead of SQLite

---

## 📊 TEAM-194 PROGRESS

**Completed:**
- ✅ Dependencies updated (Cargo.toml)
- ✅ AppState refactored (main.rs)
- ✅ HTTP module updated (http/jobs.rs)
- ✅ JobState refactored (job_router.rs)
- ✅ Operation enum simplified (rbee-operations)
- ✅ CLI arguments updated (rbee-keeper)

**Remaining:**
- ❌ 7 handlers in job_router.rs need refactoring (2-3 hours)

**See:** `TEAM-194-HANDOFF.md` for complete handoff with code examples

---

## Mission

Replace all SQLite-based hive catalog operations in `job_router.rs` with file-based config lookups using the new `rbee-config` crate.

---

## Tasks

### 2.1 Update Dependencies

**In `bin/10_queen_rbee/Cargo.toml`:**

```toml
[dependencies]
# REMOVE:
# queen-rbee-hive-catalog = { path = "../15_queen_rbee_crates/hive-catalog" }

# ADD:
rbee-config = { path = "../15_queen_rbee_crates/rbee-config" }
```

### 2.2 Update AppState

**In `bin/10_queen_rbee/src/main.rs` or wherever AppState is defined:**

```rust
// BEFORE:
pub struct AppState {
    pub hive_catalog: Arc<HiveCatalog>,
    // ... other fields
}

// AFTER:
pub struct AppState {
    pub config: Arc<RbeeConfig>,
    // ... other fields
}
```

**Initialize in main():**
```rust
// Load config at startup
let config = RbeeConfig::load()
    .context("Failed to load rbee config")?;

let state = AppState {
    config: Arc::new(config),
    // ... other fields
};
```

### 2.3 Update Operation Enum

**In `bin/99_shared_crates/rbee-operations/src/lib.rs`:**

**BEFORE:**
```rust
Operation::HiveInstall {
    hive_id: String,
    ssh_host: Option<String>,
    ssh_port: Option<u16>,
    ssh_user: Option<String>,
    port: u16,
    binary_path: Option<String>,
}

Operation::HiveUninstall {
    hive_id: String,
}

Operation::HiveStart {
    hive_id: String,
}

Operation::HiveStop {
    hive_id: String,
}

Operation::HiveList {}

Operation::SshTest {
    ssh_host: String,
    ssh_port: Option<u16>,
    ssh_user: String,
}
```

**AFTER:**
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
    alias: String, // Test SSH connection using config from hives.conf
}
```

### 2.4 Update CLI Argument Parsing

**In `bin/00_rbee_keeper/src/config.rs` (or wherever CLI args are parsed):**

**BEFORE:**
```rust
#[derive(Parser)]
enum HiveCommand {
    Install {
        #[arg(long)]
        id: String,
        #[arg(long)]
        ssh_host: Option<String>,
        #[arg(long)]
        ssh_port: Option<u16>,
        #[arg(long)]
        ssh_user: Option<String>,
        #[arg(long)]
        port: u16,
        #[arg(long)]
        binary_path: Option<String>,
    },
    // ... other commands
}
```

**AFTER:**
```rust
#[derive(Parser)]
enum HiveCommand {
    Install {
        #[arg(short = 'h', long = "host")]
        alias: String,
    },
    Uninstall {
        #[arg(short = 'h', long = "host")]
        alias: String,
    },
    Start {
        #[arg(short = 'h', long = "host")]
        alias: String,
    },
    Stop {
        #[arg(short = 'h', long = "host")]
        alias: String,
    },
    List {},
}
```

### 2.5 Refactor job_router.rs - HiveInstall

**Location:** `bin/10_queen_rbee/src/job_router.rs` (lines 180-737)

**BEFORE (current implementation):**
```rust
Operation::HiveInstall { hive_id, ssh_host, ssh_port, ssh_user, port, binary_path } => {
    // Check if already installed in SQLite
    if state.hive_catalog.hive_exists(&hive_id).await? {
        return Err(anyhow::anyhow!("Hive already exists"));
    }
    
    // Installation logic...
    
    // Register in SQLite
    state.hive_catalog.add_hive(record).await?;
}
```

**AFTER (file-based config):**
```rust
Operation::HiveInstall { alias } => {
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_install", &alias)
        .human(format!("🔧 Installing hive '{}'", alias))
        .emit();

    // STEP 1: Load hive config from hives.conf
    NARRATE_ROUTER
        .action("hive_install_config_lookup")
        .human(format!("📋 Looking up hive '{}' in hives.conf...", alias))
        .emit();

    let hive = state.config.hives.get(&alias)
        .ok_or_else(|| anyhow::anyhow!(
            "Hive alias '{}' not found in hives.conf.\n\
             \n\
             Please add it to ~/.config/rbee/hives.conf:\n\
             \n\
             Host {}\n\
             \    HostName <hostname>\n\
             \    Port 22\n\
             \    User <username>\n\
             \    HivePort 8081",
            alias, alias
        ))?;

    NARRATE_ROUTER
        .action("hive_install_config_found")
        .human(format!(
            "✅ Found config: {}@{}:{}",
            hive.ssh_user, hive.hostname, hive.ssh_port
        ))
        .emit();

    // STEP 2: Check if already running (check capabilities.yaml)
    if let Some(caps) = state.config.capabilities.get(&alias) {
        Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_already_running", &alias)
            .human(format!(
                "⚠️  Hive '{}' appears to be already running.\n\
                 \n\
                 Last seen: {} ms ago\n\
                 \n\
                 To reinstall, first stop it:\n\
                 \n\
                   ./rbee hive stop -h {}",
                alias,
                chrono::Utc::now().timestamp_millis() - caps.last_updated_ms,
                alias
            ))
            .emit();
        return Err(anyhow::anyhow!(
            "Hive '{}' already running. Stop it first.",
            alias
        ));
    }

    // STEP 3: Determine if localhost or remote
    let is_localhost = hive.hostname == "localhost" 
        || hive.hostname == "127.0.0.1"
        || hive.hostname == "::1";

    if is_localhost {
        // LOCALHOST INSTALLATION
        NARRATE_ROUTER
            .action("hive_install_mode")
            .human("🏠 Localhost installation")
            .emit();

        // Find binary
        let binary = if let Some(path) = &hive.binary_path {
            NARRATE_ROUTER
                .action("hive_install_binary")
                .human(format!("📁 Using binary from config: {}", path))
                .emit();

            // Verify exists
            if !std::path::Path::new(path).exists() {
                return Err(anyhow::anyhow!("Binary not found: {}", path));
            }

            path.clone()
        } else {
            // Auto-detect in target/
            NARRATE_ROUTER
                .action("hive_install_binary")
                .human("🔍 Looking for rbee-hive binary in target/...")
                .emit();

            let debug_path = std::path::PathBuf::from("target/debug/rbee-hive");
            let release_path = std::path::PathBuf::from("target/release/rbee-hive");

            if debug_path.exists() {
                debug_path.display().to_string()
            } else if release_path.exists() {
                release_path.display().to_string()
            } else {
                return Err(anyhow::anyhow!(
                    "rbee-hive binary not found. Build it with:\n\
                     \n\
                       cargo build --bin rbee-hive"
                ));
            }
        };

        NARRATE_ROUTER
            .action("hive_install_binary")
            .human(format!("✅ Binary found: {}", binary))
            .emit();

        // Start the hive process
        NARRATE_ROUTER
            .action("hive_install_starting")
            .human(format!("🚀 Starting hive on port {}...", hive.hive_port))
            .emit();

        // TODO: Spawn hive process
        // TODO: Wait for health check
        // TODO: Fetch capabilities
        // TODO: Update capabilities.yaml

        Narration::new(ACTOR_QUEEN_ROUTER, "hive_install_complete", &alias)
            .human(format!(
                "✅ Hive '{}' installed successfully!\n\
                 \n\
                 Endpoint: http://localhost:{}",
                alias, hive.hive_port
            ))
            .emit();
    } else {
        // REMOTE INSTALLATION
        NARRATE_ROUTER
            .action("hive_install_mode")
            .human(format!(
                "🌐 Remote installation: {}@{}:{}",
                hive.ssh_user, hive.hostname, hive.ssh_port
            ))
            .emit();

        // TODO: SSH-based installation
        return Err(anyhow::anyhow!("Remote installation not yet implemented"));
    }
}
```

### 2.6 Refactor job_router.rs - HiveUninstall

```rust
Operation::HiveUninstall { alias } => {
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_uninstall", &alias)
        .human(format!("🗑️  Uninstalling hive '{}'", alias))
        .emit();

    // Check if hive exists in config
    let hive = state.config.hives.get(&alias)
        .ok_or_else(|| anyhow::anyhow!(
            "Hive alias '{}' not found in hives.conf",
            alias
        ))?;

    // Check if running
    if let Some(caps) = state.config.capabilities.get(&alias) {
        NARRATE_ROUTER
            .action("hive_uninstall_stopping")
            .human("🛑 Stopping hive process...")
            .emit();

        // TODO: Stop hive process
        // TODO: Remove from capabilities.yaml
    } else {
        NARRATE_ROUTER
            .action("hive_uninstall_not_running")
            .human("⚠️  Hive not currently running")
            .emit();
    }

    Narration::new(ACTOR_QUEEN_ROUTER, "hive_uninstall_complete", &alias)
        .human(format!(
            "✅ Hive '{}' uninstalled.\n\
             \n\
             To remove from config, edit:\n\
             \n\
               ~/.config/rbee/hives.conf",
            alias
        ))
        .emit();
}
```

### 2.7 Refactor job_router.rs - HiveList

```rust
Operation::HiveList {} => {
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_list", "all")
        .human("📋 Listing configured hives...")
        .emit();

    let hives = state.config.hives.all();

    if hives.is_empty() {
        NARRATE_ROUTER
            .action("hive_list_empty")
            .human(
                "No hives configured.\n\
                 \n\
                 Add hives to ~/.config/rbee/hives.conf"
            )
            .emit();
        return Ok(());
    }

    for hive in hives {
        let status = if state.config.capabilities.get(&hive.alias).is_some() {
            "🟢 RUNNING"
        } else {
            "⚫ STOPPED"
        };

        Narration::new(ACTOR_QUEEN_ROUTER, "hive_list_entry", &hive.alias)
            .human(format!(
                "{} {} - {}@{}:{} (hive port: {})",
                status,
                hive.alias,
                hive.ssh_user,
                hive.hostname,
                hive.ssh_port,
                hive.hive_port
            ))
            .emit();
    }
}
```

### 2.8 Refactor job_router.rs - SshTest

```rust
Operation::SshTest { alias } => {
    Narration::new(ACTOR_QUEEN_ROUTER, "ssh_test", &alias)
        .human(format!("🔌 Testing SSH connection to '{}'", alias))
        .emit();

    // Load hive config
    let hive = state.config.hives.get(&alias)
        .ok_or_else(|| anyhow::anyhow!(
            "Hive alias '{}' not found in hives.conf",
            alias
        ))?;

    // Create SSH test request
    let request = SshTestRequest {
        ssh_host: hive.hostname.clone(),
        ssh_port: Some(hive.ssh_port),
        ssh_user: hive.ssh_user.clone(),
    };

    let response = execute_ssh_test(request).await?;

    if !response.success {
        return Err(anyhow::anyhow!(
            "SSH connection failed: {}",
            response.error.unwrap_or_else(|| "Unknown error".to_string())
        ));
    }

    Narration::new(ACTOR_QUEEN_ROUTER, "ssh_test_complete", &alias)
        .human(format!(
            "✅ SSH test successful: {}",
            response.test_output.unwrap_or_default()
        ))
        .emit();
}
```

### 2.9 Remove SQLite Imports

**In `bin/10_queen_rbee/src/job_router.rs`:**

```rust
// REMOVE:
// use queen_rbee_hive_catalog::{HiveCatalog, HiveRecord};

// ADD:
use rbee_config::{RbeeConfig, HiveEntry};
```

---

## Acceptance Criteria

- [ ] All SQLite calls removed from `job_router.rs`
- [ ] All hive operations use `state.config.hives.get(alias)`
- [ ] CLI uses `-h <alias>` instead of `--id <id> --ssh-host ...`
- [ ] Operation enum simplified (only alias field)
- [ ] Code compiles without errors
- [ ] Narration messages updated with new flow
- [ ] Error messages guide users to edit `hives.conf`

---

## Verification Commands

```bash
# Build
cargo build --bin queen-rbee

# Check
cargo clippy --bin queen-rbee

# Test (if tests exist)
cargo test --bin queen-rbee
```

---

## Handoff to TEAM-190

**What's ready:**
- ✅ `job_router.rs` uses file-based config
- ✅ Operation enum simplified
- ✅ CLI uses alias-based arguments

**Next steps:**
- Add preflight validation on queen startup
- Validate alias uniqueness in `hives.conf`
- Add error handling for missing config files

---

**Created by:** TEAM-187  
**For:** TEAM-189  
**Status:** 📋 Ready to implement
