# Installation Specification - Rust Binary Distribution

**Status:** Normative  
**Version:** 1.0  
**Date:** 2025-10-09  
**Architecture:** See CRITICAL_RULES.md for latest architecture (queen-rbee orchestration)

---

## Overview
**Status:** SPECIFICATION ONLY - NOT IMPLEMENTED

## Problem Statement

Current implementation has **HARDCODED PATHS** in `bin/rbee-keeper/src/commands/pool.rs`:
```rust
"cd ~/Projects/llama-orch && ./target/release/rbee-hive models list"
```

This BREAKS remote deployment and violates industry standards.

## Solution: Rust-Based Installation

### Architecture

```
rbee (CLI)
├── install (subcommand) - Install binaries to standard paths
├── config (subcommand) - Manage configuration
└── deploy (subcommand) - Deploy to remote machines
```

### Implementation Plan

#### 1. Add `rbee install` Subcommand

**File:** `bin/rbee-keeper/src/commands/install.rs` (NEW)

```rust
//! Installation command - Install rbee binaries to standard paths
//!
//! TEAM-036: Implements XDG Base Directory specification
//! Replaces shell script with proper Rust implementation

use anyhow::Result;
use std::fs;
use std::path::PathBuf;

pub enum InstallTarget {
    User,   // ~/.local/bin
    System, // /usr/local/bin (requires sudo)
}

pub fn handle(target: InstallTarget) -> Result<()> {
    let (bin_dir, config_dir, data_dir) = match target {
        InstallTarget::User => (
            dirs::home_dir().unwrap().join(".local/bin"),
            dirs::home_dir().unwrap().join(".config/rbee"),
            dirs::home_dir().unwrap().join(".local/share/rbee"),
        ),
        InstallTarget::System => (
            PathBuf::from("/usr/local/bin"),
            PathBuf::from("/etc/rbee"),
            PathBuf::from("/var/lib/rbee"),
        ),
    };

    // 1. Create directories
    fs::create_dir_all(&bin_dir)?;
    fs::create_dir_all(&config_dir)?;
    fs::create_dir_all(&data_dir.join("models"))?;

    // 2. Copy binaries
    let current_exe = std::env::current_exe()?;
    let exe_dir = current_exe.parent().unwrap();
    
    copy_binary(exe_dir, &bin_dir, "rbee")?;
    copy_binary(exe_dir, &bin_dir, "rbee-hive")?;
    copy_binary(exe_dir, &bin_dir, "llm-worker-rbee")?;

    // 3. Create default config
    create_default_config(&config_dir, &data_dir)?;

    println!("✅ Installation complete!");
    println!("Binaries: {}", bin_dir.display());
    println!("Config: {}", config_dir.display());
    
    Ok(())
}

fn copy_binary(src_dir: &Path, dest_dir: &Path, name: &str) -> Result<()> {
    let src = src_dir.join(name);
    let dest = dest_dir.join(name);
    fs::copy(&src, &dest)?;
    
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&dest)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&dest, perms)?;
    }
    
    Ok(())
}

fn create_default_config(config_dir: &Path, data_dir: &Path) -> Result<()> {
    let config_path = config_dir.join("config.toml");
    if config_path.exists() {
        return Ok(()); // Don't overwrite
    }

    let config = format!(
        r#"# rbee configuration
[pool]
name = "{}"
listen_addr = "0.0.0.0:8080"

[paths]
models_dir = "{}/models"
catalog_db = "{}/models.db"
"#,
        hostname::get()?.to_string_lossy(),
        data_dir.display(),
        data_dir.display()
    );

    fs::write(config_path, config)?;
    Ok(())
}
```

#### 2. Add Configuration Loading

**File:** `bin/rbee-keeper/src/config.rs` (NEW)

```rust
//! Configuration file loading
//!
//! TEAM-036: Implements XDG Base Directory specification

use anyhow::Result;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub pool: PoolConfig,
    pub paths: PathsConfig,
    pub remote: Option<RemoteConfig>,
}

#[derive(Debug, Deserialize)]
pub struct PoolConfig {
    pub name: String,
    pub listen_addr: String,
}

#[derive(Debug, Deserialize)]
pub struct PathsConfig {
    pub models_dir: PathBuf,
    pub catalog_db: PathBuf,
}

#[derive(Debug, Deserialize)]
pub struct RemoteConfig {
    /// Custom binary path on remote machines
    pub binary_path: Option<String>,
}

impl Config {
    /// Load config from standard locations
    /// Priority: RBEE_CONFIG env var > ~/.config/rbee/config.toml > /etc/rbee/config.toml
    pub fn load() -> Result<Self> {
        let config_path = if let Ok(path) = std::env::var("RBEE_CONFIG") {
            PathBuf::from(path)
        } else if let Some(home) = dirs::home_dir() {
            let user_config = home.join(".config/rbee/config.toml");
            if user_config.exists() {
                user_config
            } else {
                PathBuf::from("/etc/rbee/config.toml")
            }
        } else {
            PathBuf::from("/etc/rbee/config.toml")
        };

        let contents = std::fs::read_to_string(&config_path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }
}
```

#### 3. Fix pool.rs to Use PATH

**File:** `bin/rbee-keeper/src/commands/pool.rs` (MODIFY)

```rust
// BEFORE (WRONG):
"cd ~/Projects/llama-orch && ./target/release/rbee-hive models list"

// AFTER (CORRECT):
fn handle_models(action: ModelsAction, host: &str) -> Result<()> {
    // Load config to get custom binary path if set
    let binary = if let Ok(config) = Config::load() {
        config.remote
            .and_then(|r| r.binary_path)
            .unwrap_or_else(|| "rbee-hive".to_string())
    } else {
        "rbee-hive".to_string() // Default: use PATH
    };

    let command = match action {
        ModelsAction::List => format!("{} models list", binary),
        ModelsAction::Download { model } => {
            format!("{} models download {}", binary, model)
        }
        // ... etc
    };

    ssh::execute_remote_command_streaming(host, &command)?;
    Ok(())
}
```

#### 4. Add Deploy Command

**File:** `bin/rbee-keeper/src/commands/deploy.rs` (NEW)

```rust
//! Deploy command - Deploy rbee to remote machine
//!
//! TEAM-036: Automates remote deployment

use anyhow::Result;
use crate::ssh;

pub fn handle(host: &str, repo_url: Option<String>) -> Result<()> {
    println!("🚀 Deploying rbee to {}...", host);

    // 1. Check if repo exists
    let check_cmd = "test -d ~/llama-orch && echo 'exists' || echo 'missing'";
    // ... (check output)

    // 2. Clone or pull
    if repo_missing {
        let url = repo_url.unwrap_or_else(|| 
            "https://github.com/user/llama-orch.git".to_string()
        );
        ssh::execute_remote_command_streaming(
            host,
            &format!("git clone {} ~/llama-orch", url)
        )?;
    } else {
        ssh::execute_remote_command_streaming(
            host,
            "cd ~/llama-orch && git pull"
        )?;
    }

    // 3. Build
    ssh::execute_remote_command_streaming(
        host,
        "cd ~/llama-orch && cargo build --release"
    )?;

    // 4. Install
    ssh::execute_remote_command_streaming(
        host,
        "cd ~/llama-orch && ./target/release/rbee install --user"
    )?;

    println!("✅ Deployment complete!");
    Ok(())
}
```

### CLI Changes

**File:** `bin/rbee-keeper/src/cli.rs`

```rust
#[derive(Subcommand)]
pub enum Commands {
    /// Install rbee binaries to standard paths
    Install {
        /// Install to system paths (requires sudo)
        #[arg(long)]
        system: bool,
    },
    
    /// Deploy to remote machine
    Deploy {
        /// Remote host
        #[arg(long)]
        host: String,
        
        /// Git repository URL (optional)
        #[arg(long)]
        repo: Option<String>,
    },
    
    /// Manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    
    // ... existing commands
}
```

### Usage Examples

```bash
# Local installation
cargo build --release
./target/release/rbee install --user

# Now binaries are in ~/.local/bin
rbee --version
rbee-hive --version

# Deploy to remote
rbee deploy --host mac.home.arpa

# Use remote commands (now works!)
rbee pool models list --host mac.home.arpa
rbee pool git pull --host mac.home.arpa
```

### Dependencies to Add

**File:** `bin/rbee-keeper/Cargo.toml`

```toml
[dependencies]
dirs = "5.0"           # XDG directory discovery
toml = "0.8"           # Config file parsing
hostname = "0.4"       # Get hostname for default config
```

### Testing

```bash
# Test local install
cargo build --release
./target/release/rbee install --user
ls ~/.local/bin/rbee*

# Test config loading
cat ~/.config/rbee/config.toml

# Test remote deployment
rbee deploy --host mac.home.arpa
ssh mac.home.arpa "which rbee-hive"
```

---

## Implementation Checklist

- [ ] Create `commands/install.rs`
- [ ] Create `config.rs`
- [ ] Create `commands/deploy.rs`
- [ ] Modify `commands/pool.rs` to use PATH
- [ ] Update `cli.rs` with new commands
- [ ] Add dependencies to `Cargo.toml`
- [ ] Write unit tests
- [ ] Update README.md
- [ ] Test on fresh remote machine
- [ ] Delete shell scripts

---

**Status:** SPECIFICATION COMPLETE  
**Implementation:** TEAM-036  
**Estimated effort:** 6-8 hours  
**Priority:** HIGH - Blocks production deployment
