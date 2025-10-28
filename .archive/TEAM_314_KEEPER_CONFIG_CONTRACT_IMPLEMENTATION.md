# TEAM-314: keeper-config-contract Implementation

**Priority:** 🟢 MEDIUM  
**Estimated Time:** 1 day  
**Status:** 📋 PLAN

---

## Goal

Create stable keeper configuration contract with schema validation.

---

## Step 1: Create Crate Structure

```bash
cd /home/vince/Projects/llama-orch/bin/97_contracts
cargo new --lib keeper-config-contract
cd keeper-config-contract
```

### Cargo.toml

```toml
[package]
name = "keeper-config-contract"
version = "0.1.0"
edition = "2021"
license = "GPL-3.0-or-later"
description = "Keeper configuration contract for rbee ecosystem"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"
anyhow = "1.0"

[dev-dependencies]
tempfile = "3.8"

[lints]
workspace = true
```

---

## Step 2: Find Current Implementation

### Search Commands

```bash
# Find Config struct
rg "pub struct Config" bin/00_rbee_keeper/src/config.rs -A 20

# Find all Config usages
rg "Config::" bin/00_rbee_keeper/ --type rust

# Find config file operations
rg "config.toml" bin/00_rbee_keeper/ --type rust
```

---

## Step 3: Implement KeeperConfig (config.rs)

### Source to Copy From

**Location:** `bin/00_rbee_keeper/src/config.rs`

### Implementation

**File:** `src/config.rs`

```rust
//! Keeper configuration types
//!
//! TEAM-314: Extracted from rbee-keeper for stability

use serde::{Deserialize, Serialize};

/// Keeper configuration
///
/// Loaded from ~/.config/rbee/config.toml
///
/// # Example
///
/// ```rust
/// use keeper_config_contract::KeeperConfig;
///
/// let config = KeeperConfig::default();
/// assert_eq!(config.queen_port, 7833);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeeperConfig {
    /// Queen HTTP port
    #[serde(default = "default_queen_port")]
    pub queen_port: u16,
}

fn default_queen_port() -> u16 {
    7833
}

impl Default for KeeperConfig {
    fn default() -> Self {
        Self {
            queen_port: default_queen_port(),
        }
    }
}

impl KeeperConfig {
    /// Create new config with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Get queen URL based on configured port
    pub fn queen_url(&self) -> String {
        format!("http://localhost:{}", self.queen_port)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Port must be in valid range
        if self.queen_port < 1024 {
            return Err(ValidationError::InvalidPort {
                port: self.queen_port,
                reason: "Port must be >= 1024 (non-privileged)".to_string(),
            });
        }

        if self.queen_port > 65535 {
            return Err(ValidationError::InvalidPort {
                port: self.queen_port,
                reason: "Port must be <= 65535".to_string(),
            });
        }

        Ok(())
    }

    /// Parse from TOML string
    pub fn from_toml(toml_str: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(toml_str)
    }

    /// Serialize to TOML string
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }
}

/// Configuration validation error
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    /// Invalid port number
    #[error("Invalid port {port}: {reason}")]
    InvalidPort {
        /// The invalid port number
        port: u16,
        /// Reason why it's invalid
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = KeeperConfig::default();
        assert_eq!(config.queen_port, 7833);
    }

    #[test]
    fn test_queen_url() {
        let config = KeeperConfig {
            queen_port: 8080,
        };
        assert_eq!(config.queen_url(), "http://localhost:8080");
    }

    #[test]
    fn test_validation_valid() {
        let config = KeeperConfig {
            queen_port: 7833,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validation_port_too_low() {
        let config = KeeperConfig {
            queen_port: 80,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_toml_roundtrip() {
        let config = KeeperConfig {
            queen_port: 7833,
        };
        
        let toml_str = config.to_toml().unwrap();
        let parsed = KeeperConfig::from_toml(&toml_str).unwrap();
        
        assert_eq!(config, parsed);
    }

    #[test]
    fn test_toml_with_defaults() {
        let toml_str = "";  // Empty config should use defaults
        let config = KeeperConfig::from_toml(toml_str).unwrap();
        assert_eq!(config.queen_port, 7833);
    }
}
```

---

## Step 4: Main Library File (lib.rs)

**File:** `src/lib.rs`

```rust
//! keeper-config-contract
//!
//! TEAM-314: Keeper configuration contract
//!
//! # Purpose
//!
//! This crate provides the configuration schema for rbee-keeper.
//! It ensures configuration stability across versions.
//!
//! # Components
//!
//! - **KeeperConfig** - Main configuration type
//! - **ValidationError** - Configuration validation errors

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Configuration types
pub mod config;

// Re-export main types
pub use config::{KeeperConfig, ValidationError};
```

---

## Step 5: Create README.md

**File:** `README.md`

```markdown
# keeper-config-contract

Keeper configuration contract for the rbee ecosystem.

## Purpose

This crate provides the configuration schema for rbee-keeper, ensuring stability across versions.

## Configuration

Configuration is loaded from `~/.config/rbee/config.toml`:

\`\`\`toml
queen_port = 7833
\`\`\`

## Usage

\`\`\`rust
use keeper_config_contract::KeeperConfig;

// Create default config
let config = KeeperConfig::default();

// Get queen URL
let url = config.queen_url();
println!("Queen URL: {}", url);

// Validate config
config.validate()?;

// Parse from TOML
let toml_str = r#"
queen_port = 8080
"#;
let config = KeeperConfig::from_toml(toml_str)?;

// Serialize to TOML
let toml_str = config.to_toml()?;
\`\`\`

## Validation

Configuration is validated to ensure:
- Port is in valid range (1024-65535)
- Port is not privileged (< 1024)

## Add to Cargo.toml

\`\`\`toml
[dependencies]
keeper-config-contract = { path = "../97_contracts/keeper-config-contract" }
\`\`\`

## License

GPL-3.0-or-later
```

---

## Step 6: Migration Guide

### Update rbee-keeper

**File:** `bin/00_rbee_keeper/src/config.rs`

**Before:**
```rust
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_queen_port")]
    pub queen_port: u16,
}

fn default_queen_port() -> u16 {
    7833
}

impl Config {
    pub fn load() -> Result<Self> {
        // ... load logic
    }

    pub fn save(&self) -> Result<()> {
        // ... save logic
    }

    pub fn queen_url(&self) -> String {
        format!("http://localhost:{}", self.queen_port)
    }
}
```

**After:**
```rust
// TEAM-314: Use KeeperConfig from contract
use keeper_config_contract::KeeperConfig;

// Keep Config as alias for compatibility
pub type Config = KeeperConfig;

// Add I/O operations (not in contract)
impl Config {
    /// Load config from ~/.config/rbee/config.toml
    pub fn load() -> Result<Self> {
        let config_path = Self::config_path()?;

        if !config_path.exists() {
            let config = Self::default();
            config.save()?;
            return Ok(config);
        }

        let contents = fs::read_to_string(&config_path)?;
        let config = KeeperConfig::from_toml(&contents)?;
        
        // Validate
        config.validate()?;

        Ok(config)
    }

    /// Save config to ~/.config/rbee/config.toml
    pub fn save(&self) -> Result<()> {
        let config_path = Self::config_path()?;

        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let contents = self.to_toml()?;
        fs::write(&config_path, contents)?;

        Ok(())
    }

    /// Get config file path
    fn config_path() -> Result<PathBuf> {
        let config_dir = dirs::config_dir()
            .context("Failed to get config directory")?;
        Ok(config_dir.join("rbee").join("config.toml"))
    }
}
```

---

## Step 7: Testing

```bash
# Build the contract
cd bin/97_contracts/keeper-config-contract
cargo build

# Run tests
cargo test

# Check documentation
cargo doc --open

# Test with rbee-keeper
cd ../../00_rbee_keeper
cargo build

# Full integration test
cd ../../
cargo build --bin rbee-keeper
```

---

## Step 8: Add thiserror Dependency

The contract uses `thiserror` for error types. Add to Cargo.toml:

```toml
[dependencies]
thiserror = "1.0"
```

---

## Verification Checklist

- [ ] keeper-config-contract crate created
- [ ] KeeperConfig implemented
- [ ] Validation logic added
- [ ] TOML serialization works
- [ ] All tests pass
- [ ] Documentation complete
- [ ] rbee-keeper updated
- [ ] Config loading works
- [ ] Full build passes

---

## Future Enhancements

### Add More Config Options

```rust
pub struct KeeperConfig {
    pub queen_port: u16,
    
    // Future additions
    pub hive_default_port: Option<u16>,
    pub log_level: Option<String>,
    pub data_dir: Option<PathBuf>,
}
```

### Add Config Migrations

```rust
impl KeeperConfig {
    pub fn migrate_from_v0(old: ConfigV0) -> Self {
        // Migration logic
    }
}
```

### Add Environment Variable Override

```rust
impl KeeperConfig {
    pub fn with_env_overrides(mut self) -> Self {
        if let Ok(port) = std::env::var("RBEE_QUEEN_PORT") {
            self.queen_port = port.parse().unwrap_or(self.queen_port);
        }
        self
    }
}
```

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27  
**Status:** PLAN 📋
