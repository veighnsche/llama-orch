# Phase 3: Preflight Validation

**Team:** TEAM-190  
**Duration:** 3-4 hours  
**Dependencies:** Phase 2 (TEAM-189) complete  
**Deliverables:** Queen startup validation, alias uniqueness checks

---

## Mission

Add comprehensive preflight validation to ensure config files are valid before queen-rbee starts, and validate operations before execution.

---

## Tasks

### 3.1 Queen Startup Validation

**Location:** `bin/00_rbee_keeper/src/queen_lifecycle.rs`

**Add validation before starting queen:**

```rust
pub async fn start_queen(
    queen_port: u16,
    hive_catalog_path: Option<PathBuf>,
) -> Result<()> {
    use rbee_config::RbeeConfig;

    NARRATE_KEEPER
        .action("queen_startup_config_loading")
        .human("📋 Loading rbee configuration...")
        .emit();

    // STEP 1: Load config
    let config = RbeeConfig::load().context("Failed to load rbee config")?;

    NARRATE_KEEPER
        .action("queen_startup_config_loaded")
        .human(format!(
            "✅ Config loaded from {}",
            RbeeConfig::config_dir().display()
        ))
        .emit();

    // STEP 2: Validate config
    NARRATE_KEEPER
        .action("queen_startup_validation")
        .human("🔍 Validating configuration...")
        .emit();

    if let Err(e) = config.validate() {
        Narration::new(ACTOR_KEEPER, "queen_startup_validation_failed", "config")
            .human(format!(
                "❌ Configuration validation failed:\n\
                 \n\
                 {}\n\
                 \n\
                 Please fix the errors in ~/.config/rbee/ and try again.",
                e
            ))
            .emit();
        return Err(e);
    }

    // STEP 3: Validate hives.conf
    NARRATE_KEEPER
        .action("queen_startup_hives_validation")
        .human("🔍 Validating hives.conf...")
        .emit();

    if let Err(e) = config.hives.validate_unique_aliases() {
        Narration::new(ACTOR_KEEPER, "queen_startup_duplicate_aliases", "hives")
            .human(format!(
                "❌ Duplicate hive aliases detected:\n\
                 \n\
                 {}\n\
                 \n\
                 Each Host entry in ~/.config/rbee/hives.conf must have a unique alias.",
                e
            ))
            .emit();
        return Err(e);
    }

    let hive_count = config.hives.all().len();
    NARRATE_KEEPER
        .action("queen_startup_hives_validated")
        .human(format!("✅ {} hive(s) configured", hive_count))
        .emit();

    // STEP 4: Check capabilities.yaml
    let caps_count = config.capabilities.all().len();
    if caps_count > 0 {
        NARRATE_KEEPER
            .action("queen_startup_capabilities_loaded")
            .human(format!("📊 {} hive(s) have cached capabilities", caps_count))
            .emit();
    } else {
        NARRATE_KEEPER
            .action("queen_startup_capabilities_empty")
            .human("⚠️  No cached capabilities found (hives not yet started)")
            .emit();
    }

    // STEP 5: Validate config directory permissions
    let config_dir = RbeeConfig::config_dir();
    if !config_dir.exists() {
        NARRATE_KEEPER
            .action("queen_startup_config_dir_create")
            .human(format!("📁 Creating config directory: {}", config_dir.display()))
            .emit();
        
        std::fs::create_dir_all(&config_dir)
            .context("Failed to create config directory")?;
    }

    // Check write permissions
    if let Err(e) = std::fs::metadata(&config_dir) {
        Narration::new(ACTOR_KEEPER, "queen_startup_config_dir_error", "permissions")
            .human(format!(
                "❌ Cannot access config directory: {}\n\
                 \n\
                 Error: {}",
                config_dir.display(),
                e
            ))
            .emit();
        return Err(e.into());
    }

    NARRATE_KEEPER
        .action("queen_startup_validation_complete")
        .human("✅ All preflight checks passed")
        .emit();

    // Continue with normal queen startup...
    Narration::new(ACTOR_KEEPER, "queen_starting", "daemon")
        .human(format!("🚀 Starting queen-rbee on port {}", queen_port))
        .emit();

    // ... rest of existing startup code
}
```

### 3.2 Add Validation Methods to rbee-config

**In `bin/15_queen_rbee_crates/rbee-config/src/validation.rs`:**

```rust
use crate::{RbeeConfig, HivesConfig, ConfigError};
use std::collections::HashSet;

impl RbeeConfig {
    /// Validate all config files
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate hives
        self.hives.validate_unique_aliases()?;
        self.hives.validate_required_fields()?;
        
        // Validate queen config
        self.queen.validate()?;
        
        Ok(())
    }
}

impl HivesConfig {
    /// Validate that all aliases are unique
    pub fn validate_unique_aliases(&self) -> Result<(), ConfigError> {
        let mut seen = HashSet::new();
        let mut duplicates = Vec::new();
        
        for (alias, _) in &self.hives {
            if !seen.insert(alias) {
                duplicates.push(alias.clone());
            }
        }
        
        if !duplicates.is_empty() {
            return Err(ConfigError::DuplicateAliases {
                aliases: duplicates,
            });
        }
        
        Ok(())
    }
    
    /// Validate that all required fields are present
    pub fn validate_required_fields(&self) -> Result<(), ConfigError> {
        let mut errors = Vec::new();
        
        for (alias, hive) in &self.hives {
            // Check required fields
            if hive.hostname.is_empty() {
                errors.push(format!("Host '{}': missing HostName", alias));
            }
            if hive.ssh_user.is_empty() {
                errors.push(format!("Host '{}': missing User", alias));
            }
            if hive.ssh_port == 0 {
                errors.push(format!("Host '{}': invalid Port (must be > 0)", alias));
            }
            if hive.hive_port == 0 {
                errors.push(format!("Host '{}': missing HivePort", alias));
            }
            
            // Validate port ranges
            if hive.ssh_port > 65535 {
                errors.push(format!("Host '{}': invalid Port (must be <= 65535)", alias));
            }
            if hive.hive_port > 65535 {
                errors.push(format!("Host '{}': invalid HivePort (must be <= 65535)", alias));
            }
        }
        
        if !errors.is_empty() {
            return Err(ConfigError::InvalidFields {
                errors,
            });
        }
        
        Ok(())
    }
}

impl QueenConfig {
    /// Validate queen configuration
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.port == 0 || self.port > 65535 {
            return Err(ConfigError::InvalidQueenPort {
                port: self.port,
            });
        }
        
        Ok(())
    }
}
```

### 3.3 Update Error Types

**In `bin/15_queen_rbee_crates/rbee-config/src/error.rs`:**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Duplicate hive aliases found: {}", aliases.join(", "))]
    DuplicateAliases {
        aliases: Vec<String>,
    },
    
    #[error("Invalid fields in hives.conf:\n{}", errors.join("\n"))]
    InvalidFields {
        errors: Vec<String>,
    },
    
    #[error("Invalid queen port: {port} (must be 1-65535)")]
    InvalidQueenPort {
        port: u16,
    },
    
    #[error("Config file not found: {path}")]
    FileNotFound {
        path: String,
    },
    
    #[error("Failed to parse config file: {0}")]
    ParseError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("YAML error: {0}")]
    YamlError(#[from] serde_yaml::Error),
    
    #[error("TOML error: {0}")]
    TomlError(#[from] toml::de::Error),
}
```

### 3.4 Operation-Level Validation

**In `bin/10_queen_rbee/src/job_router.rs`:**

Add validation helper at the top of the file:

```rust
/// Validate that a hive alias exists in config
fn validate_hive_exists(
    config: &RbeeConfig,
    alias: &str,
) -> Result<&HiveEntry, anyhow::Error> {
    config.hives.get(alias).ok_or_else(|| {
        anyhow::anyhow!(
            "Hive alias '{}' not found in hives.conf.\n\
             \n\
             Available hives:\n\
             {}\n\
             \n\
             Add '{}' to ~/.config/rbee/hives.conf to use it.",
            alias,
            config.hives.all()
                .iter()
                .map(|h| format!("  - {}", h.alias))
                .collect::<Vec<_>>()
                .join("\n"),
            alias
        )
    })
}
```

Use in operations:

```rust
Operation::HiveInstall { alias } => {
    // Validate alias exists
    let hive = validate_hive_exists(&state.config, &alias)?;
    
    // Continue with installation...
}

Operation::HiveStart { alias } => {
    let hive = validate_hive_exists(&state.config, &alias)?;
    // ...
}

Operation::HiveStop { alias } => {
    let hive = validate_hive_exists(&state.config, &alias)?;
    // ...
}

Operation::SshTest { alias } => {
    let hive = validate_hive_exists(&state.config, &alias)?;
    // ...
}
```

### 3.5 Add Config Reload Command

**Optional enhancement:**

Add ability to reload config without restarting queen:

```rust
Operation::ConfigReload {} => {
    Narration::new(ACTOR_QUEEN_ROUTER, "config_reload", "all")
        .human("🔄 Reloading configuration...")
        .emit();
    
    let new_config = RbeeConfig::load()
        .context("Failed to reload config")?;
    
    // Validate before applying
    new_config.validate()
        .context("Config validation failed")?;
    
    // Replace config in state
    *state.config = new_config;
    
    Narration::new(ACTOR_QUEEN_ROUTER, "config_reload_complete", "all")
        .human("✅ Configuration reloaded successfully")
        .emit();
}
```

### 3.6 Write Validation Tests

**In `bin/15_queen_rbee_crates/rbee-config/tests/validation_tests.rs`:**

```rust
#[test]
fn test_duplicate_aliases_detected() {
    let config_content = r#"
Host localhost
    HostName 127.0.0.1
    Port 22
    User vince
    HivePort 8081

Host localhost
    HostName 192.168.1.100
    Port 22
    User admin
    HivePort 8082
"#;
    
    let hives = HivesConfig::parse(config_content).unwrap();
    let result = hives.validate_unique_aliases();
    
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("localhost"));
}

#[test]
fn test_missing_required_fields() {
    let config_content = r#"
Host incomplete
    HostName 192.168.1.100
    # Missing: Port, User, HivePort
"#;
    
    let hives = HivesConfig::parse(config_content).unwrap();
    let result = hives.validate_required_fields();
    
    assert!(result.is_err());
}

#[test]
fn test_invalid_port_ranges() {
    let config_content = r#"
Host invalid
    HostName 192.168.1.100
    Port 99999
    User admin
    HivePort 8081
"#;
    
    let hives = HivesConfig::parse(config_content).unwrap();
    let result = hives.validate_required_fields();
    
    assert!(result.is_err());
}
```

---

## Acceptance Criteria

- [ ] Queen startup validates config before starting
- [ ] Duplicate aliases are detected and reported
- [ ] Missing required fields are detected
- [ ] Invalid port ranges are rejected
- [ ] Config directory is created if missing
- [ ] Clear error messages guide users to fix issues
- [ ] All validation tests pass

---

## Verification Commands

```bash
# Test validation
cargo test -p rbee-config validation

# Test queen startup with invalid config
# (manually create invalid hives.conf)
cargo run --bin rbee-keeper -- queen start

# Test with valid config
cargo run --bin rbee-keeper -- queen start
```

---

## Handoff to TEAM-191

**What's ready:**
- ✅ Preflight validation on queen startup
- ✅ Alias uniqueness checks
- ✅ Required field validation
- ✅ Operation-level validation helpers

**Next steps:**
- Implement capabilities auto-generation
- Update capabilities.yaml when hives start
- Add capabilities refresh command

---

**Created by:** TEAM-187  
**For:** TEAM-190  
**Status:** 📋 Ready to implement
