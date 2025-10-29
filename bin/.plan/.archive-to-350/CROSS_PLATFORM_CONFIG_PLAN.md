# Cross-Platform Config Directory Plan

**Issue:** rbee-config currently hardcodes `~/.config/rbee/` which is Linux-specific  
**Goal:** Support Linux, Mac, and Windows with platform-appropriate directories  
**Estimated Effort:** 4-6 hours

---

## 🎯 Problem

Current implementation in `rbee-config/src/lib.rs`:

```rust
pub fn config_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").map_err(|_| {
        ConfigError::InvalidConfig("HOME environment variable not set".to_string())
    })?;

    let config_dir = PathBuf::from(home).join(".config").join("rbee");
    // ...
}
```

**Issues:**
1. ❌ Uses `$HOME` which doesn't exist on Windows (`%USERPROFILE%` instead)
2. ❌ Uses `.config/` which is Linux XDG standard, not Mac or Windows
3. ❌ Doesn't follow platform conventions

---

## 📁 Platform-Specific Directories

### Linux (XDG Base Directory)
- **Config:** `~/.config/rbee/`
- **Cache:** `~/.cache/rbee/`
- **Data:** `~/.local/share/rbee/`

### macOS (Apple Guidelines)
- **Config:** `~/Library/Application Support/rbee/`
- **Cache:** `~/Library/Caches/rbee/`
- **Data:** `~/Library/Application Support/rbee/`

### Windows (Known Folders)
- **Config:** `%APPDATA%\rbee\` (e.g., `C:\Users\vince\AppData\Roaming\rbee\`)
- **Cache:** `%LOCALAPPDATA%\rbee\` (e.g., `C:\Users\vince\AppData\Local\rbee\`)
- **Data:** `%APPDATA%\rbee\`

---

## 🛠️ Solution: Use `dirs` Crate

The `dirs` crate provides cross-platform directory resolution:

```rust
use dirs;

// Config directory
dirs::config_dir()     // Linux: ~/.config, Mac: ~/Library/Application Support, Windows: %APPDATA%

// Cache directory
dirs::cache_dir()      // Linux: ~/.cache, Mac: ~/Library/Caches, Windows: %LOCALAPPDATA%

// Data directory
dirs::data_local_dir() // Linux: ~/.local/share, Mac: ~/Library/Application Support, Windows: %LOCALAPPDATA%
```

---

## 📝 Implementation Plan

### Step 1: Update Dependencies

**File:** `bin/99_shared_crates/rbee-config/Cargo.toml`

```toml
[dependencies]
# ... existing dependencies ...
dirs = "5.0"  # Add this
```

### Step 2: Update config_dir() Function

**File:** `bin/99_shared_crates/rbee-config/src/lib.rs`

**Replace:**
```rust
pub fn config_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").map_err(|_| {
        ConfigError::InvalidConfig("HOME environment variable not set".to_string())
    })?;

    let config_dir = PathBuf::from(home).join(".config").join("rbee");

    // Create directory if it doesn't exist
    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir)?;
    }

    Ok(config_dir)
}
```

**With:**
```rust
/// Get config directory path (cross-platform)
///
/// Returns:
/// - Linux: ~/.config/rbee/
/// - macOS: ~/Library/Application Support/rbee/
/// - Windows: %APPDATA%\rbee\
pub fn config_dir() -> Result<PathBuf> {
    let base_config_dir = dirs::config_dir()
        .ok_or_else(|| ConfigError::InvalidConfig(
            "Cannot determine config directory for this platform".to_string()
        ))?;

    let config_dir = base_config_dir.join("rbee");

    // Create directory if it doesn't exist
    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir)?;
    }

    Ok(config_dir)
}
```

### Step 3: Add Helper Functions for Cache and Data

**File:** `bin/99_shared_crates/rbee-config/src/lib.rs`

Add these new functions:

```rust
/// Get cache directory path (cross-platform)
///
/// Returns:
/// - Linux: ~/.cache/rbee/
/// - macOS: ~/Library/Caches/rbee/
/// - Windows: %LOCALAPPDATA%\rbee\
pub fn cache_dir() -> Result<PathBuf> {
    let base_cache_dir = dirs::cache_dir()
        .ok_or_else(|| ConfigError::InvalidConfig(
            "Cannot determine cache directory for this platform".to_string()
        ))?;

    let cache_dir = base_cache_dir.join("rbee");

    // Create directory if it doesn't exist
    if !cache_dir.exists() {
        std::fs::create_dir_all(&cache_dir)?;
    }

    Ok(cache_dir)
}

/// Get data directory path (cross-platform)
///
/// Returns:
/// - Linux: ~/.local/share/rbee/
/// - macOS: ~/Library/Application Support/rbee/
/// - Windows: %LOCALAPPDATA%\rbee\
pub fn data_dir() -> Result<PathBuf> {
    let base_data_dir = dirs::data_local_dir()
        .ok_or_else(|| ConfigError::InvalidConfig(
            "Cannot determine data directory for this platform".to_string()
        ))?;

    let data_dir = base_data_dir.join("rbee");

    // Create directory if it doesn't exist
    if !data_dir.exists() {
        std::fs::create_dir_all(&data_dir)?;
    }

    Ok(data_dir)
}
```

### Step 4: Update Public API

**File:** `bin/99_shared_crates/rbee-config/src/lib.rs`

Add to public exports:

```rust
pub use capabilities::{CapabilitiesCache, DeviceInfo, DeviceType, HiveCapabilities};
pub use error::{ConfigError, Result};
pub use hives_config::{HiveEntry, HivesConfig};
pub use queen_config::{QueenConfig, QueenSettings, RuntimeSettings};
pub use validation::{
    preflight_validation, validate_capabilities_sync, validate_hives_config, ValidationResult,
};

// Add these:
pub use config_dir;
pub use cache_dir;
pub use data_dir;
```

### Step 5: Update Documentation

**File:** `bin/99_shared_crates/rbee-config/src/lib.rs`

Update the module documentation:

```rust
//! File-based configuration for rbee
//!
//! Created by: TEAM-193
//! Updated by: TEAM-XXX (cross-platform support)
//!
//! This crate provides file-based configuration management following platform conventions.
//!
//! # Configuration Files
//!
//! Config files are stored in platform-specific directories:
//!
//! **Linux:**
//! - Config: `~/.config/rbee/`
//! - Cache: `~/.cache/rbee/`
//! - Data: `~/.local/share/rbee/`
//!
//! **macOS:**
//! - Config: `~/Library/Application Support/rbee/`
//! - Cache: `~/Library/Caches/rbee/`
//! - Data: `~/Library/Application Support/rbee/`
//!
//! **Windows:**
//! - Config: `%APPDATA%\rbee\`
//! - Cache: `%LOCALAPPDATA%\rbee\`
//! - Data: `%LOCALAPPDATA%\rbee\`
//!
//! Files in config directory:
//! - `config.toml` - Queen-level settings (port, log level, etc.)
//! - `hives.conf` - SSH/hive definitions (SSH config style)
//! - `capabilities.yaml` - Auto-generated device capabilities cache
```

### Step 6: Update Tests

**File:** `bin/99_shared_crates/rbee-config/src/lib.rs`

Update the test that checks config directory:

```rust
#[test]
fn test_config_dir_creation() {
    // This test now works cross-platform
    let config_dir = RbeeConfig::config_dir().unwrap();
    assert!(config_dir.exists());
    assert!(config_dir.ends_with("rbee"));
    
    // Verify it's in the right base directory for the platform
    #[cfg(target_os = "linux")]
    assert!(config_dir.to_str().unwrap().contains(".config"));
    
    #[cfg(target_os = "macos")]
    assert!(config_dir.to_str().unwrap().contains("Library/Application Support"));
    
    #[cfg(target_os = "windows")]
    assert!(config_dir.to_str().unwrap().contains("AppData\\Roaming"));
}

#[test]
fn test_cache_dir_creation() {
    let cache_dir = cache_dir().unwrap();
    assert!(cache_dir.exists());
    assert!(cache_dir.ends_with("rbee"));
}

#[test]
fn test_data_dir_creation() {
    let data_dir = data_dir().unwrap();
    assert!(data_dir.exists());
    assert!(data_dir.ends_with("rbee"));
}
```

---

## 🔄 Impact on Other Components

### 1. Model Catalog (Already Updated!)

The model catalog in `STORAGE_ARCHITECTURE.md` already uses `dirs::cache_dir()`:

```rust
let cache_dir = dirs::cache_dir()
    .ok_or_else(|| anyhow!("Cannot determine cache directory"))?
    .join("rbee")
    .join("models");
```

**Result:**
- Linux: `~/.cache/rbee/models/`
- macOS: `~/Library/Caches/rbee/models/`
- Windows: `%LOCALAPPDATA%\rbee\models\`

✅ **No changes needed** - already cross-platform!

### 2. Other Components Using Config

Any component that uses `RbeeConfig::config_dir()` will automatically become cross-platform:

```rust
use rbee_config::RbeeConfig;

// This now works on all platforms
let config_dir = RbeeConfig::config_dir()?;
let config_file = config_dir.join("config.toml");
```

---

## 📊 Directory Layout (All Platforms)

### Linux
```
~/.config/rbee/
├── config.toml
├── hives.conf
└── capabilities.yaml

~/.cache/rbee/
└── models/
    └── meta-llama/
        └── Llama-2-7b-chat-hf/
            ├── metadata.yaml
            └── model.safetensors
```

### macOS
```
~/Library/Application Support/rbee/
├── config.toml
├── hives.conf
└── capabilities.yaml

~/Library/Caches/rbee/
└── models/
    └── meta-llama/
        └── Llama-2-7b-chat-hf/
            ├── metadata.yaml
            └── model.safetensors
```

### Windows
```
C:\Users\vince\AppData\Roaming\rbee\
├── config.toml
├── hives.conf
└── capabilities.yaml

C:\Users\vince\AppData\Local\rbee\
└── models\
    └── meta-llama\
        └── Llama-2-7b-chat-hf\
            ├── metadata.yaml
            └── model.safetensors
```

---

## ✅ Acceptance Criteria

- [ ] `dirs` crate added to dependencies
- [ ] `config_dir()` uses `dirs::config_dir()`
- [ ] `cache_dir()` function added
- [ ] `data_dir()` function added
- [ ] Documentation updated with platform-specific paths
- [ ] Tests updated for cross-platform
- [ ] Compilation succeeds on Linux, macOS, Windows
- [ ] Tests pass on all platforms
- [ ] No hardcoded `$HOME` or `.config` references

---

## 🧪 Testing Strategy

### Manual Testing

**Linux:**
```bash
cargo test --package rbee-config
ls -la ~/.config/rbee/
ls -la ~/.cache/rbee/
```

**macOS:**
```bash
cargo test --package rbee-config
ls -la ~/Library/Application\ Support/rbee/
ls -la ~/Library/Caches/rbee/
```

**Windows (PowerShell):**
```powershell
cargo test --package rbee-config
dir $env:APPDATA\rbee\
dir $env:LOCALAPPDATA\rbee\
```

### CI Testing

Add to `.github/workflows/`:

```yaml
name: Cross-Platform Config Test

on: [push, pull_request]

jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v3
      - uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      - name: Test rbee-config
        run: cargo test --package rbee-config
```

---

## 🚨 Breaking Changes

### Potential Issues

1. **Existing Linux users** will need to migrate config files:
   ```bash
   # Old location (still works on Linux)
   ~/.config/rbee/
   
   # New location (same on Linux!)
   ~/.config/rbee/
   ```
   ✅ **No migration needed for Linux users!**

2. **Documentation** needs to be updated to show platform-specific paths

3. **Installation scripts** should create directories in the right locations

### Migration Guide (If Needed)

For users who manually created config in non-standard locations:

```bash
# Linux/Mac: No migration needed
# Windows: Config will be created automatically in %APPDATA%\rbee\
```

---

## 📝 Documentation Updates

### Files to Update

1. **README.md** - Add platform-specific directory information
2. **STORAGE_ARCHITECTURE.md** - Already mentions cross-platform (✅)
3. **rbee-config/README.md** - Update with platform paths
4. **Installation guides** - Add platform-specific instructions

### Example Documentation

```markdown
## Configuration Directories

rbee uses platform-appropriate directories for configuration and cache:

| Platform | Config | Cache |
|----------|--------|-------|
| Linux | `~/.config/rbee/` | `~/.cache/rbee/` |
| macOS | `~/Library/Application Support/rbee/` | `~/Library/Caches/rbee/` |
| Windows | `%APPDATA%\rbee\` | `%LOCALAPPDATA%\rbee\` |

All directories are created automatically on first run.
```

---

## 🎯 Implementation Checklist

### Phase 1: Core Changes (2-3 hours)
- [ ] Add `dirs` crate to Cargo.toml
- [ ] Update `config_dir()` function
- [ ] Add `cache_dir()` function
- [ ] Add `data_dir()` function
- [ ] Update public API exports

### Phase 2: Testing (1-2 hours)
- [ ] Update existing tests
- [ ] Add platform-specific tests
- [ ] Test on Linux
- [ ] Test on macOS (if available)
- [ ] Test on Windows (if available)

### Phase 3: Documentation (1 hour)
- [ ] Update module documentation
- [ ] Update README.md
- [ ] Add migration guide (if needed)
- [ ] Update installation docs

---

## 🔮 Future Enhancements

### 1. Environment Variable Override

Allow users to override directories:

```rust
pub fn config_dir() -> Result<PathBuf> {
    // Check for override first
    if let Ok(override_dir) = std::env::var("RBEE_CONFIG_DIR") {
        return Ok(PathBuf::from(override_dir));
    }
    
    // Fall back to platform default
    let base_config_dir = dirs::config_dir()
        .ok_or_else(|| ConfigError::InvalidConfig(
            "Cannot determine config directory for this platform".to_string()
        ))?;

    Ok(base_config_dir.join("rbee"))
}
```

### 2. Portable Mode

Support running from a USB drive:

```rust
pub fn config_dir() -> Result<PathBuf> {
    // Check for portable mode
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let portable_config = exe_dir.join("config");
            if portable_config.join(".portable").exists() {
                return Ok(portable_config);
            }
        }
    }
    
    // Fall back to platform default
    // ...
}
```

---

## 📊 Estimated Effort

| Task | Time | Complexity |
|------|------|------------|
| Add `dirs` dependency | 5 min | Easy |
| Update `config_dir()` | 15 min | Easy |
| Add `cache_dir()` and `data_dir()` | 15 min | Easy |
| Update tests | 30 min | Medium |
| Test on Linux | 15 min | Easy |
| Test on macOS | 30 min | Medium |
| Test on Windows | 30 min | Medium |
| Update documentation | 45 min | Easy |
| **Total** | **3-4 hours** | **Easy-Medium** |

---

## ✅ Success Criteria

After implementation:

- ✅ Works on Linux, macOS, and Windows
- ✅ Uses platform-appropriate directories
- ✅ No hardcoded paths
- ✅ Tests pass on all platforms
- ✅ Documentation is clear and accurate
- ✅ No breaking changes for existing users
- ✅ Easy to test and verify

---

## 🚀 Ready to Implement

This plan provides:
- ✅ Clear problem statement
- ✅ Platform-specific directory mappings
- ✅ Step-by-step implementation guide
- ✅ Testing strategy
- ✅ Documentation updates
- ✅ Migration considerations

**Assign to:** TEAM-276 or integrate into existing work  
**Priority:** Medium (improves cross-platform support)  
**Effort:** 3-4 hours

---

**This makes rbee truly cross-platform! 🐝**
