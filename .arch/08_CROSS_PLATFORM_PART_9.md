# rbee Cross-Platform Architecture

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Implementation Plan

---

## Overview

rbee is designed to work seamlessly on **Linux, macOS, and Windows** with platform-appropriate directory structures and conventions.

---

## Platform Support Matrix

| Feature | Linux | macOS | Windows | Status |
|---------|-------|-------|---------|--------|
| **Config Files** | ✅ | ✅ | ✅ | Planned |
| **Model Storage** | ✅ | ✅ | ✅ | Planned |
| **Worker Binaries** | ✅ | ✅ | ✅ | Planned |
| **SSH Connections** | ✅ | ✅ | ✅ | Implemented |
| **GPU Detection** | ✅ | ✅ | ⚠️ | Partial |

**Legend:**
- ✅ Fully supported
- ⚠️ Partial support (CUDA only, no Metal/DirectML yet)
- ❌ Not supported

---

## Directory Structure

### Configuration Directories

**Purpose:** Store user-editable configuration files

| Platform | Location | Example |
|----------|----------|---------|
| **Linux** | `~/.config/rbee/` | `/home/vince/.config/rbee/` |
| **macOS** | `~/Library/Application Support/rbee/` | `/Users/vince/Library/Application Support/rbee/` |
| **Windows** | `%APPDATA%\rbee\` | `C:\Users\vince\AppData\Roaming\rbee\` |

**Files:**
```
config.toml           # Queen settings (port, bind address, etc.)
hives.conf            # Hive definitions (SSH config style)
capabilities.yaml     # Auto-generated device capabilities cache
```

### Cache Directories

**Purpose:** Store downloaded models and temporary data

| Platform | Location | Example |
|----------|----------|---------|
| **Linux** | `~/.cache/rbee/` | `/home/vince/.cache/rbee/` |
| **macOS** | `~/Library/Caches/rbee/` | `/Users/vince/Library/Caches/rbee/` |
| **Windows** | `%LOCALAPPDATA%\rbee\` | `C:\Users\vince\AppData\Local\rbee\` |

**Structure:**
```
models/
├── meta-llama/
│   └── Llama-2-7b-chat-hf/
│       ├── metadata.yaml
│       ├── model.safetensors
│       └── config.json
└── mistralai/
    └── Mistral-7B-Instruct-v0.2/
        ├── metadata.yaml
        └── model.safetensors
```

### Data Directories

**Purpose:** Store persistent application data (logs, databases, etc.)

| Platform | Location | Example |
|----------|----------|---------|
| **Linux** | `~/.local/share/rbee/` | `/home/vince/.local/share/rbee/` |
| **macOS** | `~/Library/Application Support/rbee/` | `/Users/vince/Library/Application Support/rbee/` |
| **Windows** | `%LOCALAPPDATA%\rbee\` | `C:\Users\vince\AppData\Local\rbee\` |

---

## Implementation Strategy

### Using the `dirs` Crate

**Dependency:**
```toml
[dependencies]
dirs = "5.0"
```

**API:**
```rust
use dirs;

// Config directory
let config_dir = dirs::config_dir()
    .ok_or_else(|| anyhow!("Cannot determine config directory"))?
    .join("rbee");

// Cache directory
let cache_dir = dirs::cache_dir()
    .ok_or_else(|| anyhow!("Cannot determine cache directory"))?
    .join("rbee");

// Data directory
let data_dir = dirs::data_local_dir()
    .ok_or_else(|| anyhow!("Cannot determine data directory"))?
    .join("rbee");
```

### Cross-Platform Path Handling

**Always use PathBuf and join():**
```rust
// ✅ CORRECT (cross-platform)
let path = base_dir.join("rbee").join("models");

// ❌ WRONG (Linux-only)
let path = format!("{}/.config/rbee/models", home);
```

**Always use platform separators:**
```rust
// ✅ CORRECT (uses platform separator)
let path = PathBuf::from("rbee").join("models").join("llama");

// ❌ WRONG (hardcoded Unix separator)
let path = "rbee/models/llama";
```

---

## Component-Specific Implementations

### 1. rbee-config Crate

**Status:** ⚠️ Needs Update (currently Linux-only)

**Current Issue:**
```rust
// ❌ Linux-only
let home = std::env::var("HOME")?;
let config_dir = PathBuf::from(home).join(".config").join("rbee");
```

**Planned Fix:**
```rust
// ✅ Cross-platform
pub fn config_dir() -> Result<PathBuf> {
    let base = dirs::config_dir()
        .ok_or_else(|| ConfigError::InvalidConfig(
            "Cannot determine config directory".to_string()
        ))?;
    
    let config_dir = base.join("rbee");
    std::fs::create_dir_all(&config_dir)?;
    Ok(config_dir)
}
```

**See:** `bin/.plan/CROSS_PLATFORM_CONFIG_PLAN.md`

### 2. Model Catalog

**Status:** ✅ Already Cross-Platform

**Implementation:**
```rust
// Uses dirs::cache_dir() for cross-platform support
let models_dir = dirs::cache_dir()
    .ok_or_else(|| anyhow!("Cannot determine cache directory"))?
    .join("rbee")
    .join("models");
```

**See:** `bin/.plan/STORAGE_ARCHITECTURE.md`

### 3. Worker Binaries

**Status:** ✅ Cross-Platform Ready

**Location Strategy:**

**Development:**
```rust
// Debug build
let binary = if cfg!(debug_assertions) {
    PathBuf::from("./target/debug/llama-worker")
} else {
    PathBuf::from("./target/release/llama-worker")
};
```

**Production (search order):**
1. `./target/release/llama-worker` (local build)
2. `/usr/local/bin/llama-worker` (Linux/Mac system install)
3. `~/.local/bin/llama-worker` (Linux/Mac user install)
4. `%PROGRAMFILES%\rbee\llama-worker.exe` (Windows system install)
5. `%LOCALAPPDATA%\rbee\bin\llama-worker.exe` (Windows user install)
6. System PATH

### 4. SSH Client

**Status:** ✅ Cross-Platform (russh)

**Implementation:**
- Uses `russh` crate (pure Rust, no OpenSSH dependency)
- Works on Linux, macOS, Windows
- Reads SSH keys from standard locations:
  - Linux/Mac: `~/.ssh/id_ed25519`, `~/.ssh/id_rsa`
  - Windows: `%USERPROFILE%\.ssh\id_ed25519`, `%USERPROFILE%\.ssh\id_rsa`

---

## Platform-Specific Considerations

### Linux

**Standard Directories:**
- Config: `~/.config/` (XDG Base Directory Specification)
- Cache: `~/.cache/`
- Data: `~/.local/share/`

**Permissions:**
- Config files: `0644` (rw-r--r--)
- Directories: `0755` (rwxr-xr-x)
- Executables: `0755` (rwxr-xr-x)

**Package Managers:**
- Debian/Ubuntu: `.deb` packages
- Fedora/RHEL: `.rpm` packages
- Arch: AUR packages

### macOS

**Standard Directories:**
- Config: `~/Library/Application Support/` (Apple guidelines)
- Cache: `~/Library/Caches/`
- Data: `~/Library/Application Support/`

**Permissions:**
- Same as Linux (Unix-based)

**Distribution:**
- Homebrew: `brew install rbee`
- DMG installer
- App bundle (for GUI version)

**Code Signing:**
- Required for distribution
- Use Apple Developer ID

### Windows

**Standard Directories:**
- Config: `%APPDATA%\` (Roaming profile)
- Cache: `%LOCALAPPDATA%\` (Local profile)
- Data: `%LOCALAPPDATA%\`

**Paths:**
- Use backslashes in display: `C:\Users\vince\AppData\Roaming\rbee\`
- Use forward slashes or PathBuf internally

**Distribution:**
- MSI installer
- Chocolatey: `choco install rbee`
- Scoop: `scoop install rbee`
- WinGet: `winget install rbee`

**Permissions:**
- Windows uses ACLs, not Unix permissions
- Ensure user has read/write access to AppData

---

## Testing Strategy

### Unit Tests

**Test on all platforms:**
```rust
#[test]
fn test_config_dir_creation() {
    let config_dir = RbeeConfig::config_dir().unwrap();
    assert!(config_dir.exists());
    assert!(config_dir.ends_with("rbee"));
    
    // Platform-specific assertions
    #[cfg(target_os = "linux")]
    assert!(config_dir.to_str().unwrap().contains(".config"));
    
    #[cfg(target_os = "macos")]
    assert!(config_dir.to_str().unwrap().contains("Library/Application Support"));
    
    #[cfg(target_os = "windows")]
    assert!(config_dir.to_str().unwrap().contains("AppData\\Roaming"));
}
```

### Integration Tests

**Test matrix:**
- Linux (Ubuntu 22.04, Fedora 39, Arch)
- macOS (13 Ventura, 14 Sonoma)
- Windows (10, 11)

### CI/CD

**GitHub Actions:**
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
runs-on: ${{ matrix.os }}
```

---

## Migration Guide

### For Existing Linux Users

**No migration needed!** Directories stay the same:
- `~/.config/rbee/` → `~/.config/rbee/` ✅
- `~/.cache/rbee/` → `~/.cache/rbee/` ✅

### For New macOS Users

**Automatic setup:**
1. Install rbee
2. Run `rbee init`
3. Config created in `~/Library/Application Support/rbee/`
4. Models cached in `~/Library/Caches/rbee/`

### For New Windows Users

**Automatic setup:**
1. Install rbee
2. Run `rbee init`
3. Config created in `%APPDATA%\rbee\`
4. Models cached in `%LOCALAPPDATA%\rbee\`

---

## Environment Variable Overrides

**Optional:** Allow users to override directories

```bash
# Override config directory
export RBEE_CONFIG_DIR="/custom/path/to/config"

# Override cache directory
export RBEE_CACHE_DIR="/custom/path/to/cache"
```

**Implementation:**
```rust
pub fn config_dir() -> Result<PathBuf> {
    // Check for override first
    if let Ok(override_dir) = std::env::var("RBEE_CONFIG_DIR") {
        return Ok(PathBuf::from(override_dir));
    }
    
    // Fall back to platform default
    let base = dirs::config_dir()
        .ok_or_else(|| anyhow!("Cannot determine config directory"))?;
    
    Ok(base.join("rbee"))
}
```

---

## Documentation Requirements

### User Documentation

**Installation guides for each platform:**
- `docs/install/linux.md`
- `docs/install/macos.md`
- `docs/install/windows.md`

**Configuration guides:**
- Show platform-specific paths
- Include screenshots for Windows/Mac
- Provide copy-paste commands

### Developer Documentation

**Build instructions:**
- Cross-compilation setup
- Platform-specific dependencies
- Testing on each platform

---

## Future Enhancements

### 1. Portable Mode

**Use case:** Run from USB drive

```rust
// Check for .portable file next to executable
if let Ok(exe_path) = std::env::current_exe() {
    if let Some(exe_dir) = exe_path.parent() {
        let portable_marker = exe_dir.join(".portable");
        if portable_marker.exists() {
            return Ok(exe_dir.join("config"));
        }
    }
}
```

### 2. Multi-User Support

**Use case:** Shared servers

```
/etc/rbee/           # System-wide config
~/.config/rbee/      # User-specific config
```

### 3. XDG Environment Variables

**Linux:** Respect XDG environment variables

```rust
// Check XDG_CONFIG_HOME first
if let Ok(xdg_config) = std::env::var("XDG_CONFIG_HOME") {
    return Ok(PathBuf::from(xdg_config).join("rbee"));
}

// Fall back to ~/.config
dirs::config_dir()
```

---

## Implementation Checklist

### Phase 1: rbee-config Update (3-4 hours)
- [ ] Add `dirs` crate dependency
- [ ] Update `config_dir()` function
- [ ] Add `cache_dir()` function
- [ ] Add `data_dir()` function
- [ ] Update tests for cross-platform
- [ ] Update documentation

### Phase 2: Verification (2-3 hours)
- [ ] Test on Linux
- [ ] Test on macOS
- [ ] Test on Windows
- [ ] Update CI/CD for all platforms

### Phase 3: Documentation (1-2 hours)
- [ ] Update README.md
- [ ] Create platform-specific install guides
- [ ] Update architecture docs

---

## References

**Implementation Plans:**
- `bin/.plan/CROSS_PLATFORM_CONFIG_PLAN.md` - Detailed implementation
- `bin/.plan/CROSS_PLATFORM_SUMMARY.md` - Quick reference
- `bin/.plan/STORAGE_ARCHITECTURE.md` - Model storage (already cross-platform)

**External Resources:**
- [dirs crate documentation](https://docs.rs/dirs/)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
- [Apple File System Basics](https://developer.apple.com/library/archive/documentation/FileManagement/Conceptual/FileSystemProgrammingGuide/)
- [Windows Known Folders](https://learn.microsoft.com/en-us/windows/win32/shell/known-folders)

---

**Status:** Ready for implementation (TEAM-276 or integrate into Phase 1)

**Priority:** Medium (important for cross-platform support)

**Effort:** 6-9 hours total
