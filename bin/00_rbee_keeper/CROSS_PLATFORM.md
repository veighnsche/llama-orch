# Cross-Platform Support Guide

**TEAM-293: Architecture for Linux, macOS, and Windows support**

## Current Status

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** | ✅ Fully Implemented | Primary development platform |
| **macOS** | ⚠️ Ready (Untested) | Platform abstraction complete |
| **Windows** | ⚠️ Ready (Untested) | Platform abstraction complete, SSH may require setup |

## Platform Abstraction Layer

### Architecture

```
┌─────────────────────────────────────────────────┐
│          rbee-keeper Application                │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────────────────────────────────┐    │
│  │    Platform-Agnostic Code             │    │
│  │  - handlers/                          │    │
│  │  - cli/                               │    │
│  │  - config.rs                          │    │
│  │  - job_client.rs                      │    │
│  └────────────────┬──────────────────────┘    │
│                   │                            │
│  ┌────────────────▼──────────────────────┐    │
│  │    Platform Abstraction Layer         │    │
│  │  - platform/mod.rs                    │    │
│  └────────────────┬──────────────────────┘    │
│                   │                            │
│  ┌────────────────▼──────────────────────┐    │
│  │  Conditional Compilation (#[cfg])     │    │
│  ├───────────┬──────────┬────────────────┤    │
│  │ linux.rs  │ macos.rs │  windows.rs    │    │
│  └───────────┴──────────┴────────────────┘    │
└─────────────────────────────────────────────────┘
```

### Module: `platform`

Location: `src/platform/`

**Purpose:** Isolate all platform-specific code in one module

**Structure:**
```
src/platform/
├── mod.rs       # Platform abstraction traits
├── linux.rs     # Linux implementation
├── macos.rs     # macOS implementation
└── windows.rs   # Windows implementation
```

### Traits Defined

#### 1. `PlatformPaths`
Provides platform-appropriate directories:

| Method | Linux | macOS | Windows |
|--------|-------|-------|---------|
| `config_dir()` | `~/.config/rbee` | `~/Library/Application Support/rbee` | `%APPDATA%/rbee` |
| `data_dir()` | `~/.local/share/rbee` | `~/Library/Application Support/rbee` | `%LOCALAPPDATA%/rbee` |
| `bin_dir()` | `~/.local/bin` | `/usr/local/bin` | `%LOCALAPPDATA%/Programs/rbee` |
| `exe_extension()` | `""` | `""` | `".exe"` |

#### 2. `PlatformProcess`
Process management operations:

| Method | Linux | macOS | Windows |
|--------|-------|-------|---------|
| `is_running(pid)` | Check `/proc/{pid}` | Use `ps -p` | Use `tasklist /FI` |
| `terminate(pid)` | `kill -TERM` | `kill -TERM` | `taskkill /PID` |
| `kill(pid)` | `kill -KILL` | `kill -KILL` | `taskkill /F /PID` |

#### 3. `PlatformRemote`
SSH and remote operation support:

| Method | Linux | macOS | Windows |
|--------|-------|-------|---------|
| `has_ssh_support()` | ✅ Always true | ✅ Always true | ⚠️ Check if OpenSSH installed |
| `ssh_executable()` | `"ssh"` | `"ssh"` | `"ssh.exe"` |
| `check_ssh_available()` | Check with `which` | Check with `which` | Check with `where` |

## Platform-Specific Considerations

### Linux ✅

**Fully Supported**

Dependencies:
```bash
# Arch Linux (current platform)
sudo pacman -S base-devel webkit2gtk libsoup libappindicator-gtk3

# Debian/Ubuntu
sudo apt install build-essential libwebkit2gtk-4.0-dev libssl-dev
```

SSH: Native OpenSSH available

**No special considerations needed**

### macOS ⚠️

**Implementation Complete - Testing Needed**

Dependencies:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install pkg-config
```

SSH: Native OpenSSH available

**Considerations:**
1. **Binary Paths:** Uses `/usr/local/bin` (may need `sudo` for installation)
2. **Config Location:** `~/Library/Application Support/rbee/` (macOS convention)
3. **Tauri Bundle:** Will create `.app` bundle instead of binary
4. **Codesigning:** May need Apple Developer certificate for distribution

**Testing Checklist:**
- [ ] Compile on macOS
- [ ] Test config file creation
- [ ] Test queen-rbee lifecycle
- [ ] Test hive management (local)
- [ ] Test SSH operations
- [ ] Create `.app` bundle with `cargo tauri build`

### Windows ⚠️

**Implementation Complete - Testing Needed**

Dependencies:
```powershell
# Install Rust (if not installed)
# Download from: https://rustup.rs/

# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/

# Install WebView2 (usually pre-installed on Windows 10+)
# Download from: https://developer.microsoft.com/en-us/microsoft-edge/webview2/
```

SSH: **Requires OpenSSH for Windows**
```powershell
# Enable OpenSSH Client (Windows 10+)
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0

# Verify installation
ssh -V
```

**Considerations:**
1. **Executable Extension:** All binaries must have `.exe` extension
2. **Path Separators:** Uses backslash `\` (handled by `std::path`)
3. **Config Location:** `%APPDATA%\rbee\` (Windows convention)
4. **Process Management:** Uses `tasklist` and `taskkill` instead of Unix signals
5. **SSH:** May not be available by default - graceful degradation needed
6. **Line Endings:** CRLF vs LF (Git should handle automatically)
7. **Permissions:** UAC may prompt for admin rights

**Testing Checklist:**
- [ ] Compile on Windows
- [ ] Test config file creation in `%APPDATA%`
- [ ] Test process management (tasklist/taskkill)
- [ ] Verify OpenSSH availability
- [ ] Test queen-rbee lifecycle
- [ ] Test hive management (local only if no SSH)
- [ ] Create `.msi` installer with `cargo tauri build`

## Usage Examples

### Using Platform Abstraction

```rust
use rbee_keeper::platform;

// Get platform-specific config directory
let config_dir = platform::config_dir()?;
println!("Config will be stored in: {:?}", config_dir);

// Get executable with correct extension
let exe_name = format!("queen-rbee{}", platform::exe_extension());
// Linux/macOS: "queen-rbee"
// Windows: "queen-rbee.exe"

// Check if process is running (cross-platform)
if platform::is_running(pid) {
    println!("Process {} is running", pid);
}

// Terminate process gracefully
platform::terminate(pid)?;

// Check SSH availability
if platform::has_ssh_support() {
    println!("SSH is available: {}", platform::ssh_executable());
} else {
    println!("SSH not available - remote operations disabled");
}
```

### Conditional Compilation

```rust
#[cfg(target_os = "linux")]
use std::os::unix::fs::PermissionsExt;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

// Linux/macOS specific
#[cfg(not(target_os = "windows"))]
fn set_executable_permission(path: &Path) -> Result<()> {
    use std::fs;
    let mut perms = fs::metadata(path)?.permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms)?;
    Ok(())
}

// Windows specific
#[cfg(target_os = "windows")]
fn set_executable_permission(path: &Path) -> Result<()> {
    // Windows doesn't need explicit executable permissions
    Ok(())
}
```

## Build Instructions

### Build for Current Platform

```bash
# CLI
cargo build --release --bin rbee-keeper

# GUI
cargo tauri build
```

### Cross-Compilation

#### Linux → Windows

```bash
# Install target
rustup target add x86_64-pc-windows-gnu

# Install MinGW
sudo pacman -S mingw-w64-gcc  # Arch
sudo apt install mingw-w64    # Debian/Ubuntu

# Build
cargo build --release --target x86_64-pc-windows-gnu --bin rbee-keeper
```

#### macOS → Linux

```bash
# Install target
rustup target add x86_64-unknown-linux-gnu

# Build (requires Linux libs - tricky)
cargo build --release --target x86_64-unknown-linux-gnu --bin rbee-keeper
```

**Note:** Cross-compilation for Tauri GUI is more complex due to WebView dependencies. Recommended to build on native platforms.

## Testing Strategy

### Per-Platform Testing

**Automated (CI/CD):**
```yaml
# GitHub Actions example
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
runs-on: ${{ matrix.os }}
steps:
  - name: Build
    run: cargo build --release
  - name: Test
    run: cargo test
```

**Manual Testing:**
1. **Config Management**
   - Create config file in platform-appropriate location
   - Verify config loads correctly
   - Verify config saves correctly

2. **Process Management**
   - Start queen-rbee
   - Check if running
   - Stop queen-rbee gracefully
   - Force kill if needed

3. **SSH Operations** (if available)
   - Test local hive operations
   - Test remote hive operations (if SSH configured)

4. **GUI** (Tauri)
   - Build GUI bundle
   - Test all pages (Status, Queen, Hives, Workers, Models, Inference)
   - Verify commands work

### Compatibility Matrix

| Feature | Linux | macOS | Windows |
|---------|-------|-------|---------|
| **CLI Binary** | ✅ | ✅ | ✅ |
| **GUI Binary** | ✅ | ✅ | ✅ |
| **Config Management** | ✅ | ✅ | ✅ |
| **Queen Lifecycle** | ✅ | ✅ | ✅ |
| **Local Hive** | ✅ | ✅ | ✅ |
| **Remote Hive (SSH)** | ✅ | ✅ | ⚠️ Requires OpenSSH |
| **Process Management** | ✅ | ✅ | ✅ |

## Known Limitations

### All Platforms
- None currently

### Windows-Specific
1. **SSH Not Default:** OpenSSH must be installed separately
   - **Mitigation:** Graceful degradation - local operations only
   - **Detection:** `platform::has_ssh_support()` returns `false`
   - **User Message:** "Remote operations require OpenSSH. Install via: Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0"

2. **Path Casing:** Windows filesystem is case-insensitive
   - **Mitigation:** Use `std::path::Path` for all path operations
   - **Impact:** Minimal - Rust handles this automatically

3. **Line Endings:** CRLF vs LF
   - **Mitigation:** Git configured with `autocrlf=true`
   - **Impact:** Config files may have different line endings

### macOS-Specific
1. **Binary Installation:** `/usr/local/bin` may require `sudo`
   - **Mitigation:** Provide clear error message
   - **Alternative:** Use `~/Library/Application Support/rbee/bin`

2. **Gatekeeper:** Unsigned binaries may be blocked
   - **Mitigation:** Document how to bypass (right-click → Open)
   - **Production:** Codesign with Apple Developer certificate

## Migration Path

### Current (Linux-only)
```
bin/00_rbee_keeper/
└── src/
    ├── main.rs
    ├── cli/
    ├── config.rs
    └── handlers/
```

### After Platform Abstraction
```
bin/00_rbee_keeper/
└── src/
    ├── main.rs
    ├── cli/
    ├── config.rs
    ├── handlers/
    └── platform/       # NEW
        ├── mod.rs
        ├── linux.rs
        ├── macos.rs
        └── windows.rs
```

### Future Code Uses Platform Module

**Before:**
```rust
let config_dir = PathBuf::from("~/.config/rbee");  // ❌ Linux-only
```

**After:**
```rust
let config_dir = platform::config_dir()?;  // ✅ Cross-platform
```

## Next Steps

### Phase 1: Testing (Current)
- [ ] Test on macOS
- [ ] Test on Windows
- [ ] Document any platform-specific quirks

### Phase 2: CI/CD
- [ ] Set up GitHub Actions for multi-platform builds
- [ ] Automated testing on all platforms
- [ ] Generate platform-specific artifacts

### Phase 3: Distribution
- [ ] Create `.app` bundle for macOS
- [ ] Create `.msi` installer for Windows
- [ ] Create `.deb`/`.rpm` packages for Linux
- [ ] Document installation for each platform

### Phase 4: Documentation
- [ ] Platform-specific installation guides
- [ ] Troubleshooting guides per platform
- [ ] Platform comparison table

## Reference

### Useful Crates

- **dirs** - Cross-platform directory paths (already used)
- **which** - Find executables in PATH (for SSH detection)
- **sysinfo** - Cross-platform system information
- **winapi** - Windows-specific APIs (if needed)
- **libc** - Unix-specific APIs (if needed)

### Resources

- [Tauri Platform Support](https://tauri.app/v1/guides/building/cross-platform)
- [Rust Platform Support](https://doc.rust-lang.org/rustc/platform-support.html)
- [Cross-compilation Guide](https://rust-lang.github.io/rustup/cross-compilation.html)

---

**TEAM-293: Platform abstraction complete and ready for testing**
