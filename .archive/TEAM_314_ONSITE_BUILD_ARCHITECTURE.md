# TEAM-314: On-Site Build Architecture

**Status:** ✅ COMPLETE  
**Date:** 2025-10-27  
**Purpose:** Fix hive remote installation to build on-site (not upload pre-built binary)

---

## Problem

**Original (Wrong) Architecture:**
```
Keeper (low-power device)
    ↓ Build rbee-hive locally
    ↓ Upload pre-built binary
Hive (powerful server)
    ↓ Just run the binary
```

**Issue:** Keeper may be a low-power device (Raspberry Pi, old laptop) that can't build Rust binaries efficiently or at all.

---

## Solution

**Correct Architecture:**
```
Keeper (low-power device)
    ↓ SSH to hive
    ↓ Tell hive to build itself
Hive (powerful server)
    ↓ Git clone (shallow)
    ↓ Build on-site
    ↓ Install locally
```

**Rationale:** The hive is typically a powerful server with GPUs. It has the resources to build. The keeper is just an orchestrator.

---

## Implementation

### New Remote Install Flow

**File:** `bin/05_rbee_keeper_crates/hive-lifecycle/src/install.rs`

```rust
/// Install rbee-hive remotely via SSH
///
/// TEAM-314: Build on-site, not locally
/// Architecture: Keeper may run on low-power device, so we MUST build on the hive itself
/// 
/// Steps:
/// 1. Git clone (shallow, no history) on remote hive
/// 2. Build on the hive using cargo
/// 3. Install the locally-built binary
async fn install_hive_remote(
    host: &str,
    binary_path: Option<String>,
    install_dir: Option<String>,
) -> Result<()>
```

### Steps

1. **Check Prerequisites**
   - Verify git is installed on hive
   - Verify cargo/rust is installed on hive
   - Helpful error messages if missing

2. **Clone Repository**
   - Shallow clone (depth=1, no history)
   - Clone to `/tmp/llama-orch-build`
   - Remove old build dir if exists

3. **Build On-Site**
   - `cargo build --release --bin rbee-hive`
   - Build happens on the hive (not keeper)
   - May take a few minutes (hive has the power)

4. **Install Binary**
   - Copy from `target/release/rbee-hive` to `~/.local/bin/rbee-hive`
   - Make executable
   - No sudo needed (uses ~/.local/bin)

5. **Cleanup**
   - Remove build directory
   - Verify installation

### Legacy Mode

For testing/development, you can still upload a pre-built binary:

```bash
# Build locally
cargo build --bin rbee-hive

# Upload (legacy mode)
rbee hive install -a workstation -b ./target/debug/rbee-hive
```

This triggers the legacy upload path.

---

## Detailed Narration

### Prerequisites Check

```
🔍 Checking for git...
✅ Git found
🔍 Checking for cargo...
✅ Cargo found
```

**If missing:**
```
Error: Git not found on remote host. Install git first: sudo apt install git
Error: Cargo not found on remote host. Install Rust first: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Build Process

```
🏗️  Building on-site (hive has the power, keeper might not)
📥 Cloning repo to /tmp/llama-orch-build (shallow clone)...
✅ Clone complete
🔨 Building rbee-hive on 'workstation' (this may take a few minutes)...
✅ Build complete
📦 Installing binary to $HOME/.local/bin...
✅ Hive installed on 'workstation' (version: 0.1.0)
📍 Binary location: /home/user/.local/bin/rbee-hive
💡 Make sure $HOME/.local/bin is in PATH on 'workstation'
```

---

## Architecture Comparison

### Before (Wrong)

| Component | Role | Resources |
|-----------|------|-----------|
| Keeper | Build + Upload | ❌ May be low-power |
| Hive | Just run | ✅ Powerful (wasted) |

**Problem:** Keeper may not have resources to build Rust.

### After (Correct)

| Component | Role | Resources |
|-----------|------|-----------|
| Keeper | Orchestrate | ✅ Low-power OK |
| Hive | Build + Run | ✅ Uses its power |

**Solution:** Each component does what it's good at.

---

## Configuration

### Git URL

Currently hardcoded (TODO):
```rust
let git_url = "https://github.com/yourusername/llama-orch.git"; // TODO: Get from config
```

**Future:** Should be configurable via:
- Environment variable
- Config file
- CLI argument

### Install Directory

Default: `$HOME/.local/bin` (no sudo needed)

Can override:
```bash
rbee hive install -a workstation -d /usr/local/bin
```

---

## Prerequisites for Remote Hive

The remote hive must have:

1. **Git**
   ```bash
   sudo apt install git
   ```

2. **Rust/Cargo**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

3. **Build Dependencies**
   ```bash
   sudo apt install build-essential pkg-config libssl-dev
   ```

4. **SSH Access**
   - Host in `~/.ssh/config` on keeper
   - SSH key configured
   - Public key on hive

---

## Error Handling

### Git Not Found
```
Error: Git not found on remote host. Install git first: sudo apt install git
```

### Cargo Not Found
```
Error: Cargo not found on remote host. Install Rust first: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Clone Failed
```
Error: Failed to clone repository
Context: Failed to execute SSH command
```

### Build Failed
```
Error: Failed to build rbee-hive on remote host
Context: Failed to execute SSH command
```

All errors include helpful context and suggestions.

---

## Testing

### Test On-Site Build

```bash
# Make sure hive has git and cargo
ssh workstation "which git && which cargo"

# Install (will build on-site)
rbee hive install -a workstation

# Expected output:
# 🔍 Checking for git...
# ✅ Git found
# 🔍 Checking for cargo...
# ✅ Cargo found
# 📥 Cloning repo...
# 🔨 Building rbee-hive on 'workstation'...
# ✅ Build complete
# ✅ Hive installed
```

### Test Legacy Upload

```bash
# Build locally
cargo build --bin rbee-hive

# Upload (legacy mode)
rbee hive install -a workstation -b ./target/debug/rbee-hive

# Expected output:
# ⚠️  Using legacy upload mode (binary provided)
# ✅ Hive installed
```

---

## Benefits

1. **Keeper can be low-power** - Raspberry Pi, old laptop, etc.
2. **Hive uses its resources** - Powerful server builds itself
3. **Always fresh build** - Builds from latest code
4. **No binary transfer** - Saves bandwidth
5. **Correct architecture** - Each component does what it's good at

---

## Future Enhancements

1. **Configurable Git URL** - Don't hardcode GitHub URL
2. **Branch selection** - Allow installing from specific branch
3. **Caching** - Keep build dir for faster rebuilds
4. **Progress streaming** - Stream cargo build output to keeper
5. **Parallel builds** - Build on multiple hives simultaneously

---

## Related Work

- **TEAM-290:** Original hive-lifecycle implementation (upload-based)
- **TEAM-314:** Port configuration + narration migration + **on-site build**

---

**Maintained by:** TEAM-314  
**Last Updated:** 2025-10-27  
**Status:** COMPLETE ✅
