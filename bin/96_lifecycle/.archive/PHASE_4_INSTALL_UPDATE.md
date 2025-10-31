# Phase 4: Install Logic Update

**Goal:** Update install logic to handle dev vs prod correctly.

**Status:** Ready to implement  
**Estimated Time:** 20 minutes

---

## What We're Building

Install behavior that:
- **Production install** (`binary="release"`) → Copy to `~/.local/bin/`
- **Development install** (`binary=None`) → Keep in `target/debug/`

---

## Current vs New Behavior

### **Current (Always copies to ~/.local/bin)**
```
install_daemon():
  1. Build binary (debug or release based on parent)
  2. Copy to ~/.local/bin/ (ALWAYS)
  3. chmod +x
```

### **New (Conditional copy)**
```
install_daemon():
  1. Determine mode from local_binary_path parameter
  2. Build binary (debug or release)
  3. If production → Copy to ~/.local/bin/
  4. If development → Keep in target/debug/ (NO COPY)
  5. Narrate which mode was installed
```

---

## Implementation Checklist

### **Step 1: Update install_daemon logic**

**File:** `bin/96_lifecycle/lifecycle-local/src/install.rs` (MODIFY)

**Replace the install logic (RULE ZERO: Break old behavior):**

**Find this section (around line 110-167):**
```rust
pub async fn install_daemon(install_config: InstallConfig) -> Result<()> {
    let daemon_name = &install_config.daemon_name;

    n!("install_start", "📦 Installing {} locally", daemon_name);

    // Step 0: Check if already installed
    if check_binary_actually_installed(daemon_name).await {
        // ... existing check ...
    }

    // Step 1: Build or locate binary
    let binary_path = resolve_binary_path(
        daemon_name,
        install_config.local_binary_path,
        install_config.job_id.clone(),
    )
    .await?;

    // Step 2-5: Copy to ~/.local/bin, chmod, verify
    // ... existing code ...
}
```

**REPLACE with:**

```rust
pub async fn install_daemon(install_config: InstallConfig) -> Result<()> {
    let daemon_name = &install_config.daemon_name;

    n!("install_start", "📦 Installing {} locally", daemon_name);

    // Determine install mode from local_binary_path parameter
    // If path contains "release" → production install
    // Otherwise → development install
    let is_production = install_config
        .local_binary_path
        .as_ref()
        .and_then(|p| p.to_str())
        .map(|s| s.contains("release"))
        .unwrap_or(false);

    if is_production {
        n!("install_mode", "🚀 Installing in PRODUCTION mode (will copy to ~/.local/bin)");
    } else {
        n!("install_mode", "🔧 Installing in DEVELOPMENT mode (will keep in target/debug)");
    }

    // Step 0: Check if already installed (only for production)
    if is_production && check_binary_actually_installed(daemon_name).await {
        n!("already_installed", "⚠️  {} is already installed in ~/.local/bin/", daemon_name);
        anyhow::bail!("{} is already installed in ~/.local/bin/. Use rebuild to update.", daemon_name);
    }

    // Step 1: Build or locate binary locally
    let binary_path = resolve_binary_path(
        daemon_name,
        install_config.local_binary_path,
        install_config.job_id.clone(),
    )
    .await?;

    // Step 2: Handle production vs development install
    if is_production {
        // Production: Copy to ~/.local/bin/
        use lifecycle_shared::BINARY_INSTALL_DIR;
        
        let home = std::env::var("HOME").context("HOME env var not set")?;
        let local_bin_dir = std::path::PathBuf::from(&home).join(BINARY_INSTALL_DIR);
        
        n!("create_dir", "📁 Creating {}", local_bin_dir.display());
        std::fs::create_dir_all(&local_bin_dir)
            .with_context(|| format!("Failed to create ~/{}", BINARY_INSTALL_DIR))?;

        let dest_path = local_bin_dir.join(daemon_name);
        n!("copying", "📤 Copying {} to {}", daemon_name, dest_path.display());

        local_copy(&binary_path, &dest_path.to_string_lossy())
            .await
            .context("Failed to copy binary to ~/.local/bin")?;

        // Make executable
        n!("chmod", "🔐 Making binary executable");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&dest_path)?.permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&dest_path, perms).context("Failed to make binary executable")?;
        }

        // Verify installation
        n!("verify", "✅ Verifying installation");
        if !dest_path.exists() {
            anyhow::bail!("Installation verification failed: binary not found at {}", dest_path.display());
        }

        n!("install_complete", "🎉 {} installed successfully at {}", daemon_name, dest_path.display());
    } else {
        // Development: Just verify binary exists in target/debug
        n!("verify_dev", "✅ Verifying development binary");
        if !binary_path.exists() {
            anyhow::bail!("Build verification failed: binary not found at {}", binary_path.display());
        }

        n!("install_complete", "🎉 {} ready for development at {}", daemon_name, binary_path.display());
        n!("install_note", "ℹ️  Development mode: binary remains in {}", binary_path.display());
    }

    Ok(())
}
```

---

## Testing Phase 4

### **Test 1: Production install**

```bash
# Should copy to ~/.local/bin/
# Via rbee-keeper UI: Click "Install (Production)"
# Or via CLI:
cargo build --package rbee-keeper
# Then trigger install with binary="release"
```

**Expected narration:**
```
📦 Installing queen-rbee locally
🚀 Installing in PRODUCTION mode (will copy to ~/.local/bin)
🔨 Building queen-rbee from source...
📁 Creating /home/user/.local/bin
📤 Copying queen-rbee to /home/user/.local/bin/queen-rbee
🔐 Making binary executable
✅ Verifying installation
🎉 queen-rbee installed successfully at /home/user/.local/bin/queen-rbee
```

### **Test 2: Development install**

```bash
# Should keep in target/debug/
# Via rbee-keeper UI: Click "Install"
# Or via CLI:
cargo build --package rbee-keeper
# Then trigger install with binary=None
```

**Expected narration:**
```
📦 Installing queen-rbee locally
🔧 Installing in DEVELOPMENT mode (will keep in target/debug)
🔨 Building queen-rbee from source...
✅ Verifying development binary
🎉 queen-rbee ready for development at target/debug/queen-rbee
ℹ️  Development mode: binary remains in target/debug/queen-rbee
```

### **Test 3: Verify files**

```bash
# After production install
ls -la ~/.local/bin/queen-rbee
# Should exist

# After development install
ls -la ~/.local/bin/queen-rbee
# Should NOT exist (or be old version)

ls -la target/debug/queen-rbee
# Should exist
```

---

## Success Criteria

- ✅ Production install copies to `~/.local/bin/`
- ✅ Development install keeps in `target/debug/`
- ✅ Clear narration for each mode
- ✅ No unnecessary file copies in dev mode
- ✅ Proper error handling
- ✅ All tests pass

---

## Files Modified

### **MODIFIED Files**
- `bin/96_lifecycle/lifecycle-local/src/install.rs`

---

## Next Phase

After Phase 4 is complete and tested, proceed to:
**`PHASE_5_TESTING.md`** - Comprehensive end-to-end testing

---

**Ready to implement!** 📦
