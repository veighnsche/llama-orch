# ⚠️ CRITICAL: NO SHELL SCRIPTS FOR PRODUCT FEATURES

**This is a Rust project. We write Rust code, not shell scripts.**

---

## THE RULE

**IF YOU ARE TEMPTED TO WRITE A SHELL SCRIPT TO IMPLEMENT A PRODUCT FEATURE:**

### STOP. WRITE A POST-MORTEM INSTEAD.

Because you are about to create:
- ❌ Technical debt
- ❌ Unmaintainable code
- ❌ Platform-specific bugs
- ❌ Poor error handling
- ❌ No type safety
- ❌ Difficult testing

---

## What Shell Scripts Are NOT Allowed For

### ❌ FORBIDDEN - Product Features

- ❌ Installation (`rbee install` should be Rust, not `install.sh`)
- ❌ Model management (`rbee models download`, not `rbee-models` script)
- ❌ Deployment (`rbee deploy`, not `deploy.sh`)
- ❌ Configuration (`rbee config`, not shell scripts)
- ❌ Worker management (Rust commands, not scripts)
- ❌ Any user-facing functionality

**Why?** Because these are PRODUCT FEATURES that users depend on!

### ✅ ALLOWED - Development/CI Only

- ✅ CI/CD pipelines (`.github/workflows/*.yml`)
- ✅ Development setup helpers (`tools/setup-dev-workstation.sh`)
- ✅ Test harness runners (`test-harness/*.sh`)
- ✅ Build automation (Makefiles, justfiles)
- ✅ Git hooks

**Why?** These are developer tools, not product features.

---

## Examples of Past Mistakes

### ❌ BAD: Shell Script for Installation

```bash
#!/bin/bash
# scripts/install.sh - DON'T DO THIS!
cargo build --release
cp target/release/rbee ~/.local/bin/
```

**Problems:**
- No error handling
- Hardcoded paths
- Platform-specific
- Can't be tested
- No progress feedback

### ✅ GOOD: Rust Subcommand

```rust
// bin/rbee-keeper/src/commands/install.rs
pub fn handle(target: InstallTarget) -> Result<()> {
    let bin_dir = get_install_dir(target)?;
    copy_binaries(&bin_dir)?;
    create_config()?;
    Ok(())
}
```

**Benefits:**
- ✅ Type-safe
- ✅ Cross-platform
- ✅ Testable
- ✅ Proper error handling
- ✅ Progress bars with `indicatif`

---

## Current Technical Debt

### Shell Scripts That Should Be Rust

1. **`scripts/rbee-models`** (638 lines!)
   - Should be: `rbee models download <name>`
   - Status: TECHNICAL DEBT
   - Assigned: TEAM-036

2. **Hardcoded SSH commands in `pool.rs`**
   - Uses shell commands via SSH
   - Should use proper Rust client
   - Status: DOCUMENTED in TECHNICAL_DEBT.md

---

## What To Do Instead

### When You Need to Run External Commands

Use Rust's `std::process::Command`:

```rust
use std::process::Command;
use anyhow::Result;

pub fn git_pull(repo_path: &Path) -> Result<()> {
    let output = Command::new("git")
        .arg("pull")
        .current_dir(repo_path)
        .output()?;
    
    if !output.status.success() {
        anyhow::bail!("Git pull failed: {}", 
            String::from_utf8_lossy(&output.stderr));
    }
    
    Ok(())
}
```

### When You Need to Install Files

Use Rust's `std::fs`:

```rust
use std::fs;
use std::path::Path;
use anyhow::Result;

pub fn install_binary(src: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest.parent().unwrap())?;
    fs::copy(src, dest)?;
    
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(dest)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(dest, perms)?;
    }
    
    Ok(())
}
```

### When You Need Configuration

Use TOML with `serde`:

```rust
use serde::{Deserialize, Serialize};
use std::fs;
use anyhow::Result;

#[derive(Deserialize, Serialize)]
struct Config {
    pool_name: String,
    listen_addr: String,
}

pub fn load_config(path: &Path) -> Result<Config> {
    let contents = fs::read_to_string(path)?;
    let config: Config = toml::from_str(&contents)?;
    Ok(config)
}
```

---

## The Post-Mortem You'll Write

If you write a shell script for a product feature, here's what the post-mortem will say:

### Incident: Shell Script Caused Production Failure

**Date:** [When it breaks]  
**Severity:** HIGH  
**Root Cause:** Engineer wrote shell script instead of Rust code

**Timeline:**
- Engineer needed to implement feature X
- Engineer thought "shell script is faster"
- Shell script worked on engineer's machine
- Shell script deployed to production
- Shell script failed on different OS/environment
- Users couldn't use the product
- Emergency fix required

**Impact:**
- Users blocked
- Emergency deployment needed
- Trust damaged
- Technical debt created

**Why It Happened:**
- Engineer didn't follow project standards
- Engineer ignored "NO SHELL SCRIPTS" rule
- Engineer prioritized speed over quality

**Prevention:**
- Read this document
- Write Rust code
- Ask for help if Rust seems hard
- Remember: Rust is ALWAYS faster than debugging shell scripts

---

## Exceptions

### The ONLY Valid Reason to Write a Shell Script

**IF AND ONLY IF:**
1. It's for CI/CD or development tooling (not product features)
2. It's temporary (< 1 week lifetime)
3. It's documented as technical debt
4. There's a plan to replace it with Rust
5. You've discussed it with the team

**EVEN THEN:** Consider if Rust would be better!

---

## How to Get Help

### "But I don't know how to do X in Rust!"

**Good answers:**
- Ask the team
- Read the Rust book
- Check existing code for examples
- Use crates: `clap`, `anyhow`, `tokio`, `serde`

**Bad answers:**
- "I'll just write a shell script"
- "It's faster this way"
- "Nobody will notice"

### "But the deadline is tight!"

**Remember:**
- Shell scripts create MORE work later
- Debugging shell scripts takes LONGER
- Rust code is FASTER to maintain
- Technical debt ALWAYS comes due

---

## Summary

### DO ✅
- Write Rust code
- Use `std::process::Command` for external tools
- Use `std::fs` for file operations
- Use `serde` for configuration
- Ask for help if stuck

### DON'T ❌
- Write shell scripts for product features
- Hardcode paths
- Skip error handling
- Create technical debt
- Ignore this document

---

## Enforcement

**Code Review:**
- All PRs with `.sh` files in `bin/` or `scripts/` (except CI/dev tools) will be REJECTED
- All hardcoded shell commands in Rust code will be FLAGGED
- All technical debt must be DOCUMENTED

**Consequences:**
- PR rejected
- Technical debt ticket created
- Post-mortem if it reaches production

---

## Questions?

**Q: What about the existing `scripts/rbee-models` script?**  
A: That's TECHNICAL DEBT. It's documented in `TECHNICAL_DEBT.md` and assigned to TEAM-036 to convert to Rust.

**Q: What about CI scripts in `.github/workflows/`?**  
A: Those are fine - they're CI/CD tooling, not product features.

**Q: What about `tools/setup-dev-workstation.sh`?**  
A: That's fine - it's a development tool, not a product feature.

**Q: Can I use shell commands via `std::process::Command`?**  
A: Yes, but wrap them in proper Rust functions with error handling.

---

**REMEMBER: WE ARE BUILDING A RUST PRODUCT, NOT A COLLECTION OF SHELL SCRIPTS.**

**IF YOU WRITE A SHELL SCRIPT FOR A PRODUCT FEATURE, YOU MIGHT AS WELL WRITE THE POST-MORTEM NOW.**

---

**Last Updated:** 2025-10-10 by TEAM-035  
**Status:** ACTIVE POLICY  
**Violations:** Report to tech lead immediately
