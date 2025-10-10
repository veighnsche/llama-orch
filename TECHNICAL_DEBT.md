# TECHNICAL DEBT - llama-orch

**TEAM-035**  
**Date:** 2025-10-10

## CRITICAL: Hardcoded Installation Paths

### Problem

**File:** `bin/rbee-keeper/src/commands/pool.rs`

**Lines 22, 27, 30, 34, 49, 54, 58, 70, 71, 72, 82**

```rust
// HARDCODED PATH - TECHNICAL DEBT!
"cd ~/Projects/llama-orch && ./target/release/rbee-hive models download {}"
```

### Impact

- ❌ **Cannot deploy to remote machines** unless they have the exact same path
- ❌ **Breaks if user clones to different directory**
- ❌ **Not following industry standards** (XDG Base Directory)
- ❌ **Violates spec** - should use proper installation paths

### Root Cause

TEAM-022 implemented SSH commands with hardcoded paths as a shortcut.

### Proper Solution (Per Industry Standards)

1. **Install binaries to standard locations:**
   - User install: `~/.local/bin/rbee-hive`
   - System install: `/usr/local/bin/rbee-hive`

2. **Use PATH to find binaries:**
   ```rust
   // Instead of:
   "./target/release/rbee-hive models list"
   
   // Should be:
   "rbee-hive models list"  // Finds in PATH
   ```

3. **Configuration file for custom paths:**
   ```toml
   # ~/.config/rbee/config.toml
   [remote]
   binary_path = "~/.local/bin/rbee-hive"  # Override if needed
   ```

### Temporary Workaround (CURRENT)

Users MUST:
1. Clone repo to `~/Projects/llama-orch` on ALL machines
2. Build with `cargo build --release` on ALL machines
3. Keep binaries in `target/release/`

### Fix Required (TEAM-036+)

**Priority:** HIGH  
**Estimated effort:** 4-6 hours

**Tasks:**
1. Create Rust-based install command: `rbee install --user` or `rbee install --system`
2. Update `pool.rs` to use binaries from PATH
3. Add config file support for custom paths
4. Update all documentation
5. Test on fresh remote machine

**Files to modify:**
- `bin/rbee-keeper/src/commands/pool.rs` - Remove hardcoded paths
- `bin/rbee-keeper/src/config.rs` (NEW) - Config file loading
- `bin/rbee-keeper/src/install.rs` (NEW) - Installation logic
- `bin/rbee-keeper/README.md` - Update deployment instructions

### Acceptance Criteria

- [ ] Can deploy to remote machine with ANY clone path
- [ ] Binaries installed to `~/.local/bin` or `/usr/local/bin`
- [ ] SSH commands use `rbee-hive` from PATH
- [ ] Config file supports custom binary paths
- [ ] Works on fresh Ubuntu/macOS machine
- [ ] Documentation updated

---

## Other Technical Debt

### Shell Scripts (Pre-existing)

**Files:**
- `scripts/rbee-models` - Shell script for model downloads (638 lines!)

**Problem:** This should be a Rust subcommand!

**Proper solution:**
```bash
# Instead of shell script:
rbee models download tinyllama
rbee models list
```

**Why this exists:** Created before "NO SHELL SCRIPTS" policy was established.

**Fix required:** Convert to Rust subcommand (TEAM-036)

**NOTE:** TEAM-035 initially created `scripts/install.sh` but DELETED it after realizing the mistake. See [NO_SHELL_SCRIPTS.md](NO_SHELL_SCRIPTS.md) for the policy.

---

## Action Items for Next Team

### IMMEDIATE (Blocks deployment)

1. **Fix hardcoded paths in pool.rs**
   - Replace `~/Projects/llama-orch` with PATH-based lookup
   - Add config file support

2. **Implement `rbee install` command**
   - Rust-based installation (no shell scripts)
   - Copies binaries to `~/.local/bin`
   - Creates default config

### MEDIUM (Quality of life)

3. **Convert `scripts/rbee-models` to Rust**
   - `rbee models download <name>`
   - `rbee models list`

4. **Add deployment command**
   - `rbee deploy --host mac.home.arpa`
   - Handles git clone, build, install automatically

### LOW (Nice to have)

5. **Config file validation**
6. **Better error messages for missing binaries**

---

## Lessons Learned

1. ❌ **Don't hardcode paths** - Use PATH and config files
2. ❌ **Don't use shell scripts** - Implement in Rust
3. ✅ **Document technical debt immediately** - Don't hide it
4. ✅ **Follow industry standards** - XDG Base Directory, PATH lookup

---

**Status:** DOCUMENTED  
**Next:** TEAM-036 must fix before production deployment  
**Severity:** HIGH - Blocks remote deployment
