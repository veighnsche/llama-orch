# TEAM-365: RULE ZERO Compliance Report

**Date:** Oct 30, 2025  
**Status:** ✅ FULLY COMPLIANT

---

## 🔥 RULE ZERO: Breaking Changes > Backwards Compatibility

**Principle:** Pre-1.0 software is ALLOWED to break. The compiler will catch breaking changes. Entropy from "backwards compatibility" functions is PERMANENT TECHNICAL DEBT.

---

## ✅ What We Did Right

### **1. Extracted to Shared Crate (No Duplication)**
- ❌ **WRONG:** Keep `parse_ssh_config()` in both places "for compatibility"
- ✅ **RIGHT:** Created `bin/99_shared_crates/ssh-config-parser/` and **DELETED** old code

**Deleted Code:**
- `bin/00_rbee_keeper/src/ssh_resolver.rs::parse_ssh_config()` (93 LOC deleted)
- `bin/00_rbee_keeper/src/ssh_resolver.rs::get_ssh_config_path()` (4 LOC deleted)
- `bin/00_rbee_keeper/src/ssh_resolver.rs::test_parse_ssh_config()` (24 LOC deleted)

**Total Deleted:** 121 LOC of duplicate code

---

### **2. Updated All Call Sites**
- ❌ **WRONG:** Keep old function, create `parse_ssh_config_v2()`
- ✅ **RIGHT:** Updated `resolve_ssh_config()` to use shared crate, fixed all call sites

**Updated Functions:**
- `bin/00_rbee_keeper/src/ssh_resolver.rs::resolve_ssh_config()` - Now uses `ssh_config_parser::parse_ssh_config()`
- `bin/00_rbee_keeper/src/tauri_commands.rs::ssh_list()` - Now uses `ssh_config_parser::parse_ssh_config()`

---

### **3. Fixed Breaking Changes from TEAM-358**
- ❌ **WRONG:** Add `SshConfig::localhost_v2()` for backwards compatibility
- ✅ **RIGHT:** Fixed `resolve_ssh_config()` to manually create localhost config

**Breaking Change:** TEAM-358 removed `SshConfig::localhost()` method

**Our Fix:**
```rust
// TEAM-365: TEAM-358 removed SshConfig::localhost(), so we create it manually
if host_alias == "localhost" {
    return Ok(SshConfig::new(
        "localhost".to_string(),
        whoami::username(),
        22,
    ));
}
```

---

### **4. No Wrapper Functions**
- ❌ **WRONG:** Create wrapper `parse_ssh_config_new()` that calls shared crate
- ✅ **RIGHT:** Directly use `ssh_config_parser::parse_ssh_config()` everywhere

---

### **5. Deleted Tests (No Duplication)**
- ❌ **WRONG:** Keep tests in both places
- ✅ **RIGHT:** Deleted `test_parse_ssh_config()` from `ssh_resolver.rs`, kept in shared crate

**Deleted Test:**
```rust
// TEAM-365: RULE ZERO - Deleted test_parse_ssh_config()
// This test is now in bin/99_shared_crates/ssh-config-parser/src/lib.rs
// Run: cargo test -p ssh-config-parser
```

---

## 📊 Code Reduction Summary

| Action | LOC Removed | LOC Added | Net Change |
|--------|-------------|-----------|------------|
| Deleted `parse_ssh_config()` | 93 | 0 | -93 |
| Deleted `get_ssh_config_path()` | 4 | 0 | -4 |
| Deleted `test_parse_ssh_config()` | 24 | 0 | -24 |
| Updated `resolve_ssh_config()` | 19 | 22 | +3 |
| Updated `ssh_list()` | 8 | 11 | +3 |
| **TOTAL** | **148** | **33** | **-115** |

**Net Result:** 115 LOC removed from rbee-keeper, moved to shared crate

---

## 🎯 RULE ZERO Principles Applied

### **1. One Way to Do Things**
- ✅ SSH config parsing: **ONE** implementation in `ssh-config-parser` crate
- ✅ No `parse_ssh_config()` vs `parse_ssh_config_v2()`
- ✅ No wrapper functions that just call the new implementation

### **2. Delete Deprecated Code Immediately**
- ✅ Deleted `parse_ssh_config()` immediately after creating shared crate
- ✅ Deleted `get_ssh_config_path()` immediately
- ✅ Deleted duplicate tests immediately

### **3. Fix Compilation Errors**
- ✅ Updated `resolve_ssh_config()` to use shared crate
- ✅ Fixed localhost handling after TEAM-358 breaking change
- ✅ Updated tests to match new implementation

### **4. No Backwards Compatibility**
- ✅ No `#[deprecated]` attributes
- ✅ No wrapper functions
- ✅ No "keep both APIs for compatibility"

---

## 🚫 What We Avoided (Entropy Patterns)

### **❌ BANNED Pattern 1: Wrapper Functions**
```rust
// WRONG - Creates permanent technical debt
pub fn parse_ssh_config(path: &PathBuf) -> Result<HashMap<String, SshConfig>> {
    // Just call the new implementation
    let targets = ssh_config_parser::parse_ssh_config(path)?;
    // Convert back to HashMap for "compatibility"
    // ...
}
```

### **❌ BANNED Pattern 2: Versioned Functions**
```rust
// WRONG - Now we have 2 functions to maintain
pub fn parse_ssh_config_v1(path: &PathBuf) -> Result<HashMap<String, SshConfig>> { ... }
pub fn parse_ssh_config_v2(path: &Path) -> Result<Vec<SshTarget>> { ... }
```

### **❌ BANNED Pattern 3: Deprecated Attributes**
```rust
// WRONG - Code still exists, still needs maintenance
#[deprecated(note = "Use ssh_config_parser::parse_ssh_config() instead")]
pub fn parse_ssh_config(path: &PathBuf) -> Result<HashMap<String, SshConfig>> { ... }
```

---

## ✅ Verification

### **Compilation**
```bash
✅ cargo check -p ssh-config-parser  # New shared crate
✅ cargo check -p rbee-hive          # Uses shared crate
✅ cargo check -p queen-rbee         # Uses shared crate
✅ cargo check --lib -p rbee-keeper  # Updated to use shared crate
```

### **Tests**
```bash
✅ cargo test -p ssh-config-parser   # Tests in shared crate
✅ cargo test --lib -p rbee-keeper   # Tests updated
```

### **Code Search**
```bash
# Verify no duplicate implementations
grep -r "pub fn parse_ssh_config" bin/00_rbee_keeper/src/
# Result: No matches (deleted!)

grep -r "pub fn parse_ssh_config" bin/99_shared_crates/ssh-config-parser/src/
# Result: 1 match (shared crate only)
```

---

## 📝 Documentation

### **Deleted Code Comments**
All deleted code has clear comments explaining where it moved:

```rust
// TEAM-365: RULE ZERO - Deleted deprecated parse_ssh_config() and get_ssh_config_path()
// These functions are now in bin/99_shared_crates/ssh-config-parser/
// Use ssh_config_parser::parse_ssh_config() and ssh_config_parser::get_default_ssh_config_path() instead
```

### **Updated Module Documentation**
```rust
//! SSH Config Resolver
//!
//! TEAM-332: Middleware to resolve host aliases to SshConfig
//! TEAM-365: Updated to use shared ssh-config-parser crate (RULE ZERO)
```

---

## 🎓 Lessons Learned

1. **Delete immediately** - Don't wait, don't deprecate, just delete
2. **Compiler is your friend** - It finds all call sites in seconds
3. **One source of truth** - Shared crate eliminates duplication
4. **No wrappers** - Direct usage of shared crate everywhere
5. **Document deletions** - Clear comments explain where code moved

---

## 🏆 RULE ZERO Compliance: 100%

- ✅ No backwards compatibility
- ✅ No deprecated code
- ✅ No wrapper functions
- ✅ No versioned functions
- ✅ One way to do things
- ✅ Deleted immediately
- ✅ Fixed all call sites
- ✅ Compiler verified

**TEAM-365: Breaking changes are temporary. Entropy is forever.** ✅

**Don't be team 68.** ✅
