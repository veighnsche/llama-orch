# RULE ZERO FIX: Deleted process-monitor, Using Existing monitor

**Date:** Oct 30, 2025  
**Status:** ✅ FIXED

---

## ❌ RULE ZERO VIOLATION

I violated **RULE ZERO: BREAKING CHANGES > BACKWARDS COMPATIBILITY**

### What I Did Wrong:
Created a NEW `process-monitor` crate when `bin/25_rbee_hive_crates/monitor` already existed!

This is **EXACTLY** the entropy pattern RULE ZERO forbids:
- ❌ Created `process-monitor` (new)
- ❌ Left `monitor` (old) as stub
- ❌ Now have TWO crates for the same thing
- ❌ Creates permanent technical debt

### From Engineering Rules:
> **BANNED - Entropy Patterns:**
> - Creating `function_v2()`, `function_new()`, `function_with_options()` to avoid breaking `function()`
> - Adding `deprecated` attributes but keeping old code
> - Creating wrapper functions that just call new implementations
> - **"Let's keep both APIs for compatibility"**

I literally did this at the crate level! 🤦

---

## ✅ THE FIX

### **1. Deleted process-monitor**
```bash
rm -rf bin/99_shared_crates/process-monitor
```

### **2. Updated Existing monitor Crate**
```rust
// bin/25_rbee_hive_crates/monitor/src/lib.rs

// TEAM-XXX: RULE ZERO - This is THE process monitoring crate (not process-monitor)

/// Configuration for process monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    pub group: String,
    pub instance: String,
    pub cpu_limit: Option<String>,
    pub memory_limit: Option<String>,
}

/// Process statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessStats {
    pub pid: u32,
    pub group: String,
    pub instance: String,
    pub cpu_pct: f64,
    pub rss_mb: u64,
    pub vram_mb: Option<u64>,
    pub io_r_mb_s: f64,
    pub io_w_mb_s: f64,
    pub uptime_s: u64,
}
```

### **3. Updated lifecycle-monitored**
```toml
# Cargo.toml
[dependencies]
rbee-hive-monitor = { path = "../../25_rbee_hive_crates/monitor" }
```

```rust
// lib.rs
pub use rbee_hive_monitor::MonitorConfig;
```

### **4. Removed from Root Cargo.toml**
```diff
- "bin/99_shared_crates/process-monitor",     # Cross-platform process monitoring
```

---

## 🎯 Result

✅ **ONE crate for process monitoring:** `rbee-hive-monitor`  
✅ **NO entropy:** No duplicate APIs  
✅ **NO technical debt:** Only one way to do things  
✅ **Compiles successfully**

---

## 📚 Lesson Learned

### **RULE ZERO Applies to:**
- ✅ Functions (don't create `function_v2()`)
- ✅ Crates (don't create `process-monitor` when `monitor` exists!)
- ✅ APIs (don't keep both old and new)
- ✅ Everything (one way to do things)

### **When You're Tempted to Create a New Thing:**
1. **STOP** - Does something similar already exist?
2. **UPDATE** the existing thing instead
3. **DELETE** deprecated code immediately
4. **FIX** compilation errors (that's what the compiler is for!)

### **Breaking Changes are TEMPORARY:**
- Compiler finds all call sites in 30 seconds
- You fix them
- Done

### **Entropy is PERMANENT:**
- Every future developer pays the cost
- Forever

---

## ✅ Verification

```bash
# All compile successfully
cargo check --package rbee-hive-monitor
cargo check --package lifecycle-monitored

# No process-monitor exists
ls bin/99_shared_crates/process-monitor
# ls: cannot access 'bin/99_shared_crates/process-monitor': No such file or directory
```

---

## 🎉 RULE ZERO: FOLLOWED

✅ **Updated existing crate** instead of creating new one  
✅ **Deleted entropy** immediately  
✅ **One way to do things**  
✅ **No backwards compatibility cruft**

**Breaking changes > Entropy. Always.**
