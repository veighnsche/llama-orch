# TEAM-259: hive-lifecycle Refactored to Use daemon-lifecycle

**Status:** ✅ COMPLETE

**Date:** Oct 23, 2025

**Mission:** Refactor hive-lifecycle to use the new daemon-lifecycle CRUD operations.

---

## Summary

Refactored 4 modules in hive-lifecycle to use shared daemon-lifecycle functions:
1. **install.rs** - Uses `install_daemon()`
2. **list.rs** - Uses `list_daemons()`
3. **get.rs** - Uses `get_daemon()`
4. **status.rs** - Uses `check_daemon_status()`

---

## Changes Made

### 1. install.rs

**Before:** 306 lines with duplicate binary finding logic

**After:** Uses `daemon-lifecycle::install_daemon()`

**Changes:**
```rust
// OLD: Manual binary finding (60+ lines)
if let Some(provided_path) = &hive_config.binary_path {
    // Verify exists...
} else {
    let debug_path = PathBuf::from("target/debug/rbee-hive");
    let release_path = PathBuf::from("target/release/rbee-hive");
    if debug_path.exists() { ... }
    else if release_path.exists() { ... }
    else { error... }
}

// NEW: Use daemon-lifecycle (10 lines)
let install_config = InstallConfig {
    binary_name: "rbee-hive".to_string(),
    binary_path: hive_config.binary_path.clone(),
    target_path: None,
    job_id: Some(job_id.to_string()),
};
let install_result = install_daemon(install_config).await?;
let binary = install_result.binary_path;
```

**Savings:** ~50 LOC per installation path (remote + localhost) = ~100 LOC

---

### 2. list.rs

**Before:** 82 lines with manual listing and table formatting

**After:** Uses `daemon-lifecycle::list_daemons()` with wrapper

**Changes:**
```rust
// Wrapper to implement ListableConfig (avoids orphan rule)
struct HiveConfigWrapper<'a>(&'a RbeeConfig);

impl<'a> ListableConfig for HiveConfigWrapper<'a> {
    type Info = HiveInfo;
    
    fn list_all(&self) -> Vec<Self::Info> {
        self.0.hives.all().iter().map(|h| HiveInfo { ... }).collect()
    }
    
    fn daemon_type_name(&self) -> &'static str {
        "hive"
    }
}

// Use generic list_daemons
let wrapper = HiveConfigWrapper(&config);
let hives = list_daemons(&wrapper, Some(job_id)).await?;
```

**Result:** 82 → 60 lines (27% reduction)

---

### 3. get.rs

**Before:** 58 lines with manual get and validation

**After:** Uses `daemon-lifecycle::get_daemon()` with wrapper

**Changes:**
```rust
// Wrapper to implement GettableConfig
struct HiveConfigWrapper<'a>(&'a RbeeConfig);

impl<'a> GettableConfig for HiveConfigWrapper<'a> {
    type Info = HiveInfo;
    
    fn get_by_id(&self, id: &str) -> Option<Self::Info> {
        self.0.hives.all().iter()
            .find(|h| h.alias == id)
            .map(|h| HiveInfo { ... })
    }
    
    fn daemon_type_name(&self) -> &'static str {
        "hive"
    }
}

// Use generic get_daemon
let wrapper = HiveConfigWrapper(&config);
let hive = get_daemon(&wrapper, &request.alias, Some(job_id)).await?;
```

**Result:** 58 → 56 lines (minimal change, but consistent interface)

---

### 4. status.rs

**Before:** 81 lines with manual HTTP health check

**After:** Uses `daemon-lifecycle::check_daemon_status()`

**Changes:**
```rust
// OLD: Manual health check
let health_url = format!("http://{}:{}/health", ...);
let client = reqwest::Client::builder().timeout(Duration::from_secs(5)).build()?;
match client.get(&health_url).send().await {
    Ok(response) if response.status().is_success() => { ... }
    Ok(response) => { ... }
    Err(_) => { ... }
}

// NEW: Use daemon-lifecycle
let status_request = StatusRequest {
    id: request.alias.clone(),
    health_url: health_url.clone(),
    daemon_type: Some("hive".to_string()),
};
let status = check_daemon_status(status_request, Some(job_id)).await?;
```

**Result:** 81 → 50 lines (38% reduction)

---

## Code Reduction

| Module | Before | After | Saved | % |
|--------|--------|-------|-------|---|
| **install.rs** | 306 | ~200 | ~106 | 35% |
| **list.rs** | 82 | 60 | 22 | 27% |
| **get.rs** | 58 | 56 | 2 | 3% |
| **status.rs** | 81 | 50 | 31 | 38% |
| **Total** | 527 | 366 | **161 LOC** | **31%** |

---

## Benefits

### Consistency
- ✅ Same install/uninstall behavior as other lifecycles
- ✅ Same list/get/status interface as other lifecycles
- ✅ Same error handling everywhere
- ✅ Same narration patterns everywhere

### Maintainability
- ✅ Bugs fixed in daemon-lifecycle benefit all lifecycles
- ✅ Improvements to CRUD operations benefit all lifecycles
- ✅ Less code to maintain in hive-lifecycle

### Reusability
- ✅ Pattern established for worker-lifecycle
- ✅ Easy to add new daemon types
- ✅ Trait-based design allows customization

---

## Technical Details

### Orphan Rule Solution

Since we can't implement external traits (`ListableConfig`, `GettableConfig`) for external types (`RbeeConfig`), we use wrapper structs:

```rust
struct HiveConfigWrapper<'a>(&'a RbeeConfig);

impl<'a> ListableConfig for HiveConfigWrapper<'a> {
    // Implementation...
}
```

This is a common Rust pattern to work around the orphan rule while maintaining clean interfaces.

---

## Compilation Status

✅ All packages compile successfully:
```bash
cargo check -p daemon-lifecycle           ✅
cargo check -p queen-rbee-hive-lifecycle  ✅
cargo check -p queen-rbee                 ✅
```

---

## Files Modified

1. **install.rs** - Refactored to use `install_daemon()`
2. **list.rs** - Refactored to use `list_daemons()` with wrapper
3. **get.rs** - Refactored to use `get_daemon()` with wrapper
4. **status.rs** - Refactored to use `check_daemon_status()`

---

## Next Steps

### Phase 1: Implement worker-lifecycle
Now that hive-lifecycle uses daemon-lifecycle, we can implement worker-lifecycle following the same pattern:

```
worker-lifecycle/
├── install.rs → Uses daemon-lifecycle::install_daemon ✅
├── list.rs → Uses daemon-lifecycle::list_daemons ✅
├── get.rs → Uses daemon-lifecycle::get_daemon ✅
├── status.rs → Uses daemon-lifecycle::check_daemon_status ✅
├── spawn.rs → Uses daemon-lifecycle::spawn_daemon ✅
├── vllm.rs → Worker-specific logic
├── llamacpp.rs → Worker-specific logic
├── stable_diffusion.rs → Worker-specific logic
└── whisper.rs → Worker-specific logic
```

### Phase 2: Add graceful shutdown
Extract graceful shutdown pattern from hive-lifecycle/stop.rs:
- SIGTERM → wait 5s → SIGKILL if needed
- Add to daemon-lifecycle/src/shutdown.rs
- Use in hive-lifecycle and worker-lifecycle

---

## Summary

**Refactored:** 4 modules in hive-lifecycle

**Savings:** 161 LOC (31% reduction)

**Benefits:**
- ✅ Consistent CRUD operations
- ✅ Shared error handling
- ✅ Easier to maintain
- ✅ Pattern for worker-lifecycle

**Compilation:** ✅ All packages compile

**The hive-lifecycle now uses shared daemon-lifecycle operations!** 🎉
