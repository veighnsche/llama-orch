# TEAM-329: Serde Utilities Extracted

**Date:** 2025-10-27  
**Rule:** Utilities ≠ Types - Serialization helpers are utilities

## The Problem

**`systemtime_serde` was in types/install.rs:**

```rust
types/install.rs:
  - InstallConfig (type) ✅
  - InstallResult (type) ✅
  - systemtime_serde (utility) ❌ WRONG PLACE!
```

**Serialization helpers are UTILITIES, not types.**

## The Solution

**Moved to utils/serde.rs:**

```rust
utils/serde.rs:
  - serialize_systemtime()
  - deserialize_systemtime()
```

## Changes Made

1. ✅ **Created utils/serde.rs**
   - Extracted `systemtime_serde` module
   - Renamed functions: `serialize` → `serialize_systemtime`, `deserialize` → `deserialize_systemtime`
   - Added tests

2. ✅ **Updated types/install.rs**
   - Removed local `systemtime_serde` module
   - Use `crate::utils::serde::serialize_systemtime`
   - Use `crate::utils::serde::deserialize_systemtime`

3. ✅ **Updated utils/mod.rs**
   - Added `pub mod serde`

## Before vs After

### Before (Wrong)
```rust
// types/install.rs
pub struct InstallResult { ... }

mod systemtime_serde {  // ❌ Utility in types file
    pub fn serialize(...) { ... }
    pub fn deserialize(...) { ... }
}
```

### After (Correct)
```rust
// types/install.rs
pub struct InstallResult {
    #[serde(
        serialize_with = "crate::utils::serde::serialize_systemtime",
        deserialize_with = "crate::utils::serde::deserialize_systemtime"
    )]
    pub install_time: SystemTime,
}

// utils/serde.rs
pub fn serialize_systemtime(...) { ... }
pub fn deserialize_systemtime(...) { ... }
```

## Why This Matters

**Types vs Utilities:**
- **Types** = Data structures (InstallConfig, InstallResult)
- **Utilities** = Helper functions (serialization, parsing, formatting)

**Serialization helpers are utilities:**
- They're reusable across multiple types
- They're implementation details, not data structures
- They belong in utils/, not types/

## Final Structure

```
utils/
├── find.rs       - Binary finding
├── paths.rs      - Path helpers
├── pid.rs        - PID operations
├── poll.rs       - Polling with backoff
├── serde.rs      - Serialization helpers ✅ NEW
└── timeout.rs    - Timeout enforcement
```

## Compilation

✅ **daemon-lifecycle:** PASS

---

**Key Insight:** If it's a helper function (not a data structure), it's a utility.

**Serialization helpers = utilities, not types.**
