# Lifecycle Shared Crate Refactoring

**TEAM-367: Created lifecycle-shared crate to eliminate code duplication**

## Summary

Created `lifecycle-shared` crate to extract common types and utilities duplicated between `lifecycle-local` and `lifecycle-ssh`.

## Changes

### New Crate: `lifecycle-shared`

**Location:** `bin/96_lifecycle/lifecycle-shared/`

**Exports:**
- `BuildConfig` - Configuration for building daemon binaries
- `DaemonStatus` - Daemon status information (running, installed)
- `HttpDaemonConfig` - Configuration for HTTP-based daemons
- Serde utilities (`serialize_systemtime`, `deserialize_systemtime`)

**Files Created:**
- `Cargo.toml` - Dependencies: anyhow, tokio, serde, serde_json, observability-narration-core, specta (optional)
- `src/lib.rs` - Module exports
- `src/build.rs` - `BuildConfig` struct with builder pattern
- `src/status.rs` - `DaemonStatus` struct
- `src/start.rs` - `HttpDaemonConfig` struct with builder pattern
- `src/utils/mod.rs` - Utils module
- `src/utils/serde.rs` - SystemTime serialization helpers
- `README.md` - Documentation

### Modified: `lifecycle-local`

**Changes:**
- Added dependency on `lifecycle-shared`
- Removed duplicate `BuildConfig` struct (now imported from shared)
- Removed duplicate `DaemonStatus` struct (now imported from shared)
- Removed duplicate `HttpDaemonConfig` struct (now imported from shared)
- Removed duplicate serde utilities (now imported from shared)
- Updated `BuildConfig` initializers to include `features: None` field
- **Deleted** `src/utils/serde.rs` (moved to shared)

**Files Modified:**
- `Cargo.toml` - Added `lifecycle-shared` dependency
- `src/build.rs` - Import `BuildConfig` from shared
- `src/status.rs` - Import `DaemonStatus` from shared
- `src/start.rs` - Import `HttpDaemonConfig` from shared
- `src/install.rs` - Updated `BuildConfig` initializer
- `src/rebuild.rs` - Updated `BuildConfig` initializer
- `src/utils/mod.rs` - Removed serde module, re-export from shared

**Files Deleted:**
- `src/utils/serde.rs` - Now in lifecycle-shared

### Modified: `lifecycle-ssh`

**Changes:**
- Added dependency on `lifecycle-shared`
- Removed duplicate `BuildConfig` struct (now imported from shared)
- Removed duplicate `DaemonStatus` struct (now imported from shared)
- Removed duplicate `HttpDaemonConfig` struct (now imported from shared)
- Removed duplicate serde utilities (now imported from shared)
- Updated `BuildConfig` initializers to include `features: None` field
- **Deleted** `src/utils/serde.rs` (moved to shared)

**Files Modified:**
- `Cargo.toml` - Added `lifecycle-shared` dependency
- `src/build.rs` - Import `BuildConfig` from shared
- `src/status.rs` - Import `DaemonStatus` from shared
- `src/start.rs` - Import `HttpDaemonConfig` from shared
- `src/install.rs` - Updated `BuildConfig` initializer
- `src/rebuild.rs` - Updated `BuildConfig` initializer
- `src/utils/mod.rs` - Removed serde module, re-export from shared

**Files Deleted:**
- `src/utils/serde.rs` - Now in lifecycle-shared

### Modified: Workspace

**Files Modified:**
- `Cargo.toml` - Added `lifecycle-shared` to workspace members

## Code Reduction

**Eliminated Duplication:**
- `BuildConfig` struct + impl (~55 LOC × 2 = ~110 LOC saved)
- `DaemonStatus` struct (~10 LOC × 2 = ~20 LOC saved)
- `HttpDaemonConfig` struct + impl (~95 LOC × 2 = ~190 LOC saved)
- Serde utilities + tests (~52 LOC × 2 = ~104 LOC saved)

**Total:** ~424 LOC of duplication eliminated

**Files Deleted:**
- `lifecycle-local/src/utils/serde.rs` (52 LOC)
- `lifecycle-ssh/src/utils/serde.rs` (52 LOC)

## Benefits

✅ **Single Source of Truth:** Types defined once, used everywhere  
✅ **RULE ZERO Compliant:** No backwards compatibility wrappers, clean break  
✅ **Compiler-Verified:** Type changes propagate automatically  
✅ **Maintainability:** Fix bugs once, works everywhere  
✅ **Consistency:** Same behavior in local and SSH contexts  

## Verification

```bash
# All three crates compile successfully
cargo check -p lifecycle-shared -p lifecycle-local -p lifecycle-ssh
```

**Status:** ✅ COMPLETE - All crates compile with warnings only (no errors)

## Future Work

None - refactoring is complete. The shared crate is ready for use.
