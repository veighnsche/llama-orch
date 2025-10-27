# TEAM-329: Type File Pattern Alignment

**Date:** Oct 27, 2025  
**Status:** ✅ COMPLETE

## Problem

Type files in `daemon-lifecycle/src/types/` had **5 inconsistent patterns**:

1. **Serde derives** - Some had `Serialize, Deserialize`, others only `Debug, Clone`
2. **Builder methods** - Some had `new()` + `with_*()`, others had none
3. **Tests** - Some had serialization tests, others had none
4. **Field annotations** - Inconsistent use of `#[serde(skip_serializing_if = "Option::is_none")]`
5. **Empty files** - `build.rs` and `stop.rs` were just comments

## Solution

Aligned all type files to **consistent standard pattern**:

### Standard Pattern
```rust
use serde::{Deserialize, Serialize};

/// Configuration for [operation]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct [Operation]Config {
    /// Required field
    pub field: String,
    
    /// Optional field
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optional_field: Option<String>,
}

impl [Operation]Config {
    /// Create a new config
    pub fn new(required: impl Into<String>) -> Self {
        Self {
            field: required.into(),
            optional_field: None,
        }
    }
    
    /// Set optional field
    pub fn with_optional_field(mut self, value: impl Into<String>) -> Self {
        self.optional_field = Some(value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_serialization() {
        // Test serialization round-trip
    }
}
```

## Files Modified

### 1. `types/rebuild.rs` (+31 LOC)
- ✅ Added `use serde::{Deserialize, Serialize}`
- ✅ Added `#[derive(Serialize, Deserialize)]`
- ✅ Added `#[serde(skip_serializing_if = "Option::is_none")]` to optional fields
- ✅ Added tests: `test_rebuild_config_builder`, `test_rebuild_config_serialization`

### 2. `types/timeout.rs` (+51 LOC)
- ✅ Added `use serde::{Deserialize, Serialize}`
- ✅ Added `#[derive(Serialize, Deserialize)]`
- ✅ Added `#[serde(skip_serializing_if = "Option::is_none")]` to `job_id`
- ✅ Added Duration serialization helpers (serialize as seconds)
- ✅ Added tests: `test_timeout_config_builder`, `test_timeout_config_serialization`

### 3. `types/install.rs` (+28 LOC)
- ✅ Added builder methods: `new()`, `with_binary_path()`, `with_target_path()`, `with_job_id()`
- ✅ Already had serde derives and tests

### 4. `types/shutdown.rs` (+24 LOC)
- ✅ Added builder methods: `new()`, `with_graceful_timeout_secs()`, `with_job_id()`
- ✅ Already had serde derives and tests

### 5. `types/uninstall.rs` (+29 LOC)
- ✅ Added builder methods: `new()`, `with_health_url()`, `with_health_timeout_secs()`, `with_job_id()`
- ✅ Already had serde derives and tests

### 6. `types/status.rs` (+16 LOC)
- ✅ Added builder methods to `StatusRequest`: `new()`, `with_job_id()`
- ✅ Already had serde derives and tests
- ✅ `HealthPollConfig` already had builder pattern

## Files Unchanged

### `types/start.rs`
- ✅ Already followed standard pattern (serde, builder, tests)

### `types/build.rs` & `types/stop.rs`
- ✅ Empty files by design (no unique types needed)
- ✅ Kept as-is for structural parity

## Results

### Before
- ❌ 5 different patterns across 9 files
- ❌ 2 files missing serde derives
- ❌ 5 files missing builder methods
- ❌ 2 files missing tests

### After
- ✅ **Single consistent pattern** across all files
- ✅ All config types have `Serialize + Deserialize`
- ✅ All config types have builder methods
- ✅ All config types have serialization tests
- ✅ All optional fields use `#[serde(skip_serializing_if = "Option::is_none")]`

## Verification

```bash
# Compilation
cargo check -p daemon-lifecycle
# ✅ PASS (4 warnings - pre-existing unused imports)

# Tests
cargo test -p daemon-lifecycle --lib
# ✅ PASS (18 tests, all passing)
```

## Test Coverage

**Total: 18 tests**

- `types/install.rs`: 2 tests
- `types/rebuild.rs`: 2 tests (NEW)
- `types/shutdown.rs`: 2 tests
- `types/start.rs`: 2 tests
- `types/status.rs`: 2 tests
- `types/timeout.rs`: 2 tests (NEW)
- `types/uninstall.rs`: 1 test
- `rebuild.rs`: 2 tests
- `utils/pid.rs`: 1 test
- `utils/serde.rs`: 1 test
- `utils/paths.rs`: 1 test

## Benefits

1. **Consistency** - Single pattern, easy to follow
2. **Ergonomics** - Builder pattern makes construction cleaner
3. **Serialization** - All types can be serialized/deserialized
4. **Testing** - All types have test coverage
5. **Maintainability** - New types know exactly what pattern to follow

## Code Quality

- ✅ No TODO markers
- ✅ All tests passing
- ✅ Compilation clean (warnings are pre-existing)
- ✅ Follows engineering rules
- ✅ TEAM-329 signatures added where appropriate

---

**Total LOC Added:** +179 LOC  
**Files Modified:** 6 files  
**Tests Added:** 4 new tests  
**Pattern Violations:** 0
