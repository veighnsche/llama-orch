# Narration-Core Refactoring Analysis

**Date**: 2025-09-30 22:57  
**Status**: Pre-merge review

---

## Current Structure

```
src/
├── lib.rs (193 lines) - Core API + NarrationFields
├── capture.rs (266 lines) - Test capture adapter
├── redaction.rs (160 lines) - Secret masking
├── otel.rs (103 lines) - OpenTelemetry integration
├── auto.rs (201 lines) - Auto-injection helpers
└── http.rs (202 lines) - HTTP header propagation
```

**Total**: 1,125 lines across 6 files

---

## Issues Found

### 1. Hardening Issues ⚠️

#### a. Unsafe `unwrap()` calls (4 occurrences)

**Location**: `src/auto.rs:17`
```rust
SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()  // ✅ SAFE - has fallback
```
**Status**: ✅ SAFE - Uses `unwrap_or_default()`

**Location**: `src/redaction.rs:39, 46, 53`
```rust
Regex::new(r"...").unwrap()  // ⚠️ UNSAFE - panics on invalid regex
```
**Status**: ⚠️ **NEEDS FIX** - Regex compilation can fail
**Impact**: MEDIUM - Would panic on startup if regex is invalid
**Fix**: Use `expect()` with descriptive message (compile-time constant, should never fail)

**Location**: `src/capture.rs:111`
```rust
self.events.lock().unwrap().clone()  // ⚠️ UNSAFE - panics if mutex poisoned
```
**Status**: ⚠️ **NEEDS FIX** - Mutex can be poisoned
**Impact**: LOW - Only in tests, but should handle gracefully
**Fix**: Use `expect()` with descriptive message or return `Result`

#### b. Missing error handling

**Location**: `src/http.rs` - `HeaderLike` trait
```rust
fn get_str(&self, name: &str) -> Option<String>;
fn insert_str(&mut self, name: &str, value: &str);
```
**Status**: ⚠️ **POTENTIAL ISSUE** - No validation of header values
**Impact**: LOW - Callers should validate, but could add safety
**Fix**: Add validation or document requirements

### 2. Organization Issues 📁

#### a. Flat structure
All modules are in `src/` root. With 6 modules, this is getting crowded.

**Recommendation**: Group related modules into subdirectories:
```
src/
├── lib.rs
├── core/
│   ├── mod.rs (re-exports)
│   ├── fields.rs (NarrationFields struct)
│   └── narrate.rs (narrate() function)
├── adapters/
│   ├── mod.rs
│   ├── capture.rs (test capture)
│   └── redaction.rs (secret masking)
└── cloud/
    ├── mod.rs
    ├── otel.rs (OpenTelemetry)
    ├── auto.rs (auto-injection)
    └── http.rs (HTTP headers)
```

**Pros**:
- Clear separation of concerns
- Easier to navigate
- Scales better as features grow

**Cons**:
- More files to manage
- Slightly more complex imports
- Breaking change if modules are public

**Decision**: ⏸️ **DEFER** - Current flat structure is acceptable for 6 modules. Reorganize when we hit 10+ modules or add more features.

#### b. Large `NarrationFields` struct (30+ fields)

**Location**: `src/lib.rs:62-116`

**Status**: ✅ ACCEPTABLE - This is a data transfer object (DTO) with clear groupings
**Rationale**: 
- All fields are optional (except actor/action/target/human)
- Well-documented with comments
- Grouped logically (correlation, contextual, engine, performance, provenance)
- Matches spec requirements (ORCH-3304)

**No action needed**.

### 3. Code Quality Issues 🔍

#### a. Duplicate code in auto-injection

**Location**: `src/auto.rs:24-50` and `src/auto.rs:77-103`

Both `narrate_auto()` and `narrate_full()` have similar injection logic.

**Recommendation**: Extract common logic:
```rust
fn inject_provenance(fields: &mut NarrationFields) {
    if fields.emitted_by.is_none() {
        fields.emitted_by = Some(service_identity());
    }
    if fields.emitted_at_ms.is_none() {
        fields.emitted_at_ms = Some(current_timestamp_ms());
    }
}
```

**Impact**: LOW - Only 6 lines duplicated
**Decision**: ✅ **IMPLEMENT** - Reduces duplication, improves maintainability

#### b. Test-only code in production modules

**Location**: `src/capture.rs` - Entire module is test-only

**Recommendation**: Gate behind `#[cfg(test)]` or feature flag

**Current**:
```rust
// capture.rs is always compiled
pub struct CaptureAdapter { ... }
```

**Better**:
```rust
#[cfg(any(test, feature = "test-support"))]
pub struct CaptureAdapter { ... }
```

**Impact**: MEDIUM - Reduces binary size in production
**Decision**: ✅ **IMPLEMENT** - Add `test-support` feature flag

#### c. Missing documentation

**Location**: `src/http.rs:HeaderLike` trait

**Status**: ⚠️ **NEEDS DOCS** - Public trait with no documentation

**Fix**: Add trait-level documentation explaining purpose and usage

### 4. Performance Issues ⚡

#### a. Unnecessary cloning in capture adapter

**Location**: `src/capture.rs:111`
```rust
pub fn captured(&self) -> Vec<CapturedNarration> {
    self.events.lock().unwrap().clone()  // ← Clones entire vector
}
```

**Impact**: LOW - Only in tests, small vectors
**Decision**: ✅ ACCEPTABLE - Simplifies API, tests don't need zero-copy

#### b. Regex compilation on first use

**Location**: `src/redaction.rs:36-55`

**Status**: ✅ OPTIMAL - Uses `OnceLock` for lazy initialization
**Rationale**: Regex compiled once, cached forever. Zero overhead after first call.

**No action needed**.

---

## Refactoring Plan

### Priority 1: Hardening (MUST FIX)

1. **Fix regex `unwrap()` calls** ✅
   - Change to `expect()` with descriptive messages
   - Regex patterns are compile-time constants, should never fail
   - File: `src/redaction.rs`

2. **Fix mutex `unwrap()` call** ✅
   - Change to `expect()` with descriptive message
   - Document that poisoned mutex is a test bug
   - File: `src/capture.rs`

### Priority 2: Code Quality (SHOULD FIX)

3. **Extract common injection logic** ✅
   - Create `inject_provenance()` helper
   - Use in both `narrate_auto()` and `narrate_full()`
   - File: `src/auto.rs`

4. **Add `test-support` feature flag** ✅
   - Gate `capture` module behind `#[cfg(any(test, feature = "test-support"))]`
   - Update `Cargo.toml`
   - Update docs

5. **Document `HeaderLike` trait** ✅
   - Add trait-level documentation
   - Add usage examples
   - File: `src/http.rs`

### Priority 3: Organization (OPTIONAL)

6. **Reorganize into subdirectories** ⏸️ DEFER
   - Wait until we have 10+ modules
   - Current flat structure is acceptable

---

## Implementation Checklist

- [ ] Fix regex `expect()` calls (redaction.rs)
- [ ] Fix mutex `expect()` call (capture.rs)
- [ ] Extract `inject_provenance()` helper (auto.rs)
- [ ] Add `test-support` feature flag (Cargo.toml, capture.rs)
- [ ] Document `HeaderLike` trait (http.rs)
- [ ] Run tests to verify no regressions
- [ ] Update README with feature flags

---

## Estimated Time

- Priority 1 (Hardening): 15 minutes
- Priority 2 (Code Quality): 30 minutes
- Testing & verification: 15 minutes

**Total**: ~1 hour

---

## Risk Assessment

**Risk Level**: LOW

**Rationale**:
- Changes are localized
- No API changes
- All changes are internal improvements
- Tests will catch regressions

**Mitigation**:
- Run full test suite after each change
- Review diffs carefully
- Test with and without feature flags

---

## Conclusion

**Overall Assessment**: Code is in good shape, but needs minor hardening.

**Key Issues**:
1. ⚠️ 3 unsafe `unwrap()` calls in regex compilation (MUST FIX)
2. ⚠️ 1 unsafe `unwrap()` call in mutex lock (SHOULD FIX)
3. ℹ️ Minor code duplication (NICE TO HAVE)
4. ℹ️ Missing documentation (NICE TO HAVE)

**Recommendation**: Fix Priority 1 and 2 items before merge. Priority 3 can wait.
