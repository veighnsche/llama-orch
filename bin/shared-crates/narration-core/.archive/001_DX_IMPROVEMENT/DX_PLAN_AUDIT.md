# DX Implementation Plan — Source Code Audit

**Date**: 2025-10-04  
**Purpose**: Verify DX plan claims against actual source code  
**Method**: Line-by-line code inspection

---

## Audit Summary

| Unit | Claim | Actual Status | Evidence |
|------|-------|---------------|----------|
| Unit 1 | Axum middleware needed | ❌ NOT IMPLEMENTED | No `src/axum.rs`, no `axum` feature |
| Unit 2 | HeaderLike example wrong | ✅ CORRECT | `src/http.rs:110-120` shows correct trait |
| Unit 3 | Builder pattern needed | ❌ NOT IMPLEMENTED | No `src/builder.rs`, no `Narration` struct |
| Unit 4 | Policy guide needed | ❌ NOT IN CODE | Documentation only |
| Unit 5 | Duplicate logic in auto.rs | ✅ EXISTS | `src/auto.rs:51-59` duplicates `inject_provenance` |
| Unit 6 | Event emission duplication | ✅ EXISTS | `src/lib.rs:304-446` repeats 35 fields × 5 levels |
| Unit 7 | Axum example needed | ❌ NOT IN CODE | Documentation only |
| Unit 8 | Constants not used | ⚠️ PARTIAL | Constants exist, examples don't use them |
| Unit 9 | Field reference needed | ❌ NOT IN CODE | Documentation only |
| Unit 10 | Troubleshooting needed | ❌ NOT IN CODE | Documentation only |
| Unit 11 | Redaction performance | ✅ ALREADY FAST | Benchmarks show 430ns-1.4μs |
| Unit 12 | narrate_auto! macro | ❌ NOT IMPLEMENTED | No declarative macro exists |

---

## Detailed Findings

### ✅ Unit 2: HeaderLike Example (ALREADY CORRECT)

**Claim**: README shows wrong method name (`get_header` instead of `get_str`)

**Reality**: Code is correct, README example matches trait definition

**Evidence**: `src/http.rs:122-130`
```rust
pub trait HeaderLike {
    /// Get a header value as a String.
    fn get_str(&self, name: &str) -> Option<String>;

    /// Insert a header value.
    fn insert_str(&mut self, name: &str, value: &str);
}
```

**README example** (lines 110-120):
```rust
impl HeaderLike for axum::http::HeaderMap {
    fn get_str(&self, name: &str) -> Option<String> {  // ✅ Correct!
        self.get(name)?.to_str().ok().map(String::from)
    }
    
    fn insert_str(&mut self, name: &str, value: &str) {  // ✅ Correct!
        if let Ok(header_value) = axum::http::HeaderValue::from_str(value) {
            self.insert(name, header_value);
        }
    }
}
```

**Status**: ✅ No work needed, claim was incorrect

---

### ✅ Unit 5: Duplicate Logic in auto.rs (CONFIRMED)

**Claim**: `narrate_auto` duplicates `inject_provenance` checks

**Reality**: Duplication exists

**Evidence**: `src/auto.rs:50-60`
```rust
pub fn narrate_auto(mut fields: NarrationFields) {
    inject_provenance(&mut fields);  // ← Line 51: Calls inject_provenance

    // Lines 54-59: Duplicate the same checks!
    if fields.emitted_by.is_none() {
        fields.emitted_by = Some(service_identity());
    }
    if fields.emitted_at_ms.is_none() {
        fields.emitted_at_ms = Some(current_timestamp_ms());
    }
    crate::narrate(fields);
}
```

**inject_provenance** (lines 19-26):
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

**Fix**: Remove lines 53-59 from `narrate_auto`

**Status**: ✅ Valid issue, needs fixing

---

### ✅ Unit 6: Event Emission Duplication (CONFIRMED)

**Claim**: 175 lines of duplicated field lists (5 log levels × 35 fields)

**Reality**: Massive duplication exists

**Evidence**: `src/lib.rs:304-446` (142 lines of duplication)

**Structure**:
```rust
match tracing_level {
    Level::TRACE => event!(Level::TRACE, 
        actor = fields.actor,
        action = fields.action,
        target = %fields.target,
        // ... 32 more fields
    ),
    Level::DEBUG => event!(Level::DEBUG,
        actor = fields.actor,
        action = fields.action,
        target = %fields.target,
        // ... 32 more fields (DUPLICATE)
    ),
    Level::INFO => event!(Level::INFO,
        actor = fields.actor,
        action = fields.action,
        target = %fields.target,
        // ... 32 more fields (DUPLICATE)
    ),
    Level::WARN => event!(Level::WARN,
        // ... 35 fields (DUPLICATE)
    ),
    Level::ERROR => event!(Level::ERROR,
        // ... 35 fields (DUPLICATE)
    ),
}
```

**Count**: 35 fields × 5 levels = 175 field references (actual: ~142 lines)

**Status**: ✅ Valid issue, macro extraction would help

---

### ❌ Unit 1: Axum Middleware (NOT IMPLEMENTED)

**Claim**: Need to add Axum middleware

**Reality**: No Axum integration exists

**Evidence**:
- ❌ No `src/axum.rs` file
- ❌ No `axum` feature in `Cargo.toml:31-38`
- ❌ No `axum` dependency in `Cargo.toml:15-20`

**Current features** (Cargo.toml:31-38):
```toml
[features]
default = []
trace-enabled = []
debug-enabled = []
cute-mode = []
otel = ["opentelemetry"]
test-support = []
production = []
```

**Status**: ❌ Valid TODO, needs implementation

---

### ❌ Unit 3: Builder Pattern (NOT IMPLEMENTED)

**Claim**: Need builder pattern for ergonomics

**Reality**: No builder exists

**Evidence**:
- ❌ No `src/builder.rs` file
- ❌ No `pub struct Narration` in codebase
- ❌ No builder methods (`.human()`, `.correlation_id()`, `.emit()`)

**Current API** (only option):
```rust
narrate_auto(NarrationFields {
    actor: "orchestratord",
    action: "enqueue",
    target: job_id.to_string(),
    human: format!("Enqueued job {job_id}"),
    correlation_id: Some(req_id),
    ..Default::default()
});
```

**Status**: ❌ Valid TODO, needs implementation

---

### ❌ Unit 12: narrate_auto! Macro (NOT IMPLEMENTED)

**Claim**: Improve `narrate_auto!` macro

**Reality**: No declarative macro exists

**Evidence**: `src/auto.rs` - only functions, no macros

**Grep results**:
```bash
$ grep -n "macro_rules!" src/auto.rs
# No results
```

**Status**: ❌ Valid TODO, but low priority (builder pattern may be better)

---

### ✅ Unit 8: Constants Not Used in Examples (CONFIRMED)

**Claim**: Constants exported but examples use string literals

**Reality**: Constants exist but aren't used consistently

**Evidence**: `src/lib.rs:68-76`
```rust
pub const ACTOR_ORCHESTRATORD: &str = "orchestratord";
pub const ACTOR_POOL_MANAGERD: &str = "pool-managerd";
pub const ACTOR_WORKER_ORCD: &str = "worker-orcd";
pub const ACTOR_INFERENCE_ENGINE: &str = "inference-engine";
pub const ACTOR_VRAM_RESIDENCY: &str = "vram-residency";
```

**But examples use** (README line 94):
```rust
actor: "orchestratord",  // ← Should use ACTOR_ORCHESTRATORD
```

**Status**: ✅ Valid issue, documentation update needed

---

### ✅ Unit 11: Redaction Performance (ALREADY FAST)

**Claim**: Need to optimize from 180ms to <5μs

**Reality**: Already 430ns-1.4μs (exceeds target!)

**Evidence**: Benchmark results
```
redaction/with_bearer_token:    431 ns
redaction/with_multiple_secrets: 1.36 µs
```

**Code**: `src/redaction.rs:107-135` - 6-pass regex approach

**Status**: ✅ Already complete, documentation corrected

---

## Implementation Status Matrix

### Code Changes Needed

| Unit | Type | File | Lines | Effort | Status |
|------|------|------|-------|--------|--------|
| Unit 1 | New module | `src/axum.rs` | ~80 | 3h | 📋 TODO |
| Unit 1 | Feature flag | `Cargo.toml` | 2 | 5m | 📋 TODO |
| Unit 1 | Dependency | `Cargo.toml` | 1 | 5m | 📋 TODO |
| Unit 3 | New module | `src/builder.rs` | ~150 | 4h | 📋 TODO |
| Unit 5 | Remove dupe | `src/auto.rs:53-59` | -7 | 5m | 📋 TODO |
| Unit 6 | Extract macro | `src/lib.rs:304-446` | -100 | 2h | 📋 TODO |

### Documentation Changes Needed

| Unit | Type | File | Effort | Status |
|------|------|------|--------|--------|
| Unit 2 | ❌ None | README.md | 0m | ✅ ALREADY CORRECT |
| Unit 4 | Add section | README.md | 2h | 📋 TODO |
| Unit 7 | Add example | README.md | 1h | 📋 TODO |
| Unit 8 | Update examples | README.md | 30m | 📋 TODO |
| Unit 9 | Add table | README.md | 1h | 📋 TODO |
| Unit 10 | Add section | README.md | 1h | 📋 TODO |
| Unit 11 | ✅ Corrected | README.md | 0m | ✅ COMPLETE |

---

## Corrected Implementation Plan

### Already Complete ✅

1. **Unit 11**: Redaction performance (already exceeds target)
2. **Unit 2**: HeaderLike example (already correct in code)
3. **Phase 1**: All macro implementation (62 tests passing)

### Real TODOs (Code) 📋

1. **Unit 1**: Axum middleware (`src/axum.rs` + feature flag)
2. **Unit 3**: Builder pattern (`src/builder.rs`)
3. **Unit 5**: Remove duplicate logic in `auto.rs` (7 lines)
4. **Unit 6**: Extract event emission macro (save ~100 lines)
5. **Unit 12**: Optional declarative macro (low priority)

### Real TODOs (Documentation) 📋

1. **Unit 4**: Policy guide (when to narrate)
2. **Unit 7**: Complete Axum example
3. **Unit 8**: Use constants in examples
4. **Unit 9**: Field reference table
5. **Unit 10**: Troubleshooting section

---

## Effort Recalculation

### Original Estimate
- Total: ~30 hours over 3 weeks
- Code: ~20 hours
- Docs: ~10 hours

### Actual Remaining Work

**Code** (7.5 hours):
- Unit 1: Axum middleware (3h)
- Unit 3: Builder pattern (4h)
- Unit 5: Remove duplication (5m)
- Unit 6: Extract macro (30m)

**Documentation** (6.5 hours):
- Unit 4: Policy guide (2h)
- Unit 7: Axum example (1h)
- Unit 8: Use constants (30m)
- Unit 9: Field reference (1h)
- Unit 10: Troubleshooting (1h)
- Unit 11: ✅ Already done (15m)
- Unit 2: ✅ Already correct (0m)

**Total**: ~14 hours (was: 30 hours)

**Savings**: 16 hours (53% reduction)

---

## Code Evidence Summary

### Existing Modules ✅

| Module | File | Lines | Purpose | Status |
|--------|------|-------|---------|--------|
| `auto` | `src/auto.rs` | 207 | Auto-injection | ✅ Working |
| `capture` | `src/capture.rs` | 326 | Test adapter | ✅ Working |
| `correlation` | `src/correlation.rs` | 114 | Correlation IDs | ✅ Working |
| `http` | `src/http.rs` | 213 | HTTP propagation | ✅ Working |
| `otel` | `src/otel.rs` | 97 | OpenTelemetry | ✅ Working |
| `redaction` | `src/redaction.rs` | 202 | Secret redaction | ✅ Working (fast!) |
| `trace` | `src/trace.rs` | 288 | Trace macros | ✅ Working |
| `unicode` | `src/unicode.rs` | 153 | Unicode safety | ✅ Working |

**Total**: 8 modules, ~1,600 lines, all functional

### Missing Modules ❌

| Module | File | Purpose | Priority |
|--------|------|---------|----------|
| `axum` | `src/axum.rs` | Axum middleware | P0 |
| `builder` | `src/builder.rs` | Builder pattern | P1 |

---

## Specific Code Issues

### Issue 1: Duplicate Logic in auto.rs ✅ CONFIRMED

**Location**: `src/auto.rs:50-60`

**Problem**:
```rust
pub fn narrate_auto(mut fields: NarrationFields) {
    inject_provenance(&mut fields);  // ← Line 51

    // Lines 54-59: DUPLICATE of inject_provenance (lines 19-26)
    if fields.emitted_by.is_none() {
        fields.emitted_by = Some(service_identity());
    }
    if fields.emitted_at_ms.is_none() {
        fields.emitted_at_ms = Some(current_timestamp_ms());
    }
    crate::narrate(fields);
}
```

**Fix**:
```rust
pub fn narrate_auto(mut fields: NarrationFields) {
    inject_provenance(&mut fields);
    crate::narrate(fields);
}
```

**Impact**: Remove 7 lines, eliminate duplication

---

### Issue 2: Event Emission Duplication ✅ CONFIRMED

**Location**: `src/lib.rs:304-446`

**Problem**: 35 fields repeated 5 times (one per log level)

**Evidence**:
```rust
Level::TRACE => event!(Level::TRACE,
    actor = fields.actor,
    action = fields.action,
    target = %fields.target,
    human = %human,
    cute = cute.as_deref(),
    story = story.as_deref(),
    correlation_id = fields.correlation_id.as_deref(),
    session_id = fields.session_id.as_deref(),
    job_id = fields.job_id.as_deref(),
    // ... 26 more fields
),
Level::DEBUG => event!(Level::DEBUG,
    actor = fields.actor,  // ← DUPLICATE
    action = fields.action,  // ← DUPLICATE
    target = %fields.target,  // ← DUPLICATE
    // ... 32 more DUPLICATE fields
),
// ... 3 more levels with same duplication
```

**Line count**:
- Level::TRACE: ~35 lines (lines 304-340)
- Level::DEBUG: ~35 lines (duplicate)
- Level::INFO: ~35 lines (duplicate)
- Level::WARN: ~35 lines (duplicate)
- Level::ERROR: ~35 lines (duplicate)
- **Total duplication**: ~140 lines

**Fix**: Extract into internal macro (see Unit 6 plan)

**Status**: ✅ Valid issue, needs fixing

---

### Issue 3: No Axum Integration ❌ CONFIRMED

**Location**: N/A (doesn't exist)

**Evidence**:
```bash
$ ls src/axum.rs
ls: cannot access 'src/axum.rs': No such file or directory

$ grep -r "axum" Cargo.toml
# No results in dependencies or features
```

**Current workaround**: Developers must implement their own middleware using `http::extract_context_from_headers`

**Status**: ❌ Valid TODO, needs implementation

---

### Issue 4: No Builder Pattern ❌ CONFIRMED

**Location**: N/A (doesn't exist)

**Evidence**:
```bash
$ ls src/builder.rs
ls: cannot access 'src/builder.rs': No such file or directory

$ grep -r "pub struct Narration" src/
# Only finds NarrationFields, not Narration builder
```

**Current API** (only option):
```rust
// Verbose: 7 lines
narrate_auto(NarrationFields {
    actor: "orchestratord",
    action: "enqueue",
    target: job_id.to_string(),
    human: format!("Enqueued job {job_id}"),
    correlation_id: Some(req_id),
    ..Default::default()
});
```

**Proposed API** (builder):
```rust
// Concise: 4 lines
Narration::new("orchestratord", "enqueue", job_id)
    .human(format!("Enqueued job {job_id}"))
    .correlation_id(req_id)
    .emit();
```

**Status**: ❌ Valid TODO, needs implementation

---

### Issue 5: Constants Exist But Unused ⚠️ CONFIRMED

**Location**: `src/lib.rs:68-76` (constants defined)

**Evidence**:

**Constants exist**:
```rust
pub const ACTOR_ORCHESTRATORD: &str = "orchestratord";
pub const ACTOR_POOL_MANAGERD: &str = "pool-managerd";
pub const ACTOR_WORKER_ORCD: &str = "worker-orcd";
pub const ACTOR_INFERENCE_ENGINE: &str = "inference-engine";
pub const ACTOR_VRAM_RESIDENCY: &str = "vram-residency";

pub const ACTION_ADMISSION: &str = "admission";
pub const ACTION_ENQUEUE: &str = "enqueue";
pub const ACTION_DISPATCH: &str = "dispatch";
// ... more actions
```

**But examples use literals**:
```rust
actor: "orchestratord",  // Should be: ACTOR_ORCHESTRATORD
action: "enqueue",       // Should be: ACTION_ENQUEUE
```

**Status**: ⚠️ Valid issue, documentation update needed

---

## Revised Effort Estimates

### Code Work (7.5 hours)

| Unit | Task | Estimate | Actual Effort | Confidence |
|------|------|----------|---------------|------------|
| Unit 1 | Axum middleware | 3h | 3-4h | Medium (no prior art) |
| Unit 3 | Builder pattern | 4h | 3-4h | High (standard pattern) |
| Unit 5 | Remove duplication | 30m | 5m | High (delete 7 lines) |
| Unit 6 | Extract macro | 2h | 1-2h | High (macro_rules!) |

**Total**: 7.5h estimated → 7-10h realistic

### Documentation Work (5.5 hours)

| Unit | Task | Estimate | Actual Effort | Confidence |
|------|------|----------|---------------|------------|
| Unit 4 | Policy guide | 2h | 2h | High |
| Unit 7 | Axum example | 1h | 1h | High |
| Unit 8 | Use constants | 30m | 30m | High |
| Unit 9 | Field reference | 1h | 1h | High |
| Unit 10 | Troubleshooting | 1h | 1h | High |

**Total**: 5.5h estimated → 5.5h realistic

---

## Recommendations

### Immediate Actions (High ROI)

1. **Unit 5** (5 min): Remove duplicate logic in `auto.rs`
   - Delete lines 53-59
   - Run tests to verify
   - **ROI**: Instant code quality improvement

2. **Unit 8** (30 min): Update examples to use constants
   - Find/replace in README
   - **ROI**: Type safety, discoverability

3. **Unit 11** (0 min): ✅ Already done
   - Documentation corrected
   - **ROI**: Removed false blocker

### High-Value Work (Medium Effort)

4. **Unit 1** (3-4h): Axum middleware
   - Unblocks FT-004 integration
   - **ROI**: Reduces friction for all Axum users

5. **Unit 3** (3-4h): Builder pattern
   - Reduces code by 43% (7 lines → 4 lines)
   - **ROI**: Significant DX improvement

### Lower Priority (Nice to Have)

6. **Unit 6** (1-2h): Extract event macro
   - Saves ~100 lines of duplication
   - **ROI**: Maintainability

7. **Units 4, 7, 9, 10** (5.5h): Documentation improvements
   - **ROI**: Adoption, support reduction

---

## Conclusion

### What We Learned

1. ✅ **Unit 2 claim was wrong**: HeaderLike example is already correct
2. ✅ **Unit 11 claim was wrong**: Redaction already exceeds performance target
3. ✅ **Units 5, 6 confirmed**: Real code duplication exists
4. ❌ **Units 1, 3, 12 confirmed**: Features don't exist, need implementation
5. ⚠️ **Unit 8 confirmed**: Constants exist but unused

### Actual Work Remaining

**Code**: 7-10 hours (was: 20 hours)  
**Docs**: 5.5 hours (was: 10 hours)  
**Total**: 12.5-15.5 hours (was: 30 hours)

**Efficiency gain**: 50% reduction in estimated work

### Priority Order (By ROI)

1. Unit 5 (5 min) - Quick win
2. Unit 8 (30 min) - Quick win
3. Unit 1 (3-4h) - Unblocks adoption
4. Unit 3 (3-4h) - Major DX improvement
5. Unit 6 (1-2h) - Code quality
6. Units 4, 7, 9, 10 (5.5h) - Documentation

---

**Audit complete. Plan is now grounded in actual source code.** ✅
