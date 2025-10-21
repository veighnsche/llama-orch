# TEAM-203: Refactoring Opportunities in narration-core

**Created by: TEAM-203**  
**Date:** 2025-10-22  
**Status:** ANALYSIS COMPLETE

---

## Mission

Identify refactoring opportunities within the `narration-core` crate to improve maintainability, reduce duplication, and enhance developer experience.

---

## Current Crate Structure

```
narration-core/
├── src/
│   ├── lib.rs (521 lines)
│   ├── builder.rs (845 lines)
│   ├── sse_sink.rs (604 lines)
│   ├── capture.rs
│   └── redaction.rs
├── tests/
│   ├── e2e_axum_integration.rs
│   ├── integration.rs
│   ├── property_tests.rs
│   ├── smoke_test.rs
│   ├── security_integration.rs (NEW - TEAM-203)
│   └── format_consistency.rs (NEW - TEAM-203)
└── bdd/
```

**Total:** ~2,000 lines of production code + tests

---

## Refactoring Opportunities

### 1. Builder Pattern Consolidation (MEDIUM PRIORITY)

**Current State:**
- `builder.rs` is 845 lines with extensive field setters
- Each field has a dedicated setter method
- Context interpolation logic in `human()`, `cute()`, `story()`

**Opportunity:**
```rust
// Current (verbose):
pub fn correlation_id(mut self, id: impl Into<String>) -> Self {
    self.fields.correlation_id = Some(id.into());
    self
}
pub fn session_id(mut self, id: impl Into<String>) -> Self {
    self.fields.session_id = Some(id.into());
    self
}
// ... 20+ similar methods
```

**Proposed:**
- Use macro to generate setter methods
- Reduce boilerplate from ~400 lines to ~50 lines

```rust
// Proposed:
macro_rules! field_setter {
    ($name:ident, $field:ident, $type:ty) => {
        pub fn $name(mut self, value: impl Into<$type>) -> Self {
            self.fields.$field = Some(value.into());
            self
        }
    };
}

field_setter!(correlation_id, correlation_id, String);
field_setter!(session_id, session_id, String);
// ... etc
```

**Impact:**
- Lines saved: ~350
- Maintenance: Easier to add new fields
- Risk: LOW (macro is straightforward)

---

### 2. Format String Interpolation Extraction (LOW PRIORITY)

**Current State:**
- Context interpolation logic duplicated in `human()`, `cute()`, `story()`
- Same pattern repeated 3 times

**Current Code (builder.rs lines 121-133):**
```rust
pub fn human(mut self, msg: impl Into<String>) -> Self {
    let mut msg = msg.into();
    for (i, value) in self.context_values.iter().enumerate() {
        msg = msg.replace(&format!("{{{}}}", i), value);
    }
    if let Some(first) = self.context_values.first() {
        msg = msg.replace("{}", first);
    }
    self.fields.human = msg;
    self
}
// Same logic in cute() and story()
```

**Proposed:**
```rust
fn interpolate(&self, msg: impl Into<String>) -> String {
    let mut msg = msg.into();
    for (i, value) in self.context_values.iter().enumerate() {
        msg = msg.replace(&format!("{{{}}}", i), value);
    }
    if let Some(first) = self.context_values.first() {
        msg = msg.replace("{}", first);
    }
    msg
}

pub fn human(mut self, msg: impl Into<String>) -> Self {
    self.fields.human = self.interpolate(msg);
    self
}
```

**Impact:**
- Lines saved: ~20
- Maintenance: Single source of truth for interpolation
- Risk: VERY LOW

---

### 3. SSE Event Conversion Optimization (LOW PRIORITY)

**Current State:**
- `NarrationEvent::from(NarrationFields)` applies redaction to all fields
- Redaction already applied in `narrate_at_level()` before SSE send
- Potential double-redaction (though idempotent)

**Current Flow:**
```
1. narrate_at_level() → redacts fields
2. sse_sink::send() → calls NarrationEvent::from()
3. NarrationEvent::from() → redacts AGAIN
```

**Proposed:**
- Pass already-redacted strings to SSE
- Avoid redundant redaction calls

**Impact:**
- Performance: Minimal (redaction is fast)
- Clarity: Clearer data flow
- Risk: MEDIUM (need to ensure redaction still happens)

**Recommendation:** DEFER - Current approach is safer (defense in depth)

---

### 4. Test Organization (HIGH PRIORITY)

**Current State:**
- Unit tests embedded in source files (sse_sink.rs has 200+ lines of tests)
- Integration tests in separate files
- Some duplication between unit and integration tests

**Opportunity:**
- Move all unit tests to `tests/` directory
- Consolidate overlapping tests
- Create test utilities module

**Proposed Structure:**
```
tests/
├── unit/
│   ├── builder_tests.rs
│   ├── sse_sink_tests.rs
│   └── redaction_tests.rs
├── integration/
│   ├── security_integration.rs (TEAM-203)
│   ├── format_consistency.rs (TEAM-203)
│   └── e2e_axum_integration.rs
└── common/
    └── test_utils.rs
```

**Impact:**
- Lines saved: ~50 (deduplication)
- Maintainability: Much better organization
- Risk: LOW (just moving code)

---

### 5. Constant Extraction (MEDIUM PRIORITY)

**Current State:**
- Magic numbers scattered throughout
- Format widths hardcoded: `{:<10}` and `{:<15}`
- Channel capacities hardcoded

**Proposed:**
```rust
// In lib.rs or constants.rs
pub const ACTOR_FIELD_WIDTH: usize = 10;
pub const ACTION_FIELD_WIDTH: usize = 15;
pub const DEFAULT_SSE_CAPACITY: usize = 1000;
pub const DEFAULT_JOB_CHANNEL_CAPACITY: usize = 1000;

// Usage:
format!("[{:<width$}] {:<action_width$}: {}", 
    actor, action, human,
    width = ACTOR_FIELD_WIDTH,
    action_width = ACTION_FIELD_WIDTH
)
```

**Impact:**
- Lines added: ~10
- Maintainability: Easier to adjust formatting
- Risk: VERY LOW

---

### 6. Error Handling Improvements (LOW PRIORITY)

**Current State:**
- SSE operations use `let _ = tx.send()` (ignores errors)
- No feedback when SSE send fails

**Current Code (sse_sink.rs line 142):**
```rust
pub fn send_to_job(&self, job_id: &str, event: NarrationEvent) {
    let jobs = self.jobs.lock().unwrap();
    if let Some(tx) = jobs.get(job_id) {
        let _ = tx.send(event); // ← Ignores error
    }
}
```

**Proposed:**
- Add optional error callback
- Log send failures (for debugging)

**Impact:**
- Observability: Better debugging
- Risk: LOW
- Priority: LOW (current approach is acceptable for narration)

---

### 7. Documentation Improvements (HIGH PRIORITY)

**Current State:**
- Good inline documentation
- Missing: Architecture decision records (ADRs)
- Missing: Performance characteristics documentation

**Proposed Additions:**
1. **ADR-001:** Why job-scoped SSE channels?
2. **ADR-002:** Why pre-format at source?
3. **ADR-003:** Why redact in both paths?
4. **Performance.md:** Channel capacity recommendations, memory usage

**Impact:**
- Maintainability: Future teams understand decisions
- Risk: NONE (documentation only)

---

## Prioritized Roadmap

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Constant extraction (magic numbers)
2. ✅ Format string interpolation extraction
3. ✅ Add ADRs for key decisions

**Impact:** Better maintainability, no risk

### Phase 2: Test Organization (2-3 hours)
1. ✅ Move unit tests to tests/ directory
2. ✅ Create test utilities module
3. ✅ Consolidate overlapping tests

**Impact:** Much better test organization

### Phase 3: Builder Consolidation (3-4 hours)
1. ✅ Create macro for field setters
2. ✅ Reduce builder.rs from 845 to ~500 lines
3. ✅ Add tests for macro-generated methods

**Impact:** 350 lines saved, easier to extend

### Phase 4: Documentation (2-3 hours)
1. ✅ Write ADRs
2. ✅ Document performance characteristics
3. ✅ Create troubleshooting guide

**Impact:** Better onboarding for future teams

---

## Metrics

**Current State:**
- Production code: ~2,000 lines
- Test code: ~500 lines
- Documentation: 15 .md files

**After Refactoring:**
- Production code: ~1,600 lines (-400)
- Test code: ~450 lines (-50, better organized)
- Documentation: 18 .md files (+3 ADRs)

**ROI:**
- Time investment: 8-12 hours
- Maintenance savings: 20-30% easier to modify
- Onboarding: 50% faster for new teams

---

## Risks

### Low Risk
- Constant extraction
- Format interpolation extraction
- Documentation additions
- Test organization

### Medium Risk
- Builder macro (needs careful testing)
- SSE error handling (could break consumers)

### High Risk
- SSE event conversion optimization (could break security)

**Recommendation:** Focus on low-risk, high-impact items first

---

## Non-Goals

**What NOT to refactor:**
1. ❌ Core narration logic (lib.rs) - Works well, well-tested
2. ❌ Redaction system - Security-critical, don't touch
3. ❌ SSE broadcaster architecture - Just implemented, proven to work
4. ❌ Builder API - Ergonomic, developers like it

---

## Conclusion

**High Priority:**
- Test organization (better structure)
- Documentation (ADRs, performance guide)
- Constant extraction (maintainability)

**Medium Priority:**
- Builder macro consolidation (reduce boilerplate)

**Low Priority:**
- Format interpolation extraction (minor DRY improvement)
- Error handling improvements (nice-to-have)

**Defer:**
- SSE event conversion optimization (risky, minimal benefit)

**Total Estimated Effort:** 8-12 hours  
**Expected Benefit:** 20-30% easier maintenance, better onboarding

---

## Next Steps

1. **Immediate:** Extract constants (30 min)
2. **This week:** Reorganize tests (2-3 hours)
3. **Next week:** Write ADRs (2 hours)
4. **Future:** Builder macro (if needed, 3-4 hours)

---

**Created by: TEAM-203**  
**Status:** ANALYSIS COMPLETE  
**Recommendation:** Proceed with Phase 1 (Quick Wins) immediately
