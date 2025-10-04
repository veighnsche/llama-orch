# 🎯 Custom Narration System — Implementation Plan

**Project**: Build Our Own Custom Narration System  
**Decision**: Cuteness Pays the Bills! 🎀  
**Timeline**: 4 weeks  
**Status**: Planning Complete ✅

---

## 📋 Executive Summary

We're building a **custom narration system** with proc macros, auto-inferred actors, template interpolation, compile-time editorial enforcement, and built-in cute/story modes. This is **not** generic tracing — this is uniquely ours.

**Why**: 
- 🎀 Cute mode is our **brand** — needs to be first-class
- 🎭 Story mode is **unique** — no other library has it
- 🎨 Editorial enforcement is **our standard** — compile-time validation
- 🔒 Security is **built-in** — automatic redaction
- 📊 Proof bundles are **our workflow** — seamless integration
- 💝 Brand differentiation matters

---

## 🗓️ 4-Week Timeline

### Week 1: Core Proc Macro Crate
**Goal**: Create `observability-narration-macros` with basic functionality

### Week 2: Narration Core Enhancement
**Goal**: Add WARN/ERROR/FATAL levels + lightweight trace macros

### Week 3: Editorial Enforcement & Optimization
**Goal**: Compile-time validation + conditional compilation

### Week 4: Integration, Testing & Rollout
**Goal**: Migrate services + BDD tests + proof bundle integration

---

## 📦 Work Breakdown Structure

---

## **EXISTING CODE AUDIT**

### **What We Already Have** ✅
- ✅ `src/lib.rs` - Core `narrate()` function with `NarrationFields`
- ✅ `src/trace.rs` - All 6 lightweight trace macros (`trace_tiny!`, `trace_with_correlation!`, `trace_enter!`, `trace_exit!`, `trace_loop!`, `trace_state!`)
- ✅ `src/redaction.rs` - Secret redaction with regex patterns (Bearer tokens, API keys, UUIDs)
- ✅ `src/capture.rs` - Complete `CaptureAdapter` for BDD tests with assertions
- ✅ `src/auto.rs` - Auto-injection helpers (`narrate_auto`, `narrate_full`)
- ✅ `src/http.rs` - HTTP header propagation
- ✅ `src/otel.rs` - OpenTelemetry integration
- ✅ `Cargo.toml` - Basic features (`otel`, `test-support`)

### **What We Need to Build** 🚧
- 🚧 **NEW**: `bin/shared-crates/narration-macros/` - Proc macro crate
- 🚧 **ENHANCE**: Add WARN/ERROR/FATAL levels to `src/lib.rs`
- 🚧 **ENHANCE**: Add conditional compilation to `src/trace.rs`
- 🚧 **ENHANCE**: Add feature flags to `Cargo.toml`
- 🚧 **ENHANCE**: Add compile-time validation to proc macros

---

## **WEEK 1: Core Proc Macro Crate**

### **Unit 1.1: Project Setup** (Day 1)
**Owner**: Narration Core Team  
**Effort**: 4 hours

**Tasks**:
- [ ] Create `bin/shared-crates/narration-macros/` directory
- [ ] Create `Cargo.toml` with proc-macro configuration
- [ ] Add dependencies: `syn = "2.0"`, `quote = "1.0"`, `proc-macro2 = "1.0"`
- [ ] Set up basic module structure (`src/lib.rs`, `src/trace_fn.rs`, `src/narrate.rs`)
- [ ] Add to workspace `Cargo.toml`
- [ ] Create `README.md` with API overview

**Concrete File Changes**:
```toml
# NEW FILE: bin/shared-crates/narration-macros/Cargo.toml
[package]
name = "observability-narration-macros"
version = "0.0.0"
edition = "2021"
publish = false
license = "GPL-3.0-or-later"

[lib]
proc-macro = true

[dependencies]
syn = { version = "2.0", features = ["full", "extra-traits"] }
quote = "1.0"
proc-macro2 = "1.0"
```

```rust
// NEW FILE: bin/shared-crates/narration-macros/src/lib.rs
extern crate proc_macro;

mod actor_inference;
mod trace_fn;
mod narrate;
mod template;

pub use trace_fn::trace_fn;
pub use narrate::narrate;
```

**Deliverables**:
- `bin/shared-crates/narration-macros/Cargo.toml`
- `bin/shared-crates/narration-macros/src/lib.rs`
- `bin/shared-crates/narration-macros/README.md`

**Dependencies**: None

---

### **Unit 1.2: Actor Inference Module** (Day 1-2)
**Owner**: Narration Core Team  
**Effort**: 8 hours

**Tasks**:
- [ ] Implement `infer_actor_from_module_path()` function
- [ ] Parse module path from `ItemFn` context
- [ ] Extract service name (e.g., `llama_orch::orchestratord` → `"orchestratord"`)
- [ ] Handle edge cases (nested modules, tests, examples)
- [ ] Write unit tests for actor inference
- [ ] Document inference algorithm

**Deliverables**:
- `src/actor_inference.rs`
- Unit tests for actor inference
- Documentation with examples

**Dependencies**: Unit 1.1

---

### **Unit 1.3: `#[trace_fn]` Proc Macro** (Day 2-3)
**Owner**: Narration Core Team  
**Effort**: 12 hours

**Tasks**:
- [ ] Implement basic `#[trace_fn]` attribute macro
- [ ] Parse `ItemFn` (function signature, body, attributes)
- [ ] Generate entry trace with auto-inferred actor
- [ ] Generate exit trace with result/error handling
- [ ] Add automatic timing (`Instant::now()`)
- [ ] Handle `Result<T, E>` return types
- [ ] Support `?` operator in function body
- [ ] Add conditional compilation (`#[cfg(feature = "trace-enabled")]`)
- [ ] Write expansion tests (verify generated code)

**Deliverables**:
- `src/trace_fn.rs`
- Expansion tests
- Documentation with examples

**Dependencies**: Unit 1.2

**⏱️ PERFORMANCE TEAM REVIEW REQUIRED**:
- Review generated code for performance overhead
- Verify timing measurement doesn't add significant latency
- Approve before merging

---

### **Unit 1.4: Template Interpolation Engine** (Day 3-4)
**Owner**: Narration Core Team  
**Effort**: 10 hours

**Tasks**:
- [ ] Implement template parser (extract `{variable}` placeholders)
- [ ] Build variable extraction from function context
- [ ] Generate `format!()` calls with interpolated variables
- [ ] Handle nested fields (e.g., `{result.worker_id}`)
- [ ] Support `{elapsed_ms}` special variable
- [ ] Write template parsing tests
- [ ] Document template syntax

**Deliverables**:
- `src/template.rs`
- Template parsing tests
- Template syntax documentation

**Dependencies**: None

---

### **Unit 1.5: `#[narrate(...)]` Proc Macro** (Day 4-5)
**Owner**: Narration Core Team  
**Effort**: 12 hours

**Tasks**:
- [ ] Implement `#[narrate(...)]` attribute macro
- [ ] Parse attribute arguments (`actor`, `action`, `human`, `cute`, `story`)
- [ ] Integrate template interpolation engine
- [ ] Generate `narrate()` calls with interpolated fields
- [ ] Handle success/error cases separately
- [ ] Add automatic timing
- [ ] Support optional actor (auto-inferred if not provided)
- [ ] Write expansion tests

**Deliverables**:
- `src/narrate.rs`
- Expansion tests
- Documentation with examples

**Dependencies**: Unit 1.2, Unit 1.4

---

## **WEEK 2: Narration Core Enhancement**

### **Unit 2.1: Add WARN/ERROR/FATAL Levels** (Day 6)
**Owner**: Narration Core Team  
**Effort**: 6 hours

**Tasks**:
- [ ] Add `NarrationLevel` enum (Mute, Trace, Debug, Info, Warn, Error, Fatal)
- [ ] Implement `narrate_warn()`, `narrate_error()`, `narrate_fatal()` functions
- [ ] Map levels to `tracing::Level` (Fatal → Error)
- [ ] Update `narrate()` to accept level parameter
- [ ] Add level-specific field validation
- [ ] Write unit tests for each level

**Concrete Code Changes**:
```rust
// MODIFY: src/lib.rs (add after NarrationFields struct)

/// Narration logging level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NarrationLevel {
    Mute,   // No output
    Trace,  // Ultra-fine detail
    Debug,  // Developer diagnostics
    Info,   // Narration backbone (default)
    Warn,   // Anomalies & degradations
    Error,  // Operational failures
    Fatal,  // Unrecoverable errors
}

impl NarrationLevel {
    fn to_tracing_level(&self) -> Option<Level> {
        match self {
            NarrationLevel::Mute => None,
            NarrationLevel::Trace => Some(Level::TRACE),
            NarrationLevel::Debug => Some(Level::DEBUG),
            NarrationLevel::Info => Some(Level::INFO),
            NarrationLevel::Warn => Some(Level::WARN),
            NarrationLevel::Error => Some(Level::ERROR),
            NarrationLevel::Fatal => Some(Level::ERROR), // tracing doesn't have FATAL
        }
    }
}

/// Emit narration at a specific level
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    let Some(tracing_level) = level.to_tracing_level() else {
        return; // MUTE - no output
    };
    
    // Apply redaction
    let human = redact_secrets(&fields.human, RedactionPolicy::default());
    let cute = fields.cute.as_ref().map(|c| redact_secrets(c, RedactionPolicy::default()));
    let story = fields.story.as_ref().map(|s| redact_secrets(s, RedactionPolicy::default()));
    
    // Emit at appropriate level
    match tracing_level {
        Level::TRACE => event!(Level::TRACE, actor = fields.actor, action = fields.action, /* ... */),
        Level::DEBUG => event!(Level::DEBUG, actor = fields.actor, action = fields.action, /* ... */),
        Level::INFO => event!(Level::INFO, actor = fields.actor, action = fields.action, /* ... */),
        Level::WARN => event!(Level::WARN, actor = fields.actor, action = fields.action, /* ... */),
        Level::ERROR => event!(Level::ERROR, actor = fields.actor, action = fields.action, /* ... */),
    }
    
    #[cfg(any(test, feature = "test-support"))]
    capture::notify(fields);
}

/// Emit WARN-level narration
pub fn narrate_warn(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Warn)
}

/// Emit ERROR-level narration
pub fn narrate_error(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Error)
}

/// Emit FATAL-level narration
pub fn narrate_fatal(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Fatal)
}

// Keep existing narrate() as INFO-level
pub fn narrate(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Info)
}
```

**Deliverables**:
- Updated `src/lib.rs` with new levels
- Level-specific functions
- Unit tests

**Dependencies**: None

---

### **Unit 2.2: Lightweight Trace Macros - Add Conditional Compilation** (Day 6-7)
**Owner**: Narration Core Team  
**Effort**: 4 hours (REDUCED - macros already exist!)

**Current Status**: ✅ All 6 trace macros already implemented in `src/trace.rs`
- ✅ `trace_tiny!()` - line 50-60
- ✅ `trace_with_correlation!()` - line 79-90
- ✅ `trace_enter!()` - line 106-116
- ✅ `trace_exit!()` - line 139-149
- ✅ `trace_loop!()` - line 165-175
- ✅ `trace_state!()` - line 188-199

**Tasks** (ONLY need to add conditional compilation):
- [ ] Wrap all macros with `#[cfg(feature = "trace-enabled")]`
- [ ] Add no-op versions for production builds
- [ ] Write compilation tests (verify no-op in production)
- [ ] Add performance benchmarks

**Concrete Code Changes**:
```rust
// MODIFY: src/trace.rs (wrap each macro)

#[cfg(feature = "trace-enabled")]
#[macro_export]
macro_rules! trace_tiny {
    ($actor:expr, $action:expr, $target:expr, $human:expr) => {
        tracing::trace!(
            actor = $actor,
            action = $action,
            target = $target,
            human = $human,
            "trace"
        );
    };
}

#[cfg(not(feature = "trace-enabled"))]
#[macro_export]
macro_rules! trace_tiny {
    ($actor:expr, $action:expr, $target:expr, $human:expr) => {
        // No-op in production
    };
}

// Repeat for all 6 macros: trace_with_correlation, trace_enter, trace_exit, trace_loop, trace_state
```

**Deliverables**:
- Updated `src/trace.rs` with conditional compilation
- Compilation tests
- Performance benchmarks

**Dependencies**: None

**⏱️ PERFORMANCE TEAM REVIEW REQUIRED**:
- Verify ~2% overhead in dev builds
- Verify 0% overhead in production builds (code removed)
- Benchmark against full `narrate()` (~10x faster target)

---

### **Unit 2.3: Secret Redaction Enhancement** (Day 7-8)
**Owner**: Narration Core Team  
**Effort**: 4 hours (REDUCED - redaction already exists!)

**Current Status**: ✅ Redaction already implemented in `src/redaction.rs`
- ✅ `RedactionPolicy` struct with configurable options
- ✅ Regex patterns with `OnceLock` caching (already optimized!)
- ✅ Bearer token pattern (line 36-42)
- ✅ API key pattern (line 44-50)
- ✅ UUID pattern (line 52-58)
- ✅ `redact_secrets()` function (line 71-87)
- ✅ Comprehensive tests (line 89-153)

**Tasks** (ONLY need minor enhancements):
- [ ] Add JWT token pattern
- [ ] Add private key pattern (-----BEGIN PRIVATE KEY-----)
- [ ] Add password pattern in URLs
- [ ] Document all patterns with examples
- [ ] Add timing-safety verification tests

**Concrete Code Changes**:
```rust
// MODIFY: src/redaction.rs (add new patterns)

static JWT_PATTERN: OnceLock<Regex> = OnceLock::new();
static PRIVATE_KEY_PATTERN: OnceLock<Regex> = OnceLock::new();
static URL_PASSWORD_PATTERN: OnceLock<Regex> = OnceLock::new();

fn jwt_regex() -> &'static Regex {
    JWT_PATTERN.get_or_init(|| {
        // Match JWT tokens (header.payload.signature)
        Regex::new(r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+")
            .expect("BUG: JWT regex pattern is invalid")
    })
}

fn private_key_regex() -> &'static Regex {
    PRIVATE_KEY_PATTERN.get_or_init(|| {
        // Match private key blocks
        Regex::new(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----")
            .expect("BUG: private key regex pattern is invalid")
    })
}

fn url_password_regex() -> &'static Regex {
    URL_PASSWORD_PATTERN.get_or_init(|| {
        // Match passwords in URLs (user:password@host)
        Regex::new(r"://[^:]+:([^@]+)@")
            .expect("BUG: URL password regex pattern is invalid")
    })
}

// MODIFY: RedactionPolicy struct
pub struct RedactionPolicy {
    pub mask_bearer_tokens: bool,
    pub mask_api_keys: bool,
    pub mask_uuids: bool,
    pub mask_jwt_tokens: bool,      // NEW
    pub mask_private_keys: bool,    // NEW
    pub mask_url_passwords: bool,   // NEW
    pub replacement: String,
}

// MODIFY: redact_secrets() function
pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String {
    let mut result = text.to_string();
    
    if policy.mask_bearer_tokens {
        result = bearer_token_regex().replace_all(&result, &policy.replacement).to_string();
    }
    if policy.mask_api_keys {
        result = api_key_regex().replace_all(&result, &policy.replacement).to_string();
    }
    if policy.mask_uuids {
        result = uuid_regex().replace_all(&result, &policy.replacement).to_string();
    }
    if policy.mask_jwt_tokens {
        result = jwt_regex().replace_all(&result, &policy.replacement).to_string();
    }
    if policy.mask_private_keys {
        result = private_key_regex().replace_all(&result, &policy.replacement).to_string();
    }
    if policy.mask_url_passwords {
        result = url_password_regex().replace_all(&result, &policy.replacement).to_string();
    }
    
    result
}
```

**Deliverables**:
- Enhanced `src/redaction.rs` with 3 new patterns
- Additional redaction tests
- Documentation

**Dependencies**: None

**🎭 AUTH-MIN TEAM REVIEW REQUIRED**:
- Verify all secret patterns are covered (JWT, private keys, URL passwords)
- Ensure no timing attacks via redaction (regex already uses OnceLock caching)
- Approve redaction algorithm
- Sign off on security guarantees

---

### **Unit 2.4: Correlation ID Helpers** (Day 8-9)
**Owner**: Narration Core Team  
**Effort**: 6 hours

**Tasks**:
- [ ] Implement `CorrelationId::generate()` (UUID v4)
- [ ] Implement `CorrelationId::from_header()` (HTTP extraction)
- [ ] Implement `CorrelationId::propagate()` (downstream forwarding)
- [ ] Add validation (format, length)
- [ ] Write unit tests
- [ ] Document correlation ID patterns

**Deliverables**:
- `src/correlation.rs`
- Unit tests
- Documentation

**Dependencies**: None

---

### **Unit 2.5: Tracing Backend Integration** (Day 9-10)
**Owner**: Narration Core Team  
**Effort**: 8 hours

**Tasks**:
- [ ] Integrate with `tracing` crate as backend
- [ ] Map `NarrationLevel` to `tracing::Level`
- [ ] Emit structured events via `tracing::event!()`
- [ ] Support `tracing-subscriber` for output formatting
- [ ] Add JSON output support
- [ ] Test with `RUST_LOG` environment variable
- [ ] Document integration patterns

**Deliverables**:
- Updated `src/lib.rs` with tracing integration
- Integration tests
- Documentation

**Dependencies**: Unit 2.1

---

## **WEEK 3: Editorial Enforcement & Optimization**

### **Unit 3.1: Compile-Time Length Validation** (Day 11-12)
**Owner**: Narration Core Team  
**Effort**: 10 hours

**Tasks**:
- [ ] Add length validation in `#[narrate(...)]` macro
- [ ] Enforce ≤100 chars for INFO `human` field
- [ ] Enforce ≤120 chars for WARN `human` field
- [ ] Enforce ≤150 chars for ERROR `human` field
- [ ] Generate helpful compile errors with suggestions
- [ ] Write compile-fail tests
- [ ] Document length limits

**Deliverables**:
- Updated `src/narrate.rs` with validation
- Compile-fail tests
- Error message documentation

**Dependencies**: Unit 1.5

---

### **Unit 3.2: SVO Structure Validation** (Day 12-13)
**Owner**: Narration Core Team  
**Effort**: 12 hours

**Tasks**:
- [ ] Implement basic SVO (Subject-Verb-Object) parser
- [ ] Detect passive voice patterns
- [ ] Suggest active voice alternatives
- [ ] Add compile warnings for violations
- [ ] Make validation optional (feature flag)
- [ ] Write validation tests
- [ ] Document SVO guidelines

**Deliverables**:
- `src/svo_validator.rs`
- Validation tests
- SVO guidelines documentation

**Dependencies**: None

**Note**: This is ambitious for Week 3. Consider moving to Week 4 if timeline is tight.

---

### **Unit 3.3: Feature Flags & Conditional Compilation** (Day 13-14)
**Owner**: Narration Core Team  
**Effort**: 8 hours

**Tasks**:
- [ ] Define feature flags in `Cargo.toml`
  - `default = ["trace-enabled", "debug-enabled", "cute-mode"]`
  - `trace-enabled`, `debug-enabled`, `cute-mode`, `production`
- [ ] Implement conditional compilation in proc macros
- [ ] Test production builds (verify code removal)
- [ ] Measure binary size reduction
- [ ] Document build profiles
- [ ] Create build scripts for each profile

**Deliverables**:
- Updated `Cargo.toml` with features
- Build profile documentation
- Binary size comparison

**Dependencies**: Unit 1.3, Unit 1.5

**⏱️ PERFORMANCE TEAM REVIEW REQUIRED**:
- Verify 0% overhead in production builds
- Measure binary size reduction (~5 MB target)
- Approve conditional compilation strategy

---

### **Unit 3.4: Performance Benchmarks** (Day 14-15)
**Owner**: Narration Core Team  
**Effort**: 8 hours

**Tasks**:
- [ ] Create benchmark suite (using `criterion`)
- [ ] Benchmark `narrate()` vs `trace_tiny!()`
- [ ] Benchmark `#[trace_fn]` overhead
- [ ] Benchmark template interpolation
- [ ] Benchmark redaction performance
- [ ] Document performance characteristics
- [ ] Set up CI benchmarks (prevent regressions)

**Deliverables**:
- `benches/narration_benchmarks.rs`
- Benchmark results documentation
- CI benchmark configuration

**Dependencies**: All Week 1-2 units

**⏱️ PERFORMANCE TEAM REVIEW REQUIRED**:
- Verify ~2% overhead for trace macros
- Verify ~25% overhead for full `narrate()`
- Approve performance characteristics
- Sign off on benchmarks

---

## **WEEK 4: Integration, Testing & Rollout**

### **Unit 4.1: BDD Tests for Cute/Story Modes** (Day 16-17)
**Owner**: Narration Core Team  
**Effort**: 10 hours

**Tasks**:
- [ ] Write BDD scenarios for cute mode
- [ ] Write BDD scenarios for story mode
- [ ] Test template interpolation in cute/story fields
- [ ] Test redaction in cute/story fields
- [ ] Test conditional compilation (cute mode disabled)
- [ ] Integrate with existing BDD runner
- [ ] Document BDD test patterns

**Deliverables**:
- `bdd/features/cute_mode.feature`
- `bdd/features/story_mode.feature`
- BDD step definitions
- Documentation

**Dependencies**: Unit 1.5, Unit 2.3

---

### **Unit 4.2: Proof Bundle Integration** (Day 17-18)
**Owner**: Narration Core Team  
**Effort**: 8 hours

**Tasks**:
- [ ] Integrate `CaptureAdapter` with proof bundles
- [ ] Auto-capture narration events in test runs
- [ ] Store narration in proof bundle artifacts
- [ ] Add assertions for narration presence
- [ ] Test with `LLORCH_PROOF_DIR` environment variable
- [ ] Document proof bundle integration
- [ ] Update proof bundle spec

**Deliverables**:
- Updated `src/capture.rs` with proof bundle integration
- Integration tests
- Documentation

**Dependencies**: Unit 4.1

---

### **Unit 4.3: Editorial Enforcement Tests** (Day 18)
**Owner**: Narration Core Team  
**Effort**: 6 hours

**Tasks**:
- [ ] Write compile-fail tests for length violations
- [ ] Write compile-fail tests for SVO violations
- [ ] Test helpful error messages
- [ ] Test suggestion quality
- [ ] Document editorial enforcement
- [ ] Create enforcement checklist

**Deliverables**:
- Compile-fail test suite
- Editorial enforcement documentation
- Enforcement checklist

**Dependencies**: Unit 3.1, Unit 3.2

---

### **Unit 4.4: Service Migration — orchestratord** (Day 19)
**Owner**: Narration Core Team + orchestratord Team  
**Effort**: 8 hours

**Tasks**:
- [ ] Add `#[trace_fn]` to all functions
- [ ] Convert INFO events to `#[narrate(...)]`
- [ ] Add cute mode to user-facing events
- [ ] Test with `RUST_LOG=trace`
- [ ] Verify correlation ID propagation
- [ ] Run BDD suite
- [ ] Document migration patterns

**Deliverables**:
- Updated orchestratord with narration
- Migration documentation
- BDD test results

**Dependencies**: All Week 1-3 units

**🎭 AUTH-MIN TEAM REVIEW REQUIRED**:
- Verify no secrets in narration events
- Verify timing-safe operations maintained
- Approve migration

**⏱️ PERFORMANCE TEAM REVIEW REQUIRED**:
- Verify no performance regression
- Measure overhead in production build
- Approve migration

---

### **Unit 4.5: Service Migration — pool-managerd** (Day 19-20)
**Owner**: Narration Core Team + pool-managerd Team  
**Effort**: 8 hours

**Tasks**:
- [ ] Add `#[trace_fn]` to all functions
- [ ] Convert INFO events to `#[narrate(...)]`
- [ ] Add cute mode to user-facing events
- [ ] Test with `RUST_LOG=trace`
- [ ] Verify correlation ID propagation
- [ ] Run BDD suite
- [ ] Document migration patterns

**Deliverables**:
- Updated pool-managerd with narration
- Migration documentation
- BDD test results

**Dependencies**: Unit 4.4

**🎭 AUTH-MIN TEAM REVIEW REQUIRED**:
- Verify no secrets in narration events
- Verify timing-safe operations maintained
- Approve migration

**⏱️ PERFORMANCE TEAM REVIEW REQUIRED**:
- Verify no performance regression
- Measure overhead in production build
- Approve migration

---

### **Unit 4.6: Service Migration — worker-orcd** (Day 20)
**Owner**: Narration Core Team + worker-orcd Team  
**Effort**: 8 hours

**Tasks**:
- [ ] Add `#[trace_fn]` to all functions
- [ ] Add `trace_enter!()`/`trace_exit!()` to FFI boundaries
- [ ] Convert INFO events to `#[narrate(...)]`
- [ ] Add cute mode to user-facing events
- [ ] Test with `RUST_LOG=trace`
- [ ] Verify correlation ID propagation
- [ ] Run BDD suite
- [ ] Document migration patterns

**Deliverables**:
- Updated worker-orcd with narration
- Migration documentation
- BDD test results

**Dependencies**: Unit 4.5

**🎭 AUTH-MIN TEAM REVIEW REQUIRED**:
- Verify no secrets in narration events
- Verify timing-safe operations maintained
- Approve migration

**⏱️ PERFORMANCE TEAM REVIEW REQUIRED**:
- Verify no performance regression in FFI paths
- Measure overhead in production build
- Approve migration

---

### **Unit 4.7: CI/CD Pipeline Updates** (Day 20)
**Owner**: Narration Core Team  
**Effort**: 6 hours

**Tasks**:
- [ ] Update CI to build with multiple profiles (dev, staging, production)
- [ ] Add benchmark regression tests
- [ ] Add compile-fail tests to CI
- [ ] Add proof bundle artifact collection
- [ ] Update deployment scripts
- [ ] Document CI/CD changes

**Deliverables**:
- Updated `.github/workflows/engine-ci.yml`
- CI/CD documentation
- Deployment scripts

**Dependencies**: All Week 4 units

---

## 🔍 Cross-Team Review Points

### **🎭 auth-min Team Reviews**

**Unit 2.3: Secret Redaction Enhancement**
- **Review Focus**: Verify all secret patterns covered, no timing attacks
- **Deliverable**: Security sign-off document
- **Timeline**: End of Week 2 (Day 10)

**Unit 4.4-4.6: Service Migrations**
- **Review Focus**: Verify no secrets in narration, timing-safe operations maintained
- **Deliverable**: Migration approval per service
- **Timeline**: Week 4 (Days 19-20)

**Required Signature**: `Guarded by auth-min Team 🎭`

---

### **⏱️ Performance Team Reviews**

**Unit 1.3: `#[trace_fn]` Proc Macro**
- **Review Focus**: Generated code performance, timing measurement overhead
- **Deliverable**: Performance approval
- **Timeline**: End of Week 1 (Day 5)

**Unit 2.2: Lightweight Trace Macros**
- **Review Focus**: Verify ~2% dev overhead, 0% production overhead
- **Deliverable**: Benchmark approval
- **Timeline**: Week 2 (Day 7)

**Unit 3.3: Feature Flags & Conditional Compilation**
- **Review Focus**: Verify code removal, binary size reduction
- **Deliverable**: Build profile approval
- **Timeline**: Week 3 (Day 14)

**Unit 3.4: Performance Benchmarks**
- **Review Focus**: Verify performance characteristics, approve benchmarks
- **Deliverable**: Benchmark sign-off
- **Timeline**: Week 3 (Day 15)

**Unit 4.4-4.6: Service Migrations**
- **Review Focus**: Verify no performance regression
- **Deliverable**: Migration approval per service
- **Timeline**: Week 4 (Days 19-20)

**Required Signature**: `Optimized by Performance Team ⏱️`

---

## 📊 Success Criteria

### **Week 1 Success**:
- ✅ `observability-narration-macros` crate created
- ✅ `#[trace_fn]` generates correct code
- ✅ `#[narrate(...)]` supports template interpolation
- ✅ Actor inference works from module path
- ✅ All expansion tests pass

### **Week 2 Success**:
- ✅ WARN/ERROR/FATAL levels implemented
- ✅ All lightweight trace macros functional
- ✅ Secret redaction enhanced and approved by auth-min
- ✅ Tracing backend integration complete
- ✅ All unit tests pass

### **Week 3 Success**:
- ✅ Compile-time length validation working
- ✅ Feature flags configured
- ✅ Production builds have 0% overhead (verified)
- ✅ Performance benchmarks approved by Performance Team
- ✅ Binary size reduced by ~5 MB

### **Week 4 Success**:
- ✅ All three services migrated (orchestratord, pool-managerd, worker-orcd)
- ✅ BDD tests pass for cute/story modes
- ✅ Proof bundle integration working
- ✅ CI/CD pipelines updated
- ✅ All team reviews complete (auth-min ✅, Performance Team ✅)

---

## 🚨 Risk Mitigation

### **Risk 1: Proc Macro Complexity**
- **Mitigation**: Start with simple expansion, iterate
- **Fallback**: Use declarative macros if proc macros too complex

### **Risk 2: SVO Validation Too Ambitious**
- **Mitigation**: Make it optional (feature flag), focus on length validation first
- **Fallback**: Move to Week 5 if needed

### **Risk 3: Service Migration Takes Longer**
- **Mitigation**: Prioritize orchestratord, defer worker-orcd if needed
- **Fallback**: Extend timeline by 2-3 days

### **Risk 4: Performance Regression**
- **Mitigation**: Continuous benchmarking, Performance Team reviews
- **Fallback**: Optimize hot paths, disable features if needed

### **Risk 5: Security Issues in Redaction**
- **Mitigation**: Early auth-min review (Week 2), comprehensive tests
- **Fallback**: Use existing redaction, enhance later

---

## 📅 Milestones

| Milestone | Date | Deliverables |
|-----------|------|--------------|
| **M1: Proc Macros Complete** | End of Week 1 | `#[trace_fn]`, `#[narrate(...)]`, actor inference |
| **M2: Core Enhancement Complete** | End of Week 2 | WARN/ERROR/FATAL, trace macros, redaction |
| **M3: Optimization Complete** | End of Week 3 | Validation, feature flags, benchmarks |
| **M4: Rollout Complete** | End of Week 4 | All services migrated, CI/CD updated |

---

## 📚 Documentation Deliverables

- ✅ `FINAL_SUMMARY.md` — Complete overview (already done)
- ✅ `EXISTING_SOLUTIONS.md` — Why we built our own (already done)
- ✅ `ERGONOMIC_TRACING.md` — Proc macro design (already done)
- ✅ `DEVELOPER_EXPERIENCE.md` — Developer guidelines (already done)
- ✅ `CONDITIONAL_COMPILATION.md` — Zero overhead (already done)
- ✅ `TRACE_OPTIMIZATION.md` — Performance (already done)
- ✅ `LOGGING_LEVELS.md` — 7-level spec (already done)
- ✅ `TRACING_INTEGRATION_REVIEW.md` — Editorial review (already done)
- ✅ `REVIEW_SUMMARY.md` — Executive summary (already done)
- [ ] `MIGRATION_GUIDE.md` — Service migration steps (Week 4)
- [ ] `API_REFERENCE.md` — Complete API docs (Week 4)
- [ ] `SECURITY_REVIEW.md` — auth-min sign-off (Week 2 & 4)
- [ ] `PERFORMANCE_REVIEW.md` — Performance Team sign-off (Week 3 & 4)

---

## 🎯 Team Coordination

### **Narration Core Team** (Primary Owner)
- Owns all implementation units
- Coordinates with auth-min and Performance Team
- Responsible for documentation and testing

### **auth-min Team** (Security Reviews)
- Reviews Unit 2.3 (Secret Redaction)
- Reviews Unit 4.4-4.6 (Service Migrations)
- Signs off on security guarantees
- **Signature Required**: `Guarded by auth-min Team 🎭`

### **Performance Team** (Performance Reviews)
- Reviews Unit 1.3 (`#[trace_fn]`)
- Reviews Unit 2.2 (Trace Macros)
- Reviews Unit 3.3 (Conditional Compilation)
- Reviews Unit 3.4 (Benchmarks)
- Reviews Unit 4.4-4.6 (Service Migrations)
- Signs off on performance characteristics
- **Signature Required**: `Optimized by Performance Team ⏱️`

### **Service Teams** (Migration Support)
- orchestratord Team: Supports Unit 4.4
- pool-managerd Team: Supports Unit 4.5
- worker-orcd Team: Supports Unit 4.6

---

## 💝 Final Notes

**This is not generic tracing. This is uniquely ours.**

We're building a narration system that:
- 🎀 Makes cute mode first-class (not an afterthought)
- 🎭 Supports story mode (dialogue-based narration)
- 🎨 Enforces editorial standards at compile time
- 🔒 Builds security in (automatic redaction)
- 📊 Integrates with our proof bundles
- 💝 Differentiates our brand

**Cuteness pays the bills. Let's make it happen.** 🚀

---

**With love, sass, and the confidence that cuteness pays the bills,**  
**The Narration Core Team** 🎭🎀

---

*May your proc macros be powerful, your actors be auto-inferred, and your narration be adorable!* 🎀

---

## 🎭 Security Review Section

**For auth-min Team**: Please review and sign off on security-critical units.

### Unit 2.3: Secret Redaction Enhancement
- [ ] Reviewed by auth-min Team
- [ ] All secret patterns verified
- [ ] No timing attack vectors identified
- [ ] Redaction algorithm approved
- [ ] **Signature**: ___________________________

### Unit 4.4: orchestratord Migration
- [ ] Reviewed by auth-min Team
- [ ] No secrets in narration events
- [ ] Timing-safe operations maintained
- [ ] **Signature**: ___________________________

### Unit 4.5: pool-managerd Migration
- [ ] Reviewed by auth-min Team
- [ ] No secrets in narration events
- [ ] Timing-safe operations maintained
- [ ] **Signature**: ___________________________

### Unit 4.6: worker-orcd Migration
- [ ] Reviewed by auth-min Team
- [ ] No secrets in narration events
- [ ] Timing-safe operations maintained
- [ ] **Signature**: ___________________________

---

## ⏱️ Performance Review Section

**For Performance Team**: Please review and sign off on performance-critical units.

### Unit 1.3: `#[trace_fn]` Proc Macro
- [ ] Reviewed by Performance Team
- [ ] Generated code overhead acceptable
- [ ] Timing measurement verified
- [ ] **Signature**: ___________________________

### Unit 2.2: Lightweight Trace Macros
- [ ] Reviewed by Performance Team
- [ ] ~2% dev overhead verified
- [ ] 0% production overhead verified
- [ ] **Signature**: ___________________________

### Unit 3.3: Conditional Compilation
- [ ] Reviewed by Performance Team
- [ ] Code removal verified
- [ ] Binary size reduction measured (~5 MB)
- [ ] **Signature**: ___________________________

### Unit 3.4: Performance Benchmarks
- [ ] Reviewed by Performance Team
- [ ] Benchmark methodology approved
- [ ] Performance characteristics verified
- [ ] **Signature**: ___________________________

### Unit 4.4: orchestratord Migration
- [ ] Reviewed by Performance Team
- [ ] No performance regression
- [ ] Production overhead measured
- [ ] **Signature**: ___________________________

### Unit 4.5: pool-managerd Migration
- [ ] Reviewed by Performance Team
- [ ] No performance regression
- [ ] Production overhead measured
- [ ] **Signature**: ___________________________

### Unit 4.6: worker-orcd Migration
- [ ] Reviewed by Performance Team
- [ ] No FFI path regression
- [ ] Production overhead measured
- [ ] **Signature**: ___________________________

---

---

## 📊 Revised Effort Estimates (Based on Existing Code)

### **Original Estimate**: 4 weeks (160 hours)
### **Revised Estimate**: 2.5 weeks (100 hours)

**Savings**: ~60 hours (37.5% reduction!)

### **Why the Reduction**:
- ✅ **Week 1 savings**: Trace macros already exist (save 8 hours)
- ✅ **Week 2 savings**: Redaction already exists (save 4 hours), auto-injection exists (save 8 hours), capture adapter exists (save 10 hours)
- ✅ **Week 3 savings**: SVO validation moved to optional/future (save 12 hours)
- ✅ **Week 4 savings**: BDD infrastructure already exists (save 6 hours)

### **Revised Timeline**:
- **Week 1**: Core proc macros (5 days → 4 days)
- **Week 2**: Enhancements (5 days → 3 days)
- **Week 3**: Optimization (5 days → 4 days)
- **Week 4**: Integration (5 days → 4 days)

**Total**: 15 days instead of 20 days

---

## ✅ Pre-Implementation Checklist

Before starting Week 1:
- [ ] Review existing code in `bin/shared-crates/narration-core/src/`
- [ ] Verify all dependencies are in workspace `Cargo.toml`
- [ ] Set up `bin/shared-crates/narration-macros/` directory
- [ ] Coordinate with auth-min team for review schedule
- [ ] Coordinate with Performance team for review schedule
- [ ] Create feature branch: `feat/custom-narration-system`

---

**Implementation Plan Complete** ✅  
**Verified Against Existing Code** ✅  
**Ready for Execution** 🚀  
**Cuteness Pays the Bills** 💝
