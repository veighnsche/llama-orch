# 🎯 Custom Narration System — Implementation Plan (FINAL)
**Project**: Build Our Own Custom Narration System  
**Decision**: Cuteness Pays the Bills! 🎀  
**Timeline**: 2.5 weeks (revised from 4 weeks)  
**Status**: ✅ APPROVED by Performance Team ⏱️  
**Final Authority**: Performance Team (vetoed 5 security requirements for performance)
---
## 📋 Executive Summary
We're building a **custom narration system** with proc macros, auto-inferred actors, template interpolation, compile-time editorial enforcement, and built-in cute/story modes. This is **not** generic tracing — this is uniquely ours.
**Why**: 
- 🎀 Cute mode is our **brand** — needs to be first-class
- 🎭 Story mode is **unique** — no other library has it
- 🎨 Editorial enforcement is **our standard** — compile-time validation
- 🔒 Security is **built-in** — automatic redaction
- 📊  are **our workflow** — seamless integration
- 💝 Brand differentiation matters
---
## 🗓️ 2.5-Week Timeline (Revised)
### Week 1: Core Proc Macro Crate (4 days)
**Goal**: Create `observability-narration-macros` with compile-time template expansion
**Performance Target**: <100ns template interpolation, zero heap allocations
### Week 2: Narration Core Enhancement (3 days)
**Goal**: Add WARN/ERROR/FATAL levels + optimize redaction
**Performance Target**: <5μs redaction (single-pass regex + Cow strings)
**🚫 REMOVED**: Unit 2.7 Sampling (Performance Team REJECTED)
### Week 3: Editorial Enforcement & Optimization (4 days)
**Goal**: Conditional compilation + simplified validation
**Performance Target**: 0ns overhead in production builds
**Performance Team Override**: Simplified Unicode validation (not comprehensive)
### Week 4: Integration, Testing & Rollout (4 days)
**Goal**: Migrate services + BDD tests + performance benchmarks
**Performance Target**: All benchmarks MUST pass before merge
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
- 🚧 **NEW**: `bin/shared-crates/narration-macros/` - Proc macro crate with compile-time template expansion
- 🚧 **ENHANCE**: Add WARN/ERROR/FATAL levels to `src/lib.rs`
- 🚧 **ENHANCE**: Add conditional compilation to `src/trace.rs`
- 🚧 **ENHANCE**: Optimize `src/redaction.rs` (single-pass regex + Cow strings)
- 🚧 **ENHANCE**: Add simplified Unicode validation to `src/lib.rs`
- 🚧 **ENHANCE**: Add feature flags to `Cargo.toml` (opt-in tracing)
- 🚧 **ENHANCE**: Add compile-time validation to proc macros
- 🚫 **REMOVED**: Unit 2.7 Sampling & Rate Limiting (Performance Team REJECTED)
### **Performance Team Decisions Applied** ⏱️
- ✅ **Tracing Opt-In**: Zero overhead in production (MANDATORY)
- ✅ **Template Interpolation**: Compile-time expansion + stack buffers (<100ns)
- ✅ **Redaction**: Single-pass regex + Cow strings (<5μs)
- ✅ **Unicode**: Simplified validation (ASCII fast path, not comprehensive)
- ✅ **CRLF**: Strip `\n`, `\r`, `\t` only (not all control chars)
- ✅ **Correlation ID**: Byte-level UUID validation (<100ns, no HMAC)
- 🚫 **Sampling**: REJECTED (use `RUST_LOG` instead)
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
**⏱️ PERFORMANCE TEAM COMMENT**:
- 🚨 **CRITICAL**: `Instant::now()` calls at entry/exit add ~20-50ns overhead per function. For hot-path functions called millions of times, this compounds to milliseconds. **RECOMMENDATION**: Add `#[cfg(feature = "trace-enabled")]` guard around timing code, ensure production builds have ZERO timing overhead.
- ⚡ **OPTIMIZATION**: Consider using `rdtsc` instruction directly for sub-nanosecond precision on x86_64 (via `core::arch::x86_64::_rdtsc`) instead of `Instant::now()` for ultra-hot paths.
- 📊 **BENCHMARK REQUIREMENT**: Measure overhead on empty function (baseline), 10-instruction function (typical), and 1000-instruction function (heavy). Target: <1% overhead for functions >100 instructions.
**🎭 AUTH-MIN SECURITY NOTE**:
- ⚠️ **Timing side-channel risk**: Function timing measurements could leak information about code paths taken (e.g., auth success vs failure paths). Ensure timing data is not exposed in user-facing contexts.
- ⚠️ **Actor inference spoofing**: Module path parsing via `#[path]` attributes could allow actor identity spoofing. Validate inferred actors against allowlist.
---
### **Unit 1.4: Template Interpolation Engine (COMPILE-TIME)** (Day 3-4)
**Owner**: Narration Core Team  
**Effort**: 8 hours (REDUCED - compile-time only)
**⏱️ PERFORMANCE TEAM DECISION**: Pre-compiled templates at macro expansion time (NOT runtime parsing)
**Tasks**:
- [ ] Parse template at macro expansion time (extract `{variable}` placeholders)
- [ ] Generate direct `write!()` calls (NOT `format!()`)
- [ ] Use stack buffers (ArrayString<256>) for templates <256 chars
- [ ] Fall back to heap allocation ONLY for >256 char templates
- [ ] Handle nested fields (e.g., `{result.worker_id}`)
- [ ] Support `{elapsed_ms}` special variable
- [ ] Write expansion tests (verify generated code)
- [ ] Document template syntax
**Concrete Implementation**:
```rust
// Proc macro generates:
use arrayvec::ArrayString;
let mut buf = ArrayString::<256>::new();
write!(&mut buf, "Dispatched job {} to worker {}", job_id, worker.id)?;
let human = buf.as_str();
```
**Performance Target**: <100ns interpolation, ZERO heap allocations for <256 char templates
**Deliverables**:
- `src/template.rs` (compile-time parser)
- Expansion tests
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
**⏱️ PERFORMANCE TEAM FINAL DECISION**:
- ✅ **APPROVED**: Compile-time template expansion (NOT runtime parsing)
- ✅ **APPROVED**: Stack buffers (ArrayString<256>) MANDATORY
- 🚫 **REJECTED**: Auth-min's requirement to escape ALL variables (50-100ns overhead)
- ✅ **FINAL**: Escape ONLY user-marked inputs with `#[user_input]` attribute
**Implementation**:
```rust
// Compile-time validation (FREE)
if template_literal.contains('{') || template_literal.contains('}') {
    return compile_error!("Template literal cannot contain braces");
}
// Runtime escaping ONLY for user inputs (opt-in)
#[narrate(
    human: "User {user_name} logged in",
    #[user_input(user_name)]  // Explicit marking
)]
// Generated code:
let user_name_escaped = if user_name.contains('{') || user_name.contains('}') {
    Cow::Owned(user_name.replace('{', "\\{").replace('}', "\\}"))
} else {
    Cow::Borrowed(user_name)  // Zero-copy if no braces
};
```
**Performance Target**: <100ns for template interpolation with ≤3 variables, ZERO heap allocations
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
**⏱️ PERFORMANCE TEAM COMMENT**:
- ✅ **APPROVED DESIGN**: Conditional compilation with no-op macros is the correct approach for zero-overhead production builds.
- 🚨 **VERIFICATION CRITICAL**: Must verify via `cargo expand` that production builds contain ZERO trace code (not even empty function calls). Any residual code is unacceptable.
- ⚡ **DEV BUILD OPTIMIZATION**: Even 2% overhead compounds in tight loops. Consider `#[cold]` attribute on trace paths to hint branch predictor, keeping hot path optimized.
- 📊 **ACCEPTANCE CRITERIA**: Production binary size must be identical with/without trace macros (byte-for-byte, verified via `sha256sum`).
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
**⏱️ PERFORMANCE TEAM COMMENT**:
- 🚨 **REGEX PERFORMANCE RISK**: Multiple regex passes (6+ patterns) on every narration string is expensive. Each `replace_all()` scans the entire string. **RECOMMENDATION**: Combine patterns into single regex with alternation `(pattern1|pattern2|...)` for one-pass scanning.
- ⚡ **OPTIMIZATION**: Use `aho-corasick` crate for multi-pattern matching (10-100x faster than multiple regex for literal prefixes like "Bearer ", "-----BEGIN").
- 📊 **BENCHMARK REQUIREMENT**: Measure redaction overhead on 100-char, 1000-char, and 10000-char strings. Target: <10μs for 1000-char string with 6 patterns.
- 🔥 **HOT PATH IMPACT**: If redaction runs on EVERY narration (even INFO level), this becomes a critical bottleneck. Consider lazy redaction only for ERROR/FATAL levels, or sampling.
**🎭 AUTH-MIN SECURITY FINDINGS**:
- 🚨 **ReDoS vulnerability in private key regex**: Pattern `-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----` uses lazy quantifier `[\s\S]+?` which can cause catastrophic backtracking on malicious input (e.g., missing END marker). **MITIGATION REQUIRED**: Replace with bounded quantifier or line-by-line parsing.
- ✅ JWT pattern is safe (no backtracking)
- ✅ URL password pattern is safe (negated character classes)
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
**🎭 AUTH-MIN SECURITY NOTE**:
- ⚠️ **Correlation ID injection**: If correlation IDs come from user input (HTTP headers), they could inject malicious data into tracing spans. **MITIGATION REQUIRED**: Validate correlation ID format (UUID v4 only) and reject malformed inputs.
- ⚠️ **Correlation ID forgery**: No authentication/signing of correlation IDs allows attackers to poison distributed traces. Consider HMAC-signed correlation IDs for security-critical flows.
**⏱️ PERFORMANCE TEAM COMMENT**:
- ⚡ **UUID GENERATION COST**: `uuid::Uuid::new_v4()` uses cryptographic RNG (~500ns). For high-frequency correlation ID generation, this is expensive. **RECOMMENDATION**: Use `uuid::Uuid::now_v7()` (timestamp-based, ~50ns) or pre-generate pool of UUIDs.
- 🚨 **VALIDATION OVERHEAD**: Regex validation of UUID format on every header extraction adds latency. **OPTIMIZATION**: Use byte-level validation (check hyphens at positions 8,13,18,23, hex chars elsewhere) for 10x speedup.
- 📊 **TARGET**: <100ns for correlation ID extraction and validation from HTTP headers.
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
**🎭 AUTH-MIN SECURITY NOTE**:
- ⚠️ **Redaction in async contexts**: Async examples (lines 599-623) show direct narration without explicit redaction calls. Developers might forget to redact secrets in async code. **MITIGATION REQUIRED**: Add explicit redaction examples in async patterns.
- ⚠️ **Span injection via user input**: `#[instrument(fields(correlation_id = %correlation_id))]` could inject malicious data if correlation_id is user-controlled. Validate before adding to spans.
**⏱️ PERFORMANCE TEAM COMMENT**:
- ✅ **GOOD**: Using `tracing::event!()` which is lock-free and non-blocking is the correct design.
- 🚨 **ASYNC OVERHEAD CONCERN**: `tracing::Span::current()` and `span.enter()` have overhead (~100-200ns). If called in tight async loops, this compounds. **RECOMMENDATION**: Hoist span entry outside loops, reuse entered span guard.
- ⚡ **OPTIMIZATION**: For fire-and-forget narration, avoid `_enter` guard entirely - just emit events, let subscriber handle span context.
- 📊 **BENCHMARK REQUIREMENT**: Measure narration overhead in async context with 1, 10, 100 concurrent tasks. Verify no executor starvation.
---
### **Unit 2.6: 🚨 CRITICAL - Non-Blocking Narration & Async Support** (Day 10)
**Owner**: Narration Core Team  
**Effort**: 8 hours
**Why Critical**: Narration MUST NOT block async functions! All narration must be fire-and-forget.
**Key Design Principle**: 
- ✅ `narrate()` is **synchronous** but **non-blocking** (uses `tracing` which is already non-blocking)
- ✅ Context propagation uses `tracing::Span` (already async-safe)
- ✅ No `.await` in narration calls
- ✅ No locks, no blocking I/O, no allocations in hot path
**Tasks**:
- [ ] Verify `narrate()` is non-blocking (uses `tracing::event!` which is lock-free)
- [ ] Use `tracing::Span::current()` for async context propagation (already works!)
- [ ] Add `#[instrument]` integration for automatic span propagation
- [ ] Ensure correlation IDs propagate via `tracing` spans
- [ ] Write async compatibility tests
- [ ] Document async patterns
**Concrete Code Changes**:
```rust
// MODIFY: src/lib.rs (ensure non-blocking)
/// Emit narration at a specific level (NON-BLOCKING)
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    let Some(tracing_level) = level.to_tracing_level() else {
        return; // MUTE - no output
    };
    // Apply redaction (pure function, no I/O)
    let human = redact_secrets(&fields.human, RedactionPolicy::default());
    let cute = fields.cute.as_ref().map(|c| redact_secrets(c, RedactionPolicy::default()));
    // Emit via tracing (lock-free, non-blocking)
    // tracing::event! is designed to be non-blocking
    match tracing_level {
        Level::INFO => event!(Level::INFO, 
            actor = fields.actor,
            action = fields.action,
            target = %fields.target,
            human = %human,
            cute = cute.as_deref(),
            correlation_id = fields.correlation_id.as_deref(),
        ),
        // ... other levels
    }
    // Test capture is also non-blocking (just pushes to Vec)
    #[cfg(any(test, feature = "test-support"))]
    capture::notify(fields);
}
// NEW: Async-friendly narration with automatic span propagation
pub fn narrate_in_span(fields: NarrationFields) {
    // Get current span (async-safe, no blocking)
    let span = tracing::Span::current();
    // Narrate within span context (correlation ID auto-propagated)
    let _enter = span.enter();
    narrate(fields);
}
// NEW: Integration with #[instrument] for automatic tracing
#[macro_export]
macro_rules! narrate_async {
    ($($field:ident: $value:expr),* $(,)?) => {{
        // This works in async functions because narrate() is non-blocking
        $crate::narrate($crate::NarrationFields {
            $($field: $value,)*
            ..Default::default()
        })
    }};
}
```
**Example Usage in Async Code**:
```rust
use tracing::instrument;
#[instrument(fields(correlation_id = %correlation_id))]
async fn dispatch_job(job_id: &str, correlation_id: &str) -> Result<WorkerId> {
    // Narration is non-blocking, works in async!
    narrate!(
        actor: "orchestratord",
        action: "dispatch",
        target: job_id,
        human: "Dispatching job",
    );
    let worker = select_worker().await?;
    // Another non-blocking narration
    narrate!(
        actor: "orchestratord",
        action: "dispatch",
        target: job_id,
        human: format!("Dispatched to worker {}", worker.id),
    );
    Ok(worker.id)
}
```
**Why This Design Works**:
- ✅ `tracing::event!()` is **lock-free** and **non-blocking** by design
- ✅ `tracing::Span` automatically propagates across `.await` points
- ✅ Correlation IDs are in span fields, not thread-local storage
- ✅ No blocking I/O, no mutexes, no allocations in hot path
- ✅ Works seamlessly with `tokio`, `async-std`, any async runtime
**Why This Matters**:
- 🔥 Blocking in async code causes **executor starvation**
- 🔥 Thread-local storage **doesn't work** across `.await` points
- 🔥 `tracing` already solved this - we just use it correctly!
**Deliverables**:
- Non-blocking verification tests
- Async compatibility tests
- Documentation with async examples
**Dependencies**: None (uses existing `tracing` infrastructure)
---
### **Unit 2.7: 🚫 REMOVED - Sampling & Rate Limiting**
**⏱️ PERFORMANCE TEAM DECISION**: 🔴 **REJECTED - DO NOT IMPLEMENT**
**🎭 SECURITY TEAM FINDINGS**: 5 CRITICAL vulnerabilities
- CRIT-2: Mutex poisoning DoS
- CRIT-3: HashMap collision DoS  
- CRIT-4: Unbounded memory growth
- CRIT-5: Global mutex contention
**⏱️ PERFORMANCE TEAM FINDINGS**: PERFORMANCE DISASTER
- Global mutex on EVERY narration call
- +200-400ns overhead + mutex contention
- Allocation storm (10k allocations/sec at high frequency)
- Throughput collapse from 100K/sec to 1K/sec
**ALTERNATIVE SOLUTIONS** (Performance Team Approved):
1. ✅ Use `RUST_LOG=info` environment variable (zero runtime overhead)
2. ✅ Use `tracing-subscriber::EnvFilter` (outside hot path)
3. ✅ If custom sampling required: Lock-free DashMap + AtomicU32 + LRU eviction
**VERDICT**: Unit 2.7 is REMOVED from implementation plan
**🚫 THIS UNIT HAS BEEN REMOVED**
Do NOT implement sampling as originally designed. Use alternatives:
- `RUST_LOG` for filtering (zero overhead)
- `tracing-subscriber` for advanced filtering (outside hot path)
**Concrete Code Changes**:
```rust
// NEW FILE: src/sampling.rs
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
/// Sampling configuration
#[derive(Clone, Debug)]
pub struct SamplingConfig {
    /// Sample rate (0.0 = none, 1.0 = all)
    pub sample_rate: f64,
    /// Max events per second per actor/action
    pub max_per_second: Option<u32>,
}
impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            sample_rate: 1.0,  // 100% by default
            max_per_second: None,
        }
    }
}
/// Sampler for rate limiting narration events
pub struct Sampler {
    config: SamplingConfig,
    counters: Arc<Mutex<HashMap<String, (Instant, u32)>>>,
}
impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            config,
            counters: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    /// Check if event should be sampled
    pub fn should_sample(&self, actor: &str, action: &str) -> bool {
        // Probabilistic sampling
        if rand::random::<f64>() > self.config.sample_rate {
            return false;
        }
        // Rate limiting
        if let Some(max_per_sec) = self.config.max_per_second {
            let key = format!("{}:{}", actor, action);
            let mut counters = self.counters.lock().unwrap();
            let now = Instant::now();
            let (last_reset, count) = counters
                .entry(key.clone())
                .or_insert((now, 0));
            if now.duration_since(*last_reset) > Duration::from_secs(1) {
                *last_reset = now;
                *count = 0;
            }
            if *count >= max_per_sec {
                return false;  // Rate limit exceeded
            }
            *count += 1;
        }
        true
    }
}
// MODIFY: src/lib.rs (add sampling check)
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    // Check sampling
    if !GLOBAL_SAMPLER.should_sample(fields.actor, fields.action) {
        return;  // Skip this event
    }
    // ... rest of narrate_at_level
}
```
**Why This Matters**:
- 🔥 Prevents log flooding in hot paths
- 🔥 Reduces storage costs in production
- 🔥 Maintains observability while controlling volume
**Deliverables**:
- `src/sampling.rs`
- Sampling tests
- Configuration documentation
**Dependencies**: None
**🎭 AUTH-MIN SECURITY FINDINGS**:
- 🚨 **Mutex poisoning DoS**: Line 711 uses `.unwrap()` on mutex lock, causing panic if poisoned. **ATTACK VECTOR**: Attacker triggers panic in one thread → poisons mutex → DoS all narration. **MITIGATION REQUIRED**: Use `lock().ok()` or `try_lock()` with graceful degradation.
- 🚨 **HashMap collision DoS**: Line 710 uses `format!("{}:{}", actor, action)` as key. If actor/action are user-controlled, attacker can craft collision-inducing keys to degrade HashMap to O(n). **MITIGATION REQUIRED**: Sanitize actor/action before using as keys.
- 🚨 **Unbounded memory growth**: No eviction policy for counters HashMap (line 714). **ATTACK VECTOR**: Memory exhaustion via unique actor:action combinations. **MITIGATION REQUIRED**: Add LRU eviction or TTL-based cleanup.
**⏱️ PERFORMANCE TEAM COMMENT**:
- 🚨 **CRITICAL BOTTLENECK**: Global mutex on EVERY narration call is a massive contention point. With 1000+ narrations/sec across threads, this serializes all logging. **RECOMMENDATION**: Use lock-free atomic counters per actor:action (e.g., `DashMap` or sharded counters).
- 🚨 **ALLOCATION STORM**: `format!("{}:{}", actor, action)` allocates String on every `should_sample()` call. At 10k narrations/sec, this is 10k allocations/sec. **OPTIMIZATION**: Use `(&str, &str)` tuple as key directly, or pre-intern actor:action pairs.
- 🚨 **RANDOM NUMBER OVERHEAD**: `rand::random::<f64>()` on every call adds ~50-100ns. For 100% sample rate (default), this is pure waste. **OPTIMIZATION**: Early-return if `sample_rate == 1.0`, skip RNG entirely.
- 📊 **PERFORMANCE DISASTER**: This design will destroy throughput. **REQUIRED**: Complete redesign using lock-free data structures. Target: <50ns overhead for sampling check.
---
### **Unit 2.8: 💝 Unicode Safety (SIMPLIFIED)** (Day 10)
**Owner**: Narration Core Team  
**Effort**: 3 hours (INCREASED - add CRLF + correlation ID validation)
**⏱️ PERFORMANCE TEAM DECISION**: Simplified validation (NOT comprehensive)
**Tasks**:
- [ ] Implement ASCII fast path (zero-copy for 90% of strings)
- [ ] Simplified UTF-8 validation (`c.is_control()` + 5 zero-width chars)
- [ ] Homograph detection ONLY for actor/action fields (reject non-ASCII)
- [ ] CRLF sanitization (strip `\n`, `\r`, `\t` only)
- [ ] Correlation ID validation (byte-level UUID v4 checks)
- [ ] Test with various emoji (🎀, 🎭, 🚀, etc.)
- [ ] Document validation rules
**🚫 REJECTED by Performance Team**:
- Comprehensive emoji ranges (20+ ranges) - too complex
- Encode all control chars - allocates on every narration
- Unicode normalization (NFC) for human/cute/story - too expensive
- HMAC-signed correlation IDs - 500-1000ns overhead
**Concrete Code Changes** (⏱️ Performance Team Approved):
```rust
// MODIFY: src/lib.rs (simplified validation)
/// ASCII fast path (zero-copy for 90% of cases)
pub fn sanitize_for_json(text: &str) -> Cow<'_, str> {
    if text.is_ascii() {
        return Cow::Borrowed(text);  // Zero-copy, no validation
    }
    // Simplified UTF-8 validation (not comprehensive)
    Cow::Owned(
        text.chars()
            .filter(|c| {
                !c.is_control() &&  // Basic control char filter
                !matches!(*c as u32, 
                    0x200B..=0x200D |  // Zero-width space, ZWNJ, ZWJ
                    0xFEFF |           // Zero-width no-break space
                    0x2060             // Word joiner
                )
            })
            .collect()
    )
}
/// CRLF sanitization (strip only \n, \r, \t)
pub fn sanitize_crlf(text: &str) -> Cow<'_, str> {
    if !text.contains(|c: char| matches!(c, '\n' | '\r' | '\t')) {
        return Cow::Borrowed(text);  // Zero-copy (90% of cases)
    }
    Cow::Owned(
        text.replace('\n', " ")  // Strip, not encode (faster)
            .replace('\r', " ")
            .replace('\t', " ")
    )
}
/// Correlation ID validation (byte-level, <100ns)
pub fn validate_correlation_id(id: &str) -> Option<&str> {
    if id.len() != 36 { return None; }
    let bytes = id.as_bytes();
    if bytes[8] != b'-' || bytes[13] != b'-' || 
       bytes[18] != b'-' || bytes[23] != b'-' { return None; }
    for (i, &b) in bytes.iter().enumerate() {
        if i == 8 || i == 13 || i == 18 || i == 23 { continue; }
        if !b.is_ascii_hexdigit() { return None; }
    }
    Some(id)  // Return borrowed (zero-copy)
}
/// Actor validation (reject non-ASCII to prevent homograph attacks)
pub fn validate_actor(actor: &str) -> Result<&str, Error> {
    if !actor.is_ascii() {
        return Err(Error::NonAsciiActor);
    }
    Ok(actor)
}
```
**Why This Matters**:
- 🎀 Cute mode is our brand - emoji must work!
- 🔥 Invalid Unicode breaks JSON parsers
- 🔥 Zero-width chars can hide malicious content
**Deliverables**:
- `src/unicode.rs`
- Emoji safety tests
- Emoji guidelines
**Dependencies**: None
**🎭 AUTH-MIN SECURITY FINDINGS**:
- 🚨 **Incomplete emoji range**: Line 783 checks `(*c as u32) >= 0x1F000` but emoji ranges are fragmented (0x1F600-0x1F64F, 0x1F300-0x1F5FF, etc.). **ATTACK VECTOR**: Inject malicious Unicode outside defined range. **MITIGATION REQUIRED**: Use comprehensive emoji range list or Unicode category checks.
- 🚨 **Incomplete zero-width blocklist**: Line 793 missing U+FEFF (zero-width no-break space), U+2060 (word joiner), U+180E (Mongolian vowel separator). **ATTACK VECTOR**: Hide malicious content in logs. **MITIGATION REQUIRED**: Comprehensive zero-width character blocklist.
- 🚨 **Homograph attack vulnerability**: No detection of Cyrillic/Greek lookalikes (е vs e, а vs a). **ATTACK VECTOR**: Spoof actor names in logs ("оrchestratord" vs "orchestratord"). **MITIGATION REQUIRED**: Consider homograph detection for security-critical fields (actor, action).
- ⚠️ **No Unicode normalization**: Missing NFC/NFD normalization before validation allows normalization-based bypasses. **MITIGATION REQUIRED**: Apply Unicode normalization before all validation.
**⏱️ PERFORMANCE TEAM COMMENT**:
- 🚨 **CHAR ITERATION COST**: `.chars()` iterator allocates for multi-byte UTF-8 sequences. For 1000-char strings, this is expensive. **OPTIMIZATION**: Use `.as_bytes()` for ASCII-only validation, fall back to `.chars()` only for non-ASCII.
- 🚨 **FILTER ALLOCATION**: `.filter().collect()` allocates new String on every narration. **RECOMMENDATION**: In-place sanitization using `String::retain()` or pre-validate and reject (zero-copy).
- ⚡ **UNICODE NORMALIZATION COST**: NFC/NFD normalization (if added per auth-min) is expensive (~1-10μs per string). **OPTIMIZATION**: Only normalize security-critical fields (actor, action), skip for human/cute/story.
- 📊 **TARGET**: <1μs for emoji validation on 100-char string, <5μs for full sanitization with normalization.
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
**🎭 AUTH-MIN SECURITY NOTE**:
- ⚠️ **ReDoS risk in passive voice detection**: If using complex regex patterns for SVO validation, could be vulnerable to ReDoS. **MITIGATION REQUIRED**: Review actual regex patterns when implemented; use bounded quantifiers and avoid nested quantifiers.
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
**⏱️ PERFORMANCE TEAM COMMENT**:
- ✅ **NO PERFORMANCE CONCERNS**: BDD tests run in test environment only, performance is not critical.
- 📊 **RECOMMENDATION**: Ensure `CaptureAdapter` uses efficient data structures (Vec with pre-allocated capacity) to avoid reallocation during test runs.
---
### **Unit 4.2: Proof Bundle Integration** (Day 17-18)
**Owner**: Narration Core Team  
**Effort**: 8 hours
**Tasks**:
- [ ] Integrate `CaptureAdapter` with 
- [ ] Auto-capture narration events in test runs
- [ ] Store narration in  artifacts
- [ ] Add assertions for narration presence
- [ ] Test with `LLORCH_PROOF_DIR` environment variable
- [ ] Document  integration
- [ ] Update  spec
**Deliverables**:
- Updated `src/capture.rs` with  integration
- Integration tests
- Documentation
**Dependencies**: Unit 4.1
**🎭 AUTH-MIN SECURITY NOTE**:
- ⚠️ ** data leakage**:  may contain sensitive narration data including redacted secrets. Ensure `LLORCH_PROOF_DIR` respects file permissions (0600) and doesn't leak to world-readable directories. Add security notice in documentation.
**⏱️ PERFORMANCE TEAM COMMENT**:
- 🚨 **I/O BLOCKING RISK**: Writing to  during test runs must NOT block narration in async contexts. **REQUIREMENT**: Use buffered writes or async I/O to prevent blocking.
- ⚡ **OPTIMIZATION**: Batch narration events in memory, flush to disk at test completion to minimize I/O syscalls.
- 📊 **TARGET**:  writes should add <1% overhead to test execution time.
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
- [ ] Add  artifact collection
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
- ✅  integration working
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
- 📊 Integrates with our 
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
### 🚨 CRITICAL SECURITY ISSUES IDENTIFIED
**The following attack surfaces were identified during security review and MUST be addressed before implementation:**
1. **ReDoS in private key regex** (Unit 2.3) - CRITICAL
2. **Mutex poisoning DoS** (Unit 2.7) - CRITICAL  
3. **HashMap collision DoS** (Unit 2.7) - CRITICAL
4. **Unbounded memory growth** (Unit 2.7) - CRITICAL
5. **Incomplete Unicode validation** (Unit 2.8) - HIGH
6. **Homograph attacks** (Unit 2.8) - HIGH
7. **Template injection** (Unit 1.5) - MEDIUM
8. **Correlation ID injection** (Unit 2.4) - MEDIUM
9. **Log injection (CRLF)** - MEDIUM (not addressed in plan)
10. **Timing side-channels** (Unit 1.3) - LOW
**REQUIRED ACTIONS**:
- [ ] Address all CRITICAL issues before Week 2
- [ ] Address all HIGH issues before Week 3  
- [ ] Address all MEDIUM issues before Week 4
- [ ] Add log injection prevention to implementation plan
- [ ] Add timing side-channel mitigation guidelines
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
---
## 📊 Final Implementation Summary
### **Timeline Revised**: 4 weeks → 2.5 weeks (15 days)
**Savings**: ~60 hours (37.5% reduction) due to:
- ✅ Existing trace macros, redaction, capture adapter
- 🚫 Unit 2.7 removed (Performance Team rejection)
- ⚡ Simplified validation (Performance Team overrides)
### **Performance Targets** (MANDATORY - Blocking for Merge)
| Component | Target | Verification |
|-----------|--------|--------------|
| **Template Interpolation** | <100ns | `cargo bench --bench template_interpolation` |
| **Redaction (clean)** | <1μs | `cargo bench --bench redaction_clean_1000_chars` |
| **Redaction (with secrets)** | <5μs | `cargo bench --bench redaction_with_secrets` |
| **CRLF Sanitization** | <50ns | `cargo bench --bench crlf_sanitization_clean` |
| **Unicode (ASCII)** | <1μs | `cargo bench --bench unicode_validation_ascii` |
| **Correlation ID** | <100ns | `cargo bench --bench uuid_validation` |
| **Production Build** | 0ns | `cargo expand --release` (verify ZERO trace code) |
### **Performance Team Vetoes Applied**
1. ✅ **Tracing Opt-In** - Zero overhead in production (MANDATORY)
2. ✅ **Compile-Time Templates** - Stack buffers, no runtime parsing
3. ✅ **Single-Pass Redaction** - Cow strings, <5μs target
4. ✅ **Simplified Unicode** - ASCII fast path, not comprehensive
5. ✅ **Strip CRLF Only** - `\n`, `\r`, `\t` only (not all control chars)
6. ✅ **Byte-Level UUID** - No HMAC signing (<100ns)
7. 🚫 **Sampling REJECTED** - Use `RUST_LOG` instead
### **Accepted Security Risks** (Performance Team Decision)
- ⚠️ Template injection: Only user-marked inputs escaped
- ⚠️ Log injection: Only `\n`, `\r`, `\t` stripped (not all control chars)
- ⚠️ Unicode: Simplified validation (not comprehensive emoji ranges)
- ⚠️ Correlation ID: No HMAC signing (forgery risk accepted)
- ⚠️ Timing: Dev build timing data is acceptable (not a security issue)
### **CI/CD Requirements**
```yaml
# .github/workflows/narration-performance.yml
- name: Run performance benchmarks
  run: cargo bench --bench narration_performance
- name: Verify performance targets
  run: |
    cargo bench -- --save-baseline main
    cargo bench -- --baseline main --fail-fast
- name: Verify zero-overhead production
  run: |
    cargo expand --release --features="" | grep -q "Instant::now" && exit 1 || exit 0
    cargo expand --release --features="" | grep -q "trace_tiny" && exit 1 || exit 0
```
---
**Implementation Plan Complete** ✅  
**Verified Against Existing Code** ✅  
**Performance Team Approved** ⏱️  
**Ready for Execution** 🚀  
**Cuteness Pays the Bills** 💝
