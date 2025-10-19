# 🔍 Existing Rust Tracing Solutions
**Question**: Is there already a Rust library that does automatic function tracing with attributes?
**Answer**: **YES!** Several exist. Here's what's available and how they compare to our needs.
---
## 📚 Existing Libraries
### 1. **`tracing-attributes`** (Part of `tracing` ecosystem)
**What it does**: Provides `#[instrument]` macro for automatic function tracing.
```rust
use tracing::instrument;
#[instrument]
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```
**Pros**:
- ✅ Official `tracing` ecosystem
- ✅ Well-maintained
- ✅ Automatic span creation
- ✅ Captures function arguments
- ✅ Works with `?` operator
- ✅ Conditional compilation support
**Cons**:
- ❌ No built-in timing (need to add manually)
- ❌ No automatic result/error logging
- ❌ No cute mode 🎀
- ❌ No narration-specific fields (actor, action, target)
**Verdict**: ✅ **USE THIS!** It's exactly what we need for the base functionality.
---
### 2. **`tracing-timing`**
**What it does**: Adds automatic timing to `tracing` spans.
```rust
use tracing_timing::{Builder, Histogram};
#[instrument]
fn dispatch_job(job_id: &str) -> Result<WorkerId> {
    // Automatically measures duration
}
```
**Pros**:
- ✅ Automatic timing
- ✅ Histogram support
- ✅ Works with `#[instrument]`
**Cons**:
- ❌ Requires setup
- ❌ Not as simple as we want
**Verdict**: ⚠️ **OPTIONAL** — Nice to have but not essential.
---
### 3. **`tracing-subscriber`**
**What it does**: Formats and outputs tracing events.
```rust
use tracing_subscriber::{fmt, EnvFilter};
tracing_subscriber::fmt()
    .with_env_filter(EnvFilter::from_default_env())
    .init();
```
**Pros**:
- ✅ Handles RUST_LOG filtering
- ✅ JSON output support
- ✅ Multiple output formats
**Cons**:
- ❌ Just formatting, not instrumentation
**Verdict**: ✅ **USE THIS!** We need it for output formatting.
---
### 4. **`auto-impl`** / **`async-trait`** (Not tracing-specific)
**What they do**: Procedural macros for code generation.
**Verdict**: ❌ **NOT RELEVANT** — These are for different use cases.
---
## 🎯 Recommended Approach: Use `tracing` Ecosystem
### Instead of Building Our Own:
**Use `#[instrument]` from `tracing-attributes`**:
```rust
use tracing::instrument;
#[instrument(
    name = "dispatch_job",
    fields(
        actor = "queen-rbee",
        action = "dispatch",
        job_id = %job_id,
        pool_id = %pool_id
    ),
    level = "trace"
)]
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```
**Output**:
```
TRACE queen-rbee dispatch_job actor="queen-rbee" action="dispatch" job_id="job-123" pool_id="default"
```
---
## 🔧 How to Adapt `#[instrument]` for Our Needs
### Option 1: Wrapper Macro (RECOMMENDED)
Create a thin wrapper around `#[instrument]`:
```rust
// bin/shared-crates/narration-core/src/lib.rs
/// Our wrapper around #[instrument] with narration defaults
#[macro_export]
macro_rules! trace_fn {
    ($actor:expr, $action:expr) => {
        tracing::instrument(
            fields(
                actor = $actor,
                action = $action,
            ),
            level = "trace"
        )
    };
}
// Usage:
#[trace_fn!("queen-rbee", "dispatch")]
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```
**Pros**:
- ✅ Leverages existing, well-tested library
- ✅ No need to maintain proc macro code
- ✅ Just adds our narration conventions on top
**Cons**:
- ⚠️ Slightly more verbose (need to specify actor/action)
---
### Option 2: Custom Proc Macro (What We Designed)
Build our own `#[trace_fn]` that wraps `#[instrument]`:
```rust
// bin/shared-crates/narration-macros/src/lib.rs
#[proc_macro_attribute]
pub fn trace_fn(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    // Infer actor from module path
    let actor = infer_actor_from_module();
    // Generate #[instrument] with our fields
    let expanded = quote! {
        #[tracing::instrument(
            fields(
                actor = #actor,
                action = stringify!(#fn_name),
            ),
            level = "trace"
        )]
        #input
    };
    TokenStream::from(expanded)
}
```
**Pros**:
- ✅ Zero boilerplate for developers
- ✅ Auto-infers actor from module path
- ✅ Custom to our needs
**Cons**:
- ❌ More code to maintain
- ❌ Reinventing the wheel slightly
---
## 🎯 RECOMMENDATION: Hybrid Approach
### Use `tracing` + Thin Wrapper
**Step 1**: Use `#[instrument]` directly for most cases:
```rust
use tracing::instrument;
#[instrument(
    fields(actor = "queen-rbee", action = "dispatch"),
    level = "trace"
)]
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```
**Step 2**: Add helper macros for common patterns:
```rust
// For TRACE-level with actor/action
#[macro_export]
macro_rules! trace_fn {
    ($actor:expr, $action:expr) => {
        tracing::instrument(
            fields(actor = $actor, action = $action),
            level = "trace"
        )
    };
}
// Usage:
#[trace_fn!("queen-rbee", "dispatch")]
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    // ...
}
```
**Step 3**: Use our custom `narrate()` for INFO/WARN/ERROR:
```rust
// For user-facing events with cute mode
narrate(NarrationFields {
    actor: "queen-rbee",
    action: "accept",
    target: job_id.to_string(),
    human: "Accepted request; queued at position 3".to_string(),
    cute: Some("Orchestratord welcomes job-123! 🎫".to_string()),
    ..Default::default()
});
```
---
## 📊 Comparison: Build vs Buy
| Aspect | Use `tracing` | Build Custom |
|--------|---------------|--------------|
| **Development time** | ✅ 1 day | ❌ 1-2 weeks |
| **Maintenance** | ✅ Community | ❌ Us |
| **Features** | ✅ Well-tested | ⚠️ Need to build |
| **Flexibility** | ⚠️ Some limits | ✅ Full control |
| **Cute mode** | ❌ Need to add | ✅ Built-in |
| **Narration fields** | ⚠️ Manual | ✅ Automatic |
---
## 🎯 Final Recommendation (Updated: Time is Not an Issue!)
### **Build Our Own — Cuteness Pays the Bills! 🎀**
Since we have all the time in the world and our cuteness is our competitive advantage, let's build a **first-class narration system** that's uniquely ours!
**Why build our own**:
1. 💝 **Cute mode is our differentiator** — We need it everywhere, not bolted on
2. 🎭 **Story mode is unique** — No other library has dialogue-based narration
3. 🎨 **Editorial control** — We enforce ≤100 char limits, SVO structure, etc.
4. 🔒 **Secret redaction** — Built-in, not an afterthought
5. 🎀 **Narration-first design** — actor/action/target/human are first-class, not metadata
6. 📊 **** — Integration with our existing test infrastructure
7. 🚀 **Zero compromises** — Exactly what we want, no workarounds
**What to build from scratch**:
1. ✅ Custom `#[trace_fn]` proc macro with auto-inferred actor
2. ✅ Custom `#[narrate(...)]` with template interpolation
3. ✅ Full `narrate()` with cute/story modes built-in
4. ✅ Automatic secret redaction (regex-based, cached)
5. ✅ Correlation ID propagation (automatic from context)
6. ✅ Editorial enforcement (length limits, SVO validation)
7. ✅ Conditional compilation (zero overhead in production)
8. ✅  integration (test capture adapter)
**What to use from `tracing`**:
1. ✅ `tracing` as the **backend** for event emission
2. ✅ `tracing-subscriber` for output formatting
3. ✅ `EnvFilter` for RUST_LOG support
4. ❌ **NOT** `#[instrument]` — We build our own with better ergonomics
**Why this is better**:
- 🎀 **Cute mode everywhere** — Not an add-on, it's core
- 🎭 **Story mode** — Unique to us, perfect for distributed systems
- 🎨 **Editorial standards** — Enforced at compile time
- 🔒 **Security** — Redaction is automatic, not optional
- 📊 **Testing** — CaptureAdapter integrates with our BDD suite
- 🚀 **Performance** — Optimized for our use case (lightweight trace macros)
- 💝 **Brand** — This is uniquely "us", not generic tracing
---
## 🔧 Updated Implementation Plan (Build Our Own!)
### Phase 1: Core Proc Macro Crate (Week 1)
**Create `observability-narration-macros`**:
```toml
# bin/shared-crates/narration-macros/Cargo.toml
[package]
name = "observability-narration-macros"
version = "0.1.0"
edition = "2021"
[lib]
proc-macro = true
[dependencies]
syn = { version = "2.0", features = ["full", "extra-traits"] }
quote = "1.0"
proc-macro2 = "1.0"
```
**Implement `#[trace_fn]`**:
```rust
// Auto-infers actor from module path
// Auto-generates entry/exit tracing
// Handles timing, errors, Result types
// Conditional compilation support
```
**Implement `#[narrate(...)]`**:
```rust
// Template interpolation for human/cute/story
// Auto-extracts variables from context
// Enforces editorial standards at compile time
```
---
### Phase 2: Narration Core (Week 2)
**Enhance `narration-core`**:
```rust
// Full NarrationFields with cute/story modes
pub struct NarrationFields {
    pub actor: &'static str,
    pub action: &'static str,
    pub target: String,
    pub human: String,
    pub cute: Option<String>,      // 🎀 Built-in!
    pub story: Option<String>,     // 🎭 Built-in!
    // ... all other fields
}
// Narration with automatic redaction
pub fn narrate(fields: NarrationFields) {
    // Apply editorial standards
    enforce_length_limit(&fields.human, 100);  // Compile-time check!
    // Apply redaction
    let human = redact_secrets(&fields.human);
    let cute = fields.cute.map(|c| redact_secrets(&c));
    let story = fields.story.map(|s| redact_secrets(&s));
    // Emit via tracing backend
    tracing::info!(
        actor = fields.actor,
        action = fields.action,
        target = %fields.target,
        human = %human,
        cute = cute.as_deref(),
        story = story.as_deref(),
        correlation_id = fields.correlation_id.as_deref(),
        // ... all fields
    );
}
```
---
### Phase 3: Lightweight Trace Macros (Week 2)
**Optimize for hot paths**:
```rust
// Ultra-lightweight (no struct allocation)
#[macro_export]
macro_rules! trace_tiny {
    ($actor:expr, $action:expr, $target:expr, $human:expr) => {
        #[cfg(feature = "trace-enabled")]
        tracing::trace!(
            actor = $actor,
            action = $action,
            target = $target,
            human = $human,
        );
    };
}
// With correlation ID
#[macro_export]
macro_rules! trace_with_correlation {
    // ... implementation
}
// Function boundaries
#[macro_export]
macro_rules! trace_enter { /* ... */ }
#[macro_export]
macro_rules! trace_exit { /* ... */ }
// Loop iterations
#[macro_export]
macro_rules! trace_loop { /* ... */ }
// State transitions
#[macro_export]
macro_rules! trace_state { /* ... */ }
```
---
### Phase 4: Editorial Enforcement (Week 3)
**Compile-time validation**:
```rust
// Proc macro validates at compile time
#[narrate(
    actor = "queen-rbee",
    action = "accept",
    human = "This message is way too long and will fail to compile because it exceeds 100 characters",
    cute = "..."
)]
fn accept_job() { }
// Compile error:
// error: human field exceeds 100 character limit (ORCH-3305)
//   --> src/main.rs:5:5
//    |
// 5  |     human = "This message is way too long..."
//    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```
**SVO structure validation**:
```rust
// Validates Subject-Verb-Object structure
#[narrate(
    human = "request accepted",  // ❌ Passive voice!
    cute = "..."
)]
fn accept_job() { }
// Compile error:
// error: human field should use active voice (Subject-Verb-Object)
//   --> src/main.rs:5:5
//    |
// 5  |     human = "request accepted"
//    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^
//    |
//    = help: try "Accepted request" instead
```
---
### Phase 5: Conditional Compilation (Week 3)
**Feature flags**:
```toml
[features]
default = ["trace-enabled", "debug-enabled", "cute-mode"]
trace-enabled = []
debug-enabled = []
cute-mode = []
production = []  # Disables trace, debug, cute
```
**Conditional macros**:
```rust
// Development: Full implementation
#[cfg(feature = "trace-enabled")]
#[proc_macro_attribute]
pub fn trace_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Generate full tracing code
}
// Production: No-op (code removed)
#[cfg(not(feature = "trace-enabled"))]
#[proc_macro_attribute]
pub fn trace_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Return original function unchanged
    item
}
```
---
### Phase 6: Integration & Testing (Week 4)
**BDD tests**:
```rust
// Test cute mode
#[test]
fn test_cute_mode_narration() {
    let capture = CaptureAdapter::install();
    #[narrate(
        actor = "test",
        action = "test",
        human = "Test message",
        cute = "Cute test message! 🎀"
    )]
    fn test_fn() { }
    test_fn();
    capture.assert_cute_present();
    capture.assert_cute_includes("🎀");
}
// Test editorial enforcement
#[test]
fn test_length_limit_enforcement() {
    // This should fail to compile
    #[narrate(
        human = "This is way too long..." // > 100 chars
    )]
    fn test_fn() { }
}
// Test conditional compilation
#[test]
fn test_production_build_removes_trace() {
    // Verify trace code doesn't exist in production binary
    #[cfg(not(feature = "trace-enabled"))]
    {
        // trace_fn should be no-op
        #[trace_fn]
        fn test() { }
        // Verify no tracing symbols in binary
        assert!(!binary_contains_symbol("trace_fn"));
    }
}
```
** integration**:
```rust
// Narration events automatically captured in 
#[test]
fn test_narration_proof_bundle() {
    let bundle = ProofBundle::for_type(TestType::Unit);
    narrate(NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test narration".to_string(),
        cute: Some("Cute test! 🎀".to_string()),
        ..Default::default()
    });
    // Narration automatically captured in bundle
    bundle.assert_contains_narration("Test narration");
    bundle.assert_contains_cute("Cute test! 🎀");
}
```
---
## 🎀 Summary (Updated: Build Our Own!)
### The Answer:
**YES**, `tracing` ecosystem provides the basics, **BUT** we're building our own because:
### Why Build Our Own (Time is Not an Issue):
1. 💝 **Cute mode is our brand** — It needs to be first-class, not bolted on
2. 🎭 **Story mode is unique** — No other library has dialogue-based narration
3. 🎨 **Editorial enforcement** — Compile-time validation of ≤100 chars, SVO structure
4. 🔒 **Security-first** — Automatic redaction, not optional
5. 🎀 **Narration-first design** — actor/action/target/human are core, not metadata
6. 📊 ** integration** — Works with our existing test infrastructure
7. 🚀 **Zero compromises** — Exactly what we want, no workarounds
8. 💝 **Brand differentiation** — This is uniquely "us"
### What We're Building:
1. ✅ **Custom proc macros** (`#[trace_fn]`, `#[narrate(...)]`)
2. ✅ **Auto-inferred actor** from module path
3. ✅ **Template interpolation** for human/cute/story fields
4. ✅ **Compile-time editorial enforcement** (length limits, SVO validation)
5. ✅ **Lightweight trace macros** for hot paths (~10x faster)
6. ✅ **Conditional compilation** (zero overhead in production)
7. ✅ ** integration** (automatic test capture)
8. ✅ **Full cute/story mode support** (built-in, not add-on)
### What We're Using from `tracing`:
1. ✅ `tracing` as the **backend** for event emission
2. ✅ `tracing-subscriber` for output formatting
3. ✅ `EnvFilter` for RUST_LOG support
4. ❌ **NOT** `#[instrument]` — We build our own with better ergonomics
### Timeline:
**Week 1**: Core proc macro crate (`#[trace_fn]`, `#[narrate(...)]`)  
**Week 2**: Narration core enhancements + lightweight trace macros  
**Week 3**: Editorial enforcement + conditional compilation  
**Week 4**: Integration, testing, BDD suite, 
**Total**: 4 weeks for a **world-class narration system** that's uniquely ours! 🎀
---
## 📚 Resources
- [`tracing` docs](https://docs.rs/tracing)
- [`tracing-attributes` docs](https://docs.rs/tracing-attributes)
- [`tracing-subscriber` docs](https://docs.rs/tracing-subscriber)
- [Tokio tracing guide](https://tokio.rs/tokio/topics/tracing)
---
**With love, sass, and the confidence that cuteness pays the bills,**  
**The Narration Core Team** 🎭✨
*P.S. — We're not reinventing the wheel. We're building a **cute, story-telling, editorially-enforced** wheel that's uniquely ours. Because generic tracing is boring, and we're not boring. 💝*
---
*May your proc macros be powerful and your narration be adorable! 🎀*
