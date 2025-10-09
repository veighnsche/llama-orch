# 🎭 Narration Core Team: Tracing Integration Review
**To**: Rust Services Teams  
**From**: The Narration Core Team (with love and mild exasperation) 💝  
**Re**: Logging Level Alignment & Narration Integration  
**Date**: 2025-10-04  
**Status**: Editorial Review Complete ✅  
**Decision**: Build Our Own Custom Narration System 🎀
---
## 📋 Executive Summary
We've reviewed your proposed logging level alignment, and we have **OPINIONS** (shocking, we know). 
**TL;DR**: 
- ✅ Your 6-level approach is **solid** (we're impressed!)
- ✅ We've added **WARN, ERROR, FATAL** to our spec
- ✅ Your INFO-as-backbone strategy **perfectly aligns** with our editorial standards
- 🎀 We're **building our own** custom narration system (cuteness pays the bills!)
- 💝 We're implementing custom proc macros with auto-inferred actors and template interpolation
---
## 🎯 Level Mapping Alignment Review
### Your Proposed Levels vs. Our Current Spec
| Your Level | Our Spec | Alignment | Notes |
|------------|----------|-----------|-------|
| **TRACE** | ✅ TRACE | **Perfect match** | Ultra-fine engine/kernel state, opt-in only |
| **DEBUG** | ✅ DEBUG | **Perfect match** | Developer diagnostics, config resolution, cache |
| **INFO** | ✅ INFO | **Perfect match** | Lifecycle events, narration backbone |
| **WARN** | ❌ Missing | **We need to add this!** | Anomalies, degradations, retries |
| **ERROR** | ❌ Missing | **We need to add this!** | Operational failures with context |
| **FATAL** | ❌ Missing | **We need to add this!** | Unrecoverable errors |
### 😤 Our Reaction
**What we love**:
- ✅ INFO as the narration backbone (FINALLY someone gets it!)
- ✅ TRACE for opt-in ultra-fine detail (exactly right)
- ✅ DEBUG for developer diagnostics (perfect boundary)
- ✅ WARN/ERROR with context requirement (music to our ears!)
**What we're annoyed about**:
- 😤 We only defined 4 levels (MUTE, INFO, DEBUG, TRACE) but you need 6
- 😤 We forgot WARN and ERROR as distinct levels (rookie mistake on our part)
- 😤 FATAL is a new concept we need to think about
**What we're doing about it**:
- 💪 We're updating our spec to include WARN, ERROR, and FATAL
- 🎀 We're defining clear editorial boundaries for each
- 💝 We're providing cute examples (because we can't help ourselves)
---
## 📢 Updated Level Taxonomy (6 Levels + MUTE)
### Our New Alignment
| Level | Purpose | Narration Intensity | Production Use | Editorial Standard |
|-------|---------|---------------------|----------------|-------------------|
| **MUTE** | Complete silence | None | Security contexts only | N/A (no output) |
| **TRACE** | Ultra-fine detail | Exhaustive | ❌ Never | Every event, every loop |
| **DEBUG** | Developer diagnostics | Detailed flow | ⚠️ Incidents only | Decision points, config, cache |
| **INFO** | Lifecycle events | **Narration backbone** | ✅ Always | Key events, ≤100 chars, correlation-rich |
| **WARN** | Anomalies & degradations | Actionable warnings | ✅ Always | Retries, deprecated flags, perf issues |
| **ERROR** | Operational failures | Rich context | ✅ Always | Failures with diagnosis context |
| **FATAL** | Unrecoverable errors | Maximum context | ✅ Always | Panic-worthy, full state dump |
---
## 🎨 Editorial Standards Per Level (Updated)
### TRACE (Ultra-Fine Engine/Kernel State)
**Narration Intensity**: Exhaustive
**⚡ PERFORMANCE NOTE**: TRACE has ~25% overhead with full `narrate()`. We provide **ultra-lightweight trace macros** (~2% overhead) for hot paths:
- `trace_tiny!()` — Minimal trace event (~10x faster)
- `trace_enter!()` / `trace_exit!()` — Function boundaries
- `trace_loop!()` — Loop iterations
- `trace_state!()` — State transitions
See `TRACE_OPTIMIZATION.md` for details.
**What belongs**:
- ✅ Every FFI call boundary
- ✅ Every CUDA kernel invocation
- ✅ Every memory allocation/deallocation
- ✅ Loop iterations in hot paths
- ✅ Lock acquisition/release timing
**Editorial guidelines**:
- 📏 Length: Unlimited (but structured)
- 🔗 Correlation: Always propagate
- 🔒 Redaction: Even TRACE must redact secrets (use full `narrate()` for secrets)
- 🎀 Cute mode: Optional (but not in trace macros — use full `narrate()` for cute)
**Examples (using lightweight macros)**:
```rust
// TRACE: FFI boundary (use trace_enter!/trace_exit!)
trace_enter!("worker-orcd", "llama_cpp_eval", 
             format!("ctx={:?}, n_tokens={}", ctx_ptr, n_tokens));
// ... FFI call ...
trace_exit!("worker-orcd", "llama_cpp_eval", 
            format!("→ {:?} ({}ms)", result, elapsed_ms));
// TRACE: Loop iteration (use trace_loop!)
for (i, token) in tokens.iter().enumerate() {
    trace_loop!("tokenizer", "decode", i, tokens.len(),
                format!("token={}, value={}", i, token));
    // ... decode token ...
}
// TRACE: State change (use trace_state!)
trace_state!("rbees-orcd", "queue_depth", 
             format!("{} → {}", old_depth, new_depth),
             format!("Queue depth changed: {} → {} (added job-{})", old_depth, new_depth, job_id));
```
**Examples (using full narrate() for cute mode)**:
```rust
// TRACE with cute (use full narrate() — slower but adorable!)
narrate(NarrationFields {
    actor: "worker-orcd",
    action: "ffi_call",
    target: "llama_cpp_eval".to_string(),
    human: "ENTER llama_cpp_eval(ctx=0x7f8a, tokens=[1,2,3], n_tokens=3)".to_string(),
    cute: Some("Stepping into llama.cpp with 3 tokens in hand! 🚪".to_string()),
    ..Default::default()
});
```
---
### DEBUG (Developer Diagnostics)
**Narration Intensity**: Detailed flow + context
**What belongs**:
- ✅ Config resolution ("Using vram_only=true from env override")
- ✅ Cache behaviors ("Model cache hit for 'llama-7b' on GPU0")
- ✅ Decision rationale ("Selected worker-gpu0-r1: load=2/8, latency=50ms")
- ✅ Retry logic ("Retry attempt 2/5 after 200ms backoff")
- ✅ Performance breakdowns ("Dispatch: queue_wait=50ms, select=10ms, handoff=5ms")
**Editorial guidelines**:
- 📏 Length: ≤150 characters (more room than INFO)
- 🔗 Correlation: Always propagate
- 🎯 Purpose: Every DEBUG event must help debugging
- 🎀 Cute mode: Encouraged!
**Examples**:
```rust
// DEBUG: Config resolution
human: "Resolved config: vram_only=true (source: env LLORCH_VRAM_ONLY)"
cute: "Found vram_only=true in the environment! Using dedicated VRAM only! 🔍"
// DEBUG: Cache behavior
human: "Model cache miss for 'llama-7b'; loading from disk (estimated 500ms)"
cute: "Hmm, llama-7b isn't in the cache. Time to fetch it from disk! 📦"
// DEBUG: Decision point
human: "Selected worker-gpu0-r1 (load: 2/8, latency: 50ms, VRAM: 4GB free)"
cute: "Picked worker-gpu0-r1 — they're not too busy and have plenty of room! 👍"
```
---
### INFO (Lifecycle Events — Narration Backbone)
**Narration Intensity**: Production-ready stories
**What belongs**:
- ✅ Service lifecycle (startup, shutdown, ready)
- ✅ Model lifecycle (load, seal, verify, unload)
- ✅ Job lifecycle (accept, queue, dispatch, complete)
- ✅ Pool changes (worker ready, pool registered, state transitions)
- ✅ User-facing operations (request accepted, result returned)
**Editorial guidelines**:
- 📏 Length: **≤100 characters** (ORCH-3305 requirement — we're STRICT!)
- 🔗 Correlation: **ALWAYS** propagate (non-negotiable)
- 🎯 Clarity: Operator-readable without context
- 🎀 Cute mode: **ALWAYS** (this is our bread and butter!)
**Examples**:
```rust
// INFO: Service startup
human: "Worker started successfully on GPU0 with engine llamacpp-v1"
cute: "Worker-gpu0-r1 wakes up and waves hello! Ready to help! 👋✨"
// INFO: Job acceptance
human: "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'"
cute: "Orchestratord welcomes job-123 to the queue! You're #3 in line! 🎫✨"
// INFO: Job completion
human: "Job completed in 2500 ms (150 tokens, 60 tokens/sec)"
cute: "Job-123 finished! Generated 150 tokens super fast! 🎉"
// INFO: Model loaded
human: "Sealed model shard 'llama-7b' in 2048 MB VRAM on GPU 0 (5 ms)"
cute: "Tucked llama-7b safely into GPU0's warm 2GB nest! Sweet dreams! 🛏️✨"
```
---
### WARN (Anomalies & Degradations)
**Narration Intensity**: Actionable warnings
**What belongs**:
- ✅ Retry attempts ("Retrying job-123 after timeout (attempt 2/5)")
- ✅ Deprecated flags ("Using deprecated flag --legacy-mode; migrate to --compat-mode")
- ✅ Performance degradation ("Worker-gpu0-r1 latency: 500ms (expected <100ms)")
- ✅ Resource pressure ("Queue depth: 95/100 (95% capacity)")
- ✅ Non-fatal errors ("Heartbeat missed from worker-gpu0-r1 (last seen 5000ms ago)")
**Editorial guidelines**:
- 📏 Length: ≤120 characters (need room for context)
- 🔗 Correlation: Always propagate
- 🎯 Actionability: Include what's wrong AND what to do
- ⚠️ Severity: Not fatal, but needs attention
- 🎀 Cute mode: Gentle, concerned tone
**Examples**:
```rust
// WARN: Retry attempt
human: "Retrying job-123 after timeout (attempt 2/5, backoff: 200ms)"
cute: "Job-123 didn't make it through. Let's try again! Attempt 2... 🔄"
story: "\"Timeout!\" said worker. \"Let me try again,\" offered rbees-orcd."
// WARN: Deprecated flag
human: "Using deprecated flag --legacy-mode; migrate to --compat-mode by v2.0"
cute: "Psst! --legacy-mode is getting old. Switch to --compat-mode soon! 📢"
// WARN: Performance degradation
human: "Worker-gpu0-r1 latency: 500ms (expected <100ms, threshold exceeded)"
cute: "Worker-gpu0-r1 is running a bit slow today (500ms). Might need a checkup! 🐌"
// WARN: Resource pressure
human: "Queue depth: 95/100 (95% capacity, consider scaling)"
cute: "The queue is getting pretty full! 95 jobs waiting! Maybe add more workers? 📈"
// WARN: Missed heartbeat
human: "Heartbeat missed from worker-gpu0-r1 (last seen 5000ms ago, threshold: 3000ms)"
cute: "Haven't heard from worker-gpu0-r1 in 5 seconds. Are they okay? 😟"
```
---
### ERROR (Operational Failures)
**Narration Intensity**: Rich context for diagnosis
**What belongs**:
- ✅ Failed operations with context ("VRAM allocation failed: requested 4096MB, only 2048MB available")
- ✅ Validation failures ("Job validation failed: model 'invalid-model' not found in catalog")
- ✅ Network errors ("Failed to connect to pool-managerd: connection refused (retry in 1s)")
- ✅ Resource exhaustion ("GPU0 out of memory: 8192MB allocated, 0MB free")
- ✅ Timeout errors ("Job-123 timed out after 30s (max: 30s, worker: worker-gpu0-r1)")
**Editorial guidelines**:
- 📏 Length: ≤150 characters (need room for diagnosis)
- 🔗 Correlation: **ALWAYS** propagate (critical for debugging!)
- 🎯 Diagnosis: Include what failed, why, and relevant context
- 🔒 Redaction: Especially important for error details
- 🎀 Cute mode: Empathetic, helpful tone
**Examples**:
```rust
// ERROR: VRAM allocation failure
human: "VRAM allocation failed on GPU0: requested 4096MB, only 2048MB available"
cute: "Oh no! GPU0 doesn't have enough room (need 4GB, only 2GB free). 😟"
story: "\"Do you have 4GB?\" asked rbees-orcd. \"No,\" replied GPU0 sadly, \"only 2GB free.\""
// ERROR: Validation failure
human: "Job validation failed: model 'gpt-5' not found in catalog (available: llama-7b, phi-3)"
cute: "Hmm, 'gpt-5' isn't in our catalog. We have llama-7b and phi-3 though! 🔍"
// ERROR: Network failure
human: "Failed to connect to pool-managerd at localhost:8080: connection refused (retry in 1s)"
cute: "Couldn't reach pool-managerd! They might be napping. Trying again in 1s... 😴"
// ERROR: Resource exhaustion
human: "GPU0 out of memory: 8192MB allocated, 0MB free (cannot load model 'llama-13b')"
cute: "GPU0 is completely full! No room for llama-13b right now. 😰"
// ERROR: Timeout
human: "Job-123 timed out after 30s (max: 30s, worker: worker-gpu0-r1, correlation: req-abc)"
cute: "Job-123 took too long (30s limit). Worker-gpu0-r1 might be stuck! ⏰"
```
---
### FATAL (Unrecoverable Errors)
**Narration Intensity**: Maximum context + state dump
**What belongs**:
- ✅ Panic-worthy errors ("CRITICAL: VRAM-only policy violated on GPU0: UMA detected. Aborting.")
- ✅ Data corruption ("CRITICAL: Seal verification failed for shard 'llama-7b': digest mismatch")
- ✅ Invariant violations ("CRITICAL: Worker registry corrupted: duplicate worker_id 'worker-gpu0-r1'")
- ✅ Unrecoverable resource failures ("CRITICAL: All GPUs failed health check. Cannot continue.")
- ✅ Security violations ("CRITICAL: Unauthorized access attempt detected. Shutting down.")
**Editorial guidelines**:
- 📏 Length: Unlimited (this is a crisis, tell the whole story!)
- 🔗 Correlation: **ALWAYS** propagate
- 🎯 Context: Include EVERYTHING (state, inputs, stack trace context)
- 🚨 Severity: System cannot continue
- 🎀 Cute mode: Still gentle, but serious
**Examples**:
```rust
// FATAL: Policy violation
human: "CRITICAL: VRAM-only policy violated on GPU0: UMA detected. Worker startup aborted."
cute: "STOP! GPU0 shares memory with CPU (UMA) — we need dedicated VRAM! Shutting down. 🛑"
story: "\"UMA detected!\" cried worker. \"We can't continue,\" said rbees-orcd gravely. \"Abort.\""
// FATAL: Data corruption
human: "CRITICAL: Seal verification failed for shard 'llama-7b' on GPU0: digest mismatch (expected: abc123, got: def456)"
cute: "DANGER! llama-7b's safety seal is wrong! This could be corruption! Stopping everything! 🚨"
// FATAL: Invariant violation
human: "CRITICAL: Worker registry corrupted: duplicate worker_id 'worker-gpu0-r1' (existing: GPU0, new: GPU1)"
cute: "PANIC! Two workers claim to be 'worker-gpu0-r1'! This should never happen! 😱"
// FATAL: Total resource failure
human: "CRITICAL: All GPUs failed health check (GPU0: OOM, GPU1: driver error, GPU2: offline). Cannot continue."
cute: "All GPUs are down! GPU0 is out of memory, GPU1 has driver issues, GPU2 is offline. We can't run! 💔"
// FATAL: Security violation
human: "CRITICAL: Unauthorized access attempt from IP 192.168.1.100 (invalid token). Shutting down for security."
cute: "SECURITY ALERT! Someone tried to access us without permission! Shutting down to stay safe! 🔒"
```
---
## 🔌 Our Custom Tracing Integration Architecture
### Our Decision: Build Our Own! 🎀
We're **building custom proc macros** that integrate with `tracing` as a backend, but with our unique features on top:
**What we're building**:
- ✅ Custom `#[trace_fn]` with auto-inferred actor
- ✅ Custom `#[narrate(...)]` with template interpolation
- ✅ Compile-time editorial enforcement (≤100 chars, SVO validation)
- ✅ Cute/story modes built-in (first-class, not add-on)
- ✅ Conditional compilation (zero overhead in production)
- ✅  integration
### Original Options We Considered:
#### Option 1: Narration as Tracing Fields (Embedded)
**How it works**: Narration fields become structured fields on tracing events.
```rust
tracing::info!(
    actor = "rbees-orcd",
    action = "accept",
    target = "job-123",
    correlation_id = "req-abc",
    human = "Accepted request; queued at position 3",
    cute = "Orchestratord welcomes job-123 to the queue! 🎫",
    "Narration event"
);
```
**Pros**:
- ✅ Single event stream (no duplication)
- ✅ Tracing subscriber handles formatting
- ✅ Natural RUST_LOG filtering
**Cons**:
- ❌ Couples narration to tracing (we lose independence)
- ❌ Harder to enforce editorial standards (tracing doesn't know our rules)
- ❌ Can't easily swap backends (we're stuck with tracing)
**Our take**: 😤 **We don't love this.** We lose editorial control.
---
#### Option 2: Narration Emits Tracing Events (Wrapper)
**How it works**: `narrate()` internally calls `tracing::event!()` with our fields.
```rust
pub fn narrate(fields: NarrationFields) {
    // Apply redaction, validation, editorial rules
    let sanitized = apply_editorial_standards(fields);
    // Emit as tracing event
    tracing::info!(
        actor = sanitized.actor,
        action = sanitized.action,
        target = sanitized.target,
        human = sanitized.human,
        cute = sanitized.cute,
        correlation_id = sanitized.correlation_id,
        "Narration"
    );
}
```
**Pros**:
- ✅ We maintain editorial control (redaction, validation, ≤100 char enforcement)
- ✅ Single event stream (no duplication)
- ✅ Tracing subscriber handles output formatting
- ✅ RUST_LOG filtering works naturally
**Cons**:
- ⚠️ Narration depends on tracing (acceptable trade-off)
- ⚠️ Need to map narration levels to tracing levels
**Our take**: 🎀 **This is our favorite!** We keep control, you get integration.
---
#### Option 3: Parallel Streams (Independent)
**How it works**: Narration and tracing are completely separate.
```rust
// Narration event
narrate(NarrationFields {
    actor: "rbees-orcd",
    action: "accept",
    target: "job-123".to_string(),
    human: "Accepted request".to_string(),
    ..Default::default()
});
// Separate tracing event
tracing::info!(job_id = "job-123", "Job accepted");
```
**Pros**:
- ✅ Complete independence (we can swap backends)
- ✅ Full editorial control
- ✅ Narration can have different output (JSON, NDJSON, etc.)
**Cons**:
- ❌ Duplicate events (one narration, one tracing)
- ❌ Two streams to correlate
- ❌ More complex for consumers
**Our take**: 😤 **Too much duplication.** Operators will hate this.
---
### 🏆 Our Implementation: Custom Proc Macros + Tracing Backend
**Why we're building our own**:
- 💝 Cute mode is our **brand** — needs to be first-class
- 🎭 Story mode is **unique** — no other library has it
- 🎨 Editorial enforcement is **our standard** — compile-time validation
- 🔒 Security is **built-in** — automatic redaction
- 📊  are **our workflow** — seamless integration
- 💝 Brand differentiation matters
**Implementation strategy**:
```rust
// bin/shared-crates/narration-macros/src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};
/// Our custom #[trace_fn] proc macro
#[proc_macro_attribute]
pub fn trace_fn(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    // Auto-infer actor from module path!
    let actor = infer_actor_from_module_path();
    // Conditional compilation
    #[cfg(feature = "trace-enabled")]
    {
        // Generate full tracing code with auto-inferred actor
        let expanded = generate_trace_code(&input, &actor);
        return TokenStream::from(expanded);
    }
    #[cfg(not(feature = "trace-enabled"))]
    {
        // Production: return original function unchanged
        return TokenStream::from(quote! { #input });
    }
}
/// Our custom #[narrate(...)] proc macro with template interpolation
#[proc_macro_attribute]
pub fn narrate(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let args = parse_macro_input!(attr as NarrateArgs);
    // Validate at compile time!
    if args.human.len() > 100 {
        return syn::Error::new_spanned(
            &args.human,
            "human field exceeds 100 character limit (ORCH-3305)"
        ).to_compile_error().into();
    }
    // Generate code with template interpolation
    let expanded = generate_narration_with_templates(&input, &args);
    TokenStream::from(expanded)
}
```
```rust
// narration-core/src/lib.rs
use tracing::Level;
/// Emit narration as tracing event (used by our proc macros)
pub fn narrate(mut fields: NarrationFields) {
    // Apply editorial standards
    enforce_length_limits(&mut fields);  // ≤100 chars for INFO
    apply_redaction(&mut fields);         // Automatic secret redaction
    validate_correlation_id(&fields);     // Ensure tracking
    // Emit as tracing event with structured fields
    tracing::info!(
        actor = fields.actor,
        action = fields.action,
        target = %fields.target,
        human = %fields.human,
        cute = fields.cute.as_deref(),
        story = fields.story.as_deref(),
        correlation_id = fields.correlation_id.as_deref(),
        // ... all other fields
    );
}
```
**Benefits of our custom implementation**:
- ✅ **Auto-inferred actor** from module path (zero boilerplate!)
- ✅ **Template interpolation** for human/cute/story fields
- ✅ **Compile-time editorial enforcement** (≤100 chars, SVO validation)
- ✅ **Cute/story modes built-in** (first-class, not add-on)
- ✅ **Conditional compilation** (code removed in production)
- ✅ We enforce ≤100 character limit before tracing sees it
- ✅ We apply redaction before tracing sees it
- ✅ We validate correlation IDs before tracing sees it
- ✅ RUST_LOG filtering works: `RUST_LOG=info` shows INFO+ narration
- ✅ Single event stream (no duplication)
- ✅ Tracing subscriber formats output (JSON, console, etc.)
- ✅ **Brand differentiation** (uniquely "us")
---
## 🚨 Pitfalls We've Seen (And How to Avoid Them)
### Pitfall 1: Level Confusion
**Problem**: Teams use INFO for internal details, DEBUG for lifecycle events.
**Example (WRONG)**:
```rust
// ❌ This is DEBUG, not INFO!
tracing::info!("Loop iteration 47/100 processing worker-gpu2-r1");
// ❌ This is INFO, not DEBUG!
tracing::debug!("Job completed in 2500ms");
```
**Fix**: Use our level boundaries!
```rust
// ✅ CORRECT: DEBUG for internal flow
narrate_debug!(human = "Processing worker 47/100: worker-gpu2-r1 (status=idle)");
// ✅ CORRECT: INFO for lifecycle
narrate_info!(human = "Job completed in 2500 ms (150 tokens, 60 tokens/sec)");
```
---
### Pitfall 2: Missing Correlation IDs
**Problem**: Teams forget to propagate correlation IDs, breaking request tracing.
**Example (WRONG)**:
```rust
// ❌ No correlation_id!
narrate_info!(
    actor = "rbees-orcd",
    action = "dispatch",
    target = "job-123",
    human = "Dispatching job to worker"
);
```
**Fix**: ALWAYS propagate correlation IDs!
```rust
// ✅ CORRECT: Correlation ID propagated
narrate_info!(
    actor = "rbees-orcd",
    action = "dispatch",
    target = "job-123",
    correlation_id = req_id, // ALWAYS INCLUDE THIS!
    human = "Dispatching job to worker-gpu0-r1"
);
```
---
### Pitfall 3: Vague Error Messages
**Problem**: ERROR events lack context for diagnosis.
**Example (WRONG)**:
```rust
// ❌ Too vague!
narrate_error!(human = "Allocation failed");
```
**Fix**: Include what, why, and context!
```rust
// ✅ CORRECT: Rich context
narrate_error!(
    human = "VRAM allocation failed on GPU0: requested 4096MB, only 2048MB available",
    error_kind = "vram_exhaustion",
    requested_mb = 4096,
    available_mb = 2048,
    device = "GPU0"
);
```
---
### Pitfall 4: TRACE in Production
**Problem**: Teams enable TRACE in production, overwhelming logs.
**Example (WRONG)**:
```bash
# ❌ WRONG: TRACE will kill performance
export RUST_LOG=trace
```
**Fix**: Use INFO for production, DEBUG for incidents!
```bash
# ✅ CORRECT: INFO for production
export RUST_LOG=info
# ✅ CORRECT: DEBUG for incident investigation
export RUST_LOG=debug
# ✅ CORRECT: TRACE for specific module only
export RUST_LOG=info,llama_orch::worker::inference=trace
```
---
### Pitfall 5: Ignoring Length Limits
**Problem**: Teams write 200-character INFO messages that overwhelm operators.
**Example (WRONG)**:
```rust
// ❌ Way too long! (180 characters)
narrate_info!(
    human = "Accepted inference request for model 'llama-7b' with max_tokens=150, temperature=0.7, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.3, and queued at position 3"
);
```
**Fix**: Stay under 100 characters for INFO!
```rust
// ✅ CORRECT: Concise (72 characters)
narrate_info!(
    human = "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'",
    model_ref = "llama-7b",
    max_tokens = 150,
    // Other params as structured fields, not in human!
);
```
---
## 📝 What We're Building (4-Week Plan)
### Implementation Timeline
**Week 1**: Core proc macro crate
- Custom `#[trace_fn]` with auto-inferred actor
- Custom `#[narrate(...)]` with template interpolation
- Conditional compilation support
**Week 2**: Narration core enhancements
- Add WARN/ERROR/FATAL levels
- Lightweight trace macros (trace_tiny!, trace_loop!, etc.)
- Secret redaction (regex-based, cached)
- Integrate with `tracing` backend
**Week 3**: Editorial enforcement
- Compile-time validation (≤100 chars, SVO structure)
- Feature flags (trace-enabled, debug-enabled, cute-mode, production)
- Helpful compile errors for violations
**Week 4**: Integration & testing
- BDD tests for cute/story modes
-  integration
- Migrate services (rbees-orcd → pool-managerd → worker-orcd)
- Update CI/CD pipelines
### What We'll Deliver
#### 1. Editorial Annotations
- ✅ Level boundary validation ("This belongs in DEBUG, not INFO")
- ✅ Length limit enforcement ("This is 127 chars, needs to be ≤100")
- ✅ Correlation ID discipline ("Missing correlation_id here!")
- ✅ Context richness ("Add the error_kind and device fields")
#### 2. Cute Story Examples
For every WARN/ERROR case you define, we'll provide:
- 📢 **human**: Professional, concise, context-rich
- 🎀 **cute**: Whimsical, empathetic, emoji-enhanced
- 🎭 **story**: Dialogue-based (when it makes sense)
#### 3. Integration Guidance
- 🔌 How to wire narration into your tracing subscriber
- 🎚️ RUST_LOG patterns for different environments
- 🧪 Testing strategies for each level
- 📊 Performance impact analysis
#### 4. Anti-Pattern Catalog
- 🚨 Common mistakes we've seen
- ✅ Fixes with code examples
- 📋 Checklist for code review
---
## 🎀 Example: WARN/ERROR Cute Stories (Built Into Our System!)
### WARN Examples
**Retry scenario**:
```rust
// WARN: Retry attempt
human: "Retrying job-123 after timeout (attempt 2/5, backoff: 200ms)"
cute: "Job-123 didn't make it through. Let's try again! Attempt 2... 🔄"
story: "\"Timeout!\" said worker. \"Let me try again,\" offered rbees-orcd patiently."
```
**Performance degradation**:
```rust
// WARN: Slow worker
human: "Worker-gpu0-r1 latency: 500ms (expected <100ms, threshold exceeded)"
cute: "Worker-gpu0-r1 is running a bit slow today (500ms). Might need a checkup! 🐌"
story: "\"You're slower than usual,\" observed pool-managerd. \"I know,\" sighed worker-gpu0-r1."
```
**Deprecated feature**:
```rust
// WARN: Deprecated flag
human: "Using deprecated flag --legacy-mode; migrate to --compat-mode by v2.0"
cute: "Psst! --legacy-mode is getting old. Switch to --compat-mode soon! 📢"
story: "\"Still using --legacy-mode?\" asked rbees-orcd. \"Yeah, I should update,\" admitted the config."
```
### ERROR Examples
**VRAM exhaustion**:
```rust
// ERROR: Out of memory
human: "VRAM allocation failed on GPU0: requested 4096MB, only 2048MB available"
cute: "Oh no! GPU0 doesn't have enough room (need 4GB, only 2GB free). 😟"
story: "\"Do you have 4GB?\" asked rbees-orcd. \"No,\" replied GPU0 sadly, \"only 2GB free.\""
```
**Model not found**:
```rust
// ERROR: Validation failure
human: "Job validation failed: model 'gpt-5' not found in catalog (available: llama-7b, phi-3)"
cute: "Hmm, 'gpt-5' isn't in our catalog. We have llama-7b and phi-3 though! 🔍"
story: "\"Do you have gpt-5?\" asked the client. \"No,\" replied rbees-orcd, \"but I have llama-7b and phi-3!\""
```
**Network failure**:
```rust
// ERROR: Connection refused
human: "Failed to connect to pool-managerd at localhost:8080: connection refused (retry in 1s)"
cute: "Couldn't reach pool-managerd! They might be napping. Trying again in 1s... 😴"
story: "\"Hello?\" called rbees-orcd. Silence. \"I'll try again soon,\" it decided."
```
### FATAL Examples
**Policy violation**:
```rust
// FATAL: VRAM-only violation
human: "CRITICAL: VRAM-only policy violated on GPU0: UMA detected. Worker startup aborted."
cute: "STOP! GPU0 shares memory with CPU (UMA) — we need dedicated VRAM! Shutting down. 🛑"
story: "\"UMA detected!\" cried worker. \"We can't continue,\" said rbees-orcd gravely. \"Abort.\""
```
**Data corruption**:
```rust
// FATAL: Seal verification failed
human: "CRITICAL: Seal verification failed for shard 'llama-7b' on GPU0: digest mismatch (expected: abc123, got: def456)"
cute: "DANGER! llama-7b's safety seal is wrong! This could be corruption! Stopping everything! 🚨"
story: "\"The seal doesn't match!\" gasped worker. \"Corruption?\" asked rbees-orcd. \"Possibly. Abort!\""
```
---
## ✅ Our Recommendations Summary
### 1. Level Mapping
✅ **APPROVED** with minor updates:
- Add WARN, ERROR, FATAL to our spec (we're doing this now)
- Keep INFO as narration backbone (perfect!)
- TRACE for opt-in ultra-fine detail (exactly right)
### 2. Tracing Integration
🎀 **RECOMMENDED**: Option 2 (Wrapper approach)
- Narration wraps tracing events
- We maintain editorial control
- Single event stream
- RUST_LOG filtering works naturally
### 3. Editorial Standards
💝 **ENFORCED**:
- INFO: ≤100 characters (strict)
- WARN: ≤120 characters (actionable context)
- ERROR: ≤150 characters (diagnostic context)
- FATAL: Unlimited (full state dump)
- ALL levels: Correlation IDs required
- ALL levels: Secret redaction enforced
### 4. Cute Mode
🎀 **ALWAYS AVAILABLE**:
- Every level gets cute narration
- Even FATAL can be gentle
- Emoji-enhanced, empathetic tone
- Story mode for dialogue (optional)
---
## 📅 Next Steps
### What We Need From You
1. **Draft Policy Document**
   - Send us your proposed logging level policy
   - Include example scenarios for each level
   - We'll provide annotated feedback within 48 hours
2. **Integration Requirements**
   - Which tracing subscriber are you using? (tracing-subscriber, tracing-bunyan, etc.)
   - What output format? (JSON, console, NDJSON?)
   - Any custom fields beyond our taxonomy?
3. **Rollout Plan**
   - Which services first? (rbees-orcd, pool-managerd, worker-orcd?)
   - Timeline for migration?
   - Backward compatibility needs?
### What We'll Provide
1. **Updated LOGGING_LEVELS.md**
   - Add WARN, ERROR, FATAL levels
   - Include tracing integration guidance
   - Provide cute examples for all levels
2. **Tracing Integration Module**
   - `narration-core/src/tracing_integration.rs`
   - Wrapper functions: `narrate_trace!()`, `narrate_debug!()`, etc.
   - Level mapping and editorial enforcement
3. **Editorial Review**
   - Annotated feedback on your draft policy
   - Cute story examples for WARN/ERROR cases
   - Anti-pattern catalog with fixes
4. **Migration Guide**
   - Step-by-step integration instructions
   - Code examples for each service
   - Testing strategies
---
## 💝 Final Thoughts
We're **thrilled** that you're taking logging levels seriously and aligning them with narration intensity. Your proposed mapping is **excellent**, and we're excited to collaborate on making this the **most debuggable distributed system in existence**.
A few parting thoughts:
### What We Love ❤️
- ✅ INFO as the narration backbone (FINALLY!)
- ✅ WARN/ERROR with rich context (music to our ears!)
- ✅ TRACE for opt-in ultra-fine detail (perfect boundary)
- ✅ Your commitment to correlation IDs (we're so proud!)
### What We're Excited About 🎉
- 🎀 Cute mode at ALL levels (even FATAL!)
- 🔌 Tracing integration (we'll make it seamless)
- 📚 Comprehensive examples (we'll provide dozens)
- 🚀 Rollout across all services (narration everywhere!)
### What We're Watching For 👀
- ⚠️ Level confusion (we'll catch it in review)
- ⚠️ Missing correlation IDs (we'll enforce it)
- ⚠️ Vague error messages (we'll demand context)
- ⚠️ TRACE in production (we'll stop you!)
---
## 🤝 Let's Do This!
We're building a **world-class, uniquely branded narration system** with:
- 🎀 Custom proc macros (`#[trace_fn]`, `#[narrate(...)]`)
- 🎭 Auto-inferred actors (from module path)
- 🎨 Template interpolation (for human/cute/story)
- 🔒 Compile-time editorial enforcement
- 📊 Conditional compilation (zero overhead in production)
- 💝 Cute/story modes built-in (first-class!)
We have **ultimate editorial authority** over `human` fields, and we take that responsibility **seriously** (but adorably). We're building something that's **uniquely ours** because:
**Cuteness pays the bills!** 💕
Generic tracing is boring. We're making llama-orch logs the **most delightful debugging experience** in distributed systems.
---
**With love, sass, and the confidence that cuteness pays the bills,**  
**The Narration Core Team** 🎭🎀
*P.S. — We're not using generic `tracing::instrument`. We're building a **cute, story-telling, editorially-enforced** narration system that's uniquely ours. Because boring is for other people. 💝*
---
**Deliverables**:
- 📄 LOGGING_LEVELS.md (updated with WARN/ERROR/FATAL)
- 📄 ERGONOMIC_TRACING.md (custom proc macro design)
- 📄 CONDITIONAL_COMPILATION.md (zero overhead in production)
- 📄 EXISTING_SOLUTIONS.md (why we're building our own)
- 📄 FINAL_SUMMARY.md (complete 4-week plan)
- 📄 Custom proc macro crate (`observability-narration-macros`)
*May your levels be clear, your correlation IDs present, your actors be auto-inferred, and your narration be adorable! 🎀*
