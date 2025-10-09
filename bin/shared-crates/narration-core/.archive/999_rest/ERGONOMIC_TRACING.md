# 🎯 Ergonomic Tracing — Developer-Friendly Approach
**Problem**: Developers will **hate** manually adding trace macros everywhere.  
**Solution**: Build **custom procedural macros** that auto-inject tracing with our unique features.  
**Decision**: Build our own (cuteness pays the bills!) 🎀
---
## 😤 The Problem
### What Developers DON'T Want to Do:
```rust
// ❌ TERRIBLE: Manual trace calls everywhere
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    trace_enter!("rbees-orcd", "dispatch_job", 
                 format!("job_id={}, pool_id={}", job_id, pool_id));
    let worker = select_worker(pool_id)?;
    trace_exit!("rbees-orcd", "dispatch_job", 
                format!("→ {} ({}ms)", worker.id, elapsed_ms));
    Ok(worker.id)
}
```
**Why developers hate this**:
- 🤮 Boilerplate everywhere
- 🤮 Manual timing tracking
- 🤮 Easy to forget
- 🤮 Clutters the actual logic
---
## ✨ The Solution: Our Custom Attribute Macros
### What Developers WANT to Do:
```rust
// ✅ BEAUTIFUL: Just add our custom attribute!
#[trace_fn]  // Our custom proc macro!
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```
**Our custom macro auto-generates**:
- ✅ Auto-inferred actor from module path (e.g., "rbees-orcd")
- ✅ Entry trace with args
- ✅ Exit trace with result
- ✅ Automatic timing
- ✅ Error handling
- ✅ Conditional compilation (removed in production)
- ✅ Zero boilerplate
---
## 🛠️ Our Custom Implementation
### Option 1: `#[trace_fn]` — Function-Level Tracing (RECOMMENDED)
**Usage**:
```rust
use observability_narration_core::trace_fn;
#[trace_fn]  // Our custom proc macro!
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```
**Our custom macro expands to**:
```rust
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    // Auto-inferred actor from module path!
    let _span = tracing::trace_span!(
        "dispatch_job",
        actor = "rbees-orcd",  // ← Auto-inferred!
        job_id = %job_id,
        pool_id = %pool_id
    ).entered();
    let _start = std::time::Instant::now();
    let result = (|| {
        let worker = select_worker(pool_id)?;
        Ok(worker.id)
    })();
    let elapsed_ms = _start.elapsed().as_millis();
    match &result {
        Ok(worker_id) => {
            tracing::trace!(
                result = %worker_id,
                elapsed_ms = %elapsed_ms,
                "dispatch_job completed"
            );
        }
        Err(e) => {
            tracing::trace!(
                error = %e,
                elapsed_ms = %elapsed_ms,
                "dispatch_job failed"
            );
        }
    }
    result
}
```
**Benefits of our custom implementation**:
- ✅ Zero boilerplate for developers
- ✅ **Auto-inferred actor** from module path
- ✅ Automatic timing
- ✅ Automatic error handling
- ✅ Works with `?` operator
- ✅ Conditional compilation (removed in production)
- ✅ Respects RUST_LOG levels
---
### Option 2: `#[trace_loop]` — Auto-Trace Loop Iterations
**Usage**:
```rust
#[trace_loop]
for (i, worker) in workers.iter().enumerate() {
    if worker.is_available() {
        return Some(worker.id);
    }
}
```
**Expands to**:
```rust
for (i, worker) in workers.iter().enumerate() {
    tracing::trace!(
        iteration = i,
        total = workers.len(),
        worker_id = %worker.id,
        "loop iteration"
    );
    if worker.is_available() {
        return Some(worker.id);
    }
}
```
---
### Option 3: `#[trace_hot_path]` — Conditional Tracing
**Usage**:
```rust
#[trace_hot_path]  // Only traces if RUST_LOG=trace
fn llama_eval(ctx: *mut LlamaContext, tokens: &[i32]) -> Result<()> {
    unsafe { llama_cpp_eval(ctx, tokens.as_ptr(), tokens.len() as i32) }?;
    Ok(())
}
```
**Expands to**:
```rust
fn llama_eval(ctx: *mut LlamaContext, tokens: &[i32]) -> Result<()> {
    if tracing::enabled!(tracing::Level::TRACE) {
        let _span = tracing::trace_span!(
            "llama_eval",
            ctx = ?ctx,
            n_tokens = tokens.len()
        ).entered();
        // ... actual function body ...
    } else {
        // ... actual function body (no tracing overhead) ...
    }
    unsafe { llama_cpp_eval(ctx, tokens.as_ptr(), tokens.len() as i32) }?;
    Ok(())
}
```
**Benefits**:
- ✅ Zero overhead when TRACE is disabled
- ✅ Perfect for hot paths
- ✅ Conditional compilation
---
### Option 4: `#[narrate(...)]` — Full Narration with Cute Mode (Our Custom Implementation!)
**Usage**:
```rust
#[narrate(
    actor = "rbees-orcd",  // Or auto-inferred from module path!
    action = "dispatch",
    human = "Dispatched job {job_id} to worker {worker_id} ({elapsed_ms}ms)",
    cute = "Orchestratord sends job-{job_id} to worker-{worker_id}! 🎫"
)]
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```
**Our custom macro expands to** (with template interpolation!):
```rust
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let _start = std::time::Instant::now();
    let result = (|| {
        let worker = select_worker(pool_id)?;
        Ok(worker.id)
    })();
    let elapsed_ms = _start.elapsed().as_millis();
    match &result {
        Ok(worker_id) => {
            // Template interpolation happens here!
            narrate(NarrationFields {
                actor: "rbees-orcd",
                action: "dispatch",
                target: job_id.to_string(),
                human: format!("Dispatched job {} to worker {} ({}ms)", job_id, worker_id, elapsed_ms),
                cute: Some(format!("Orchestratord sends job-{} to worker-{}! 🎫", job_id, worker_id)),
                duration_ms: Some(elapsed_ms as u64),
                ..Default::default()
            });
        }
        Err(e) => {
            narrate(NarrationFields {
                actor: "rbees-orcd",
                action: "dispatch",
                target: job_id.to_string(),
                human: format!("Failed to dispatch job {}: {}", job_id, e),
                error_kind: Some(format!("{:?}", e)),
                duration_ms: Some(elapsed_ms as u64),
                ..Default::default()
            });
        }
    }
    result
}
```
**Unique features of our implementation**:
- ✅ **Template interpolation** — Variables extracted from context
- ✅ **Auto-inferred actor** — Optional, can be inferred from module path
- ✅ **Cute mode built-in** — First-class, not add-on
- ✅ **Story mode support** — Add `story = "..."` parameter
- ✅ **Compile-time validation** — Enforces ≤100 chars for human field
- ✅ **Conditional compilation** — Cute mode can be disabled in production
---
## 🎯 Recommended Developer Workflow
### For Hot Paths (FFI, CUDA, Loops):
```rust
// Just add #[trace_fn] — that's it!
#[trace_fn]
fn llama_eval(ctx: *mut LlamaContext, tokens: &[i32]) -> Result<()> {
    unsafe { llama_cpp_eval(ctx, tokens.as_ptr(), tokens.len() as i32) }?;
    Ok(())
}
```
### For User-Facing Operations:
```rust
// Use #[narrate] with cute mode
#[narrate(
    actor = "rbees-orcd",
    action = "accept",
    cute = "Orchestratord welcomes job-{job_id} to the queue! 🎫"
)]
fn accept_job(job_id: &str) -> Result<()> {
    queue.push(job_id)?;
    Ok(())
}
```
### For Loops:
```rust
// Just add #[trace_loop]
#[trace_loop]
for (i, worker) in workers.iter().enumerate() {
    if worker.is_available() {
        return Some(worker.id);
    }
}
```
---
## 🔧 Our Implementation Plan (4 Weeks)
### Phase 1: Create Our Custom Proc Macro Crate (Week 1)
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
**What we're building**:
- ✅ `#[trace_fn]` with auto-inferred actor
- ✅ `#[narrate(...)]` with template interpolation
- ✅ Conditional compilation support
- ✅ Compile-time editorial enforcement
### Phase 2: Implement Our Custom `#[trace_fn]` (Week 1)
```rust
// bin/shared-crates/narration-macros/src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};
#[proc_macro_attribute]
pub fn trace_fn(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_name = &input.sig.ident;
    let fn_name_str = fn_name.to_string();
    let inputs = &input.sig.inputs;
    let output = &input.sig.output;
    let block = &input.block;
    let vis = &input.vis;
    let attrs = &input.attrs;
    // Auto-infer actor from module path!
    let actor = infer_actor_from_module_path();
    // Extract parameter names for tracing
    let param_traces = inputs.iter().filter_map(|arg| {
        if let syn::FnArg::Typed(pat_type) = arg {
            if let syn::Pat::Ident(ident) = &*pat_type.pat {
                let name = &ident.ident;
                return Some(quote! { #name = %#name });
            }
        }
        None
    });
    // Conditional compilation!
    #[cfg(feature = "trace-enabled")]
    let expanded = quote! {
        #(#attrs)*
        #vis fn #fn_name(#inputs) #output {
            let _span = tracing::trace_span!(
                #fn_name_str,
                actor = #actor,  // Auto-inferred!
                #(#param_traces),*
            ).entered();
            let _start = std::time::Instant::now();
            let result = (|| #block)();
            let elapsed_ms = _start.elapsed().as_millis();
            match &result {
                Ok(value) => {
                    tracing::trace!(
                        result = ?value,
                        elapsed_ms = %elapsed_ms,
                        concat!(#fn_name_str, " completed")
                    );
                }
                Err(e) => {
                    tracing::trace!(
                        error = %e,
                        elapsed_ms = %elapsed_ms,
                        concat!(#fn_name_str, " failed")
                    );
                }
            }
            result
        }
    };
    // Production: no-op
    #[cfg(not(feature = "trace-enabled"))]
    let expanded = quote! { #input };
    TokenStream::from(expanded)
}
// Helper to infer actor from module path
fn infer_actor_from_module_path() -> String {
    // Extract from module path (e.g., "llama_orch::rbees-orcd" -> "rbees-orcd")
    // Implementation details...
    "rbees-orcd".to_string()
}
```
### Phase 3: Implement `#[narrate(...)]` with Template Interpolation (Week 2)
```rust
// bin/shared-crates/narration-macros/src/lib.rs
#[proc_macro_attribute]
pub fn narrate(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let args = parse_macro_input!(attr as NarrateArgs);
    // Extract template variables from human/cute/story fields
    let human_template = &args.human;
    let cute_template = &args.cute;
    // Validate at compile time!
    if human_template.len() > 100 {
        return syn::Error::new_spanned(
            human_template,
            "human field exceeds 100 character limit (ORCH-3305)"
        ).to_compile_error().into();
    }
    // Generate code with template interpolation
    // ... implementation
}
```
### Phase 4: Export from narration-core (Week 2)
```rust
// bin/shared-crates/narration-core/src/lib.rs
pub use observability_narration_macros::{trace_fn, trace_loop, narrate};
// Feature flags
#[cfg(feature = "cute-mode")]
pub const CUTE_MODE_ENABLED: bool = true;
#[cfg(not(feature = "cute-mode"))]
pub const CUTE_MODE_ENABLED: bool = false;
```
---
## 📊 Performance Comparison
| Approach | Developer Effort | Runtime Overhead | Maintainability |
|----------|-----------------|------------------|-----------------|
| **Manual trace macros** | 😤 High (boilerplate everywhere) | ✅ Low (~2%) | ❌ Poor (easy to forget) |
| **`#[trace_fn]` attribute** | ✅ Minimal (one line) | ✅ Low (~2%) | ✅ Excellent (automatic) |
| **Full `narrate()`** | 😤 High (struct construction) | ❌ High (~25%) | ⚠️ Medium |
---
## 🎯 Migration Guide
### Before (Manual Tracing):
```rust
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    trace_enter!("rbees-orcd", "dispatch_job", 
                 format!("job_id={}, pool_id={}", job_id, pool_id));
    let worker = select_worker(pool_id)?;
    trace_exit!("rbees-orcd", "dispatch_job", 
                format!("→ {} ({}ms)", worker.id, elapsed_ms));
    Ok(worker.id)
}
```
### After (Attribute Macro):
```rust
#[trace_fn]
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```
**Developer reaction**: 😍 "This is so much better!"
---
## 🚀 Advanced: Conditional Tracing
### Only Trace in Debug Builds:
```rust
#[cfg_attr(debug_assertions, trace_fn)]
fn hot_path_function() -> Result<()> {
    // ... hot path code ...
    Ok(())
}
```
**Result**:
- ✅ Tracing in debug builds (development)
- ✅ Zero overhead in release builds (production)
---
## 🎀 Cute Mode with Attributes
### Template-Based Cute Messages:
```rust
#[narrate(
    actor = "rbees-orcd",
    action = "dispatch",
    human = "Dispatched job {job_id} to worker {worker_id} ({elapsed_ms}ms)",
    cute = "Orchestratord sends job-{job_id} to worker-{worker_id}! 🎫"
)]
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```
**The macro**:
- ✅ Extracts `job_id`, `worker_id`, `elapsed_ms` from context
- ✅ Interpolates them into templates
- ✅ Emits full narration with cute mode
- ✅ Zero manual string formatting
---
## 📋 Developer Guidelines (Our Custom Implementation)
### ✅ DO Use Our Custom Attributes For:
1. **Hot paths** → `#[trace_fn]` (auto-infers actor!)
   ```rust
   #[trace_fn]  // Actor auto-inferred from module path!
   fn llama_eval(ctx: *mut LlamaContext, tokens: &[i32]) -> Result<()>
   ```
2. **User-facing operations** → `#[narrate(...)]` (with templates!)
   ```rust
   #[narrate(
       actor = "rbees-orcd",  // Or auto-inferred!
       action = "accept",
       human = "Accepted job {job_id} at position {position}",
       cute = "Orchestratord welcomes job-{job_id}! 🎫"
   )]
   fn accept_job(job_id: &str) -> Result<()>
   ```
3. **Loops** → `trace_loop!()` macro
   ```rust
   for (i, worker) in workers.iter().enumerate() {
       trace_loop!("rbees-orcd", "select", i, workers.len(),
                   format!("worker={}", worker.id));
   }
   ```
### ❌ DON'T Use Attributes For:
1. **Functions with secrets** (use manual `narrate()` with redaction)
2. **Complex control flow** (macro can't handle all cases)
3. **Non-Result return types** (macro assumes `Result<T, E>`)
---
## 🔧 Configuration
### Enable/Disable via Feature Flags:
```toml
# Cargo.toml
[dependencies]
observability-narration-core = { version = "0.1", features = ["trace-macros"] }
```
### Control via Environment:
```bash
# Enable TRACE for specific module
export RUST_LOG=info,llama_orch::rbees-orcd=trace
# Disable all tracing
export RUST_LOG=info
```
---
## 🎯 Summary
### The Problem:
- ❌ Manual trace macros are **boilerplate hell**
- ❌ Developers will **hate** sprinkling them everywhere
- ❌ Easy to **forget** or **misuse**
### The Solution:
- ✅ **Attribute macros** auto-inject tracing
- ✅ **Zero boilerplate** for developers
- ✅ **Same performance** as manual macros (~2% overhead)
- ✅ **Automatic** timing, error handling, and formatting
### Developer Experience:
**Before**:
```rust
// 😤 UGH, so much boilerplate!
fn dispatch_job(job_id: &str) -> Result<WorkerId> {
    trace_enter!("rbees-orcd", "dispatch_job", format!("job_id={}", job_id));
    let worker = select_worker()?;
    trace_exit!("rbees-orcd", "dispatch_job", format!("→ {}", worker.id));
    Ok(worker.id)
}
```
**After**:
```rust
// 😍 BEAUTIFUL!
#[trace_fn]
fn dispatch_job(job_id: &str) -> Result<WorkerId> {
    let worker = select_worker()?;
    Ok(worker.id)
}
```
---
## 📦 Implementation Checklist (4 Weeks)
### Week 1: Core Proc Macros
- [ ] Create `observability-narration-macros` proc macro crate
- [ ] Implement `#[trace_fn]` with auto-inferred actor
- [ ] Implement conditional compilation support
- [ ] Write unit tests for proc macros
### Week 2: Advanced Features
- [ ] Implement `#[narrate(...)]` with template interpolation
- [ ] Add compile-time editorial enforcement (≤100 chars, SVO)
- [ ] Implement lightweight trace macros (trace_loop!, etc.)
- [ ] Add cute-mode feature flag
### Week 3: Editorial & Compilation
- [ ] Compile-time validation (length limits, SVO structure)
- [ ] Feature flags (trace-enabled, debug-enabled, cute-mode, production)
- [ ] Helpful compile errors for violations
- [ ] Performance optimization
### Week 4: Integration & Testing
- [ ] BDD tests for cute/story modes
- [ ]  integration
- [ ] Document all attributes
- [ ] Create migration guide
- [ ] Update developer guidelines
---
**With love, sass, and the confidence that cuteness pays the bills,**  
**The Narration Core Team** 🎭✨
*P.S. — We're building our own proc macros because cute mode is our brand, editorial enforcement is our standard, and generic tracing is boring. Developers will love it. 💝*
---
*May your code be clean, your tracing be automatic, your actor be auto-inferred, and your narration be adorable! 🎀*
