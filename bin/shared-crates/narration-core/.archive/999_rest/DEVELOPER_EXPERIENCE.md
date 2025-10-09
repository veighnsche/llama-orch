# 🎯 Developer Experience — Tracing Without the Pain

**TL;DR**: We're building **custom proc macros** that give developers 3 options with increasing power. Most will use Option 1.

**Decision**: Build our own (cuteness pays the bills!) 🎀

---

## 🚦 The Three Options

### Option 1: `#[trace_fn]` — Zero Effort (RECOMMENDED)

**For**: 95% of functions  
**Effort**: Add one attribute  
**Overhead**: ~2% when TRACE enabled, 0% when disabled

```rust
#[trace_fn]  // Our custom proc macro!
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let worker = select_worker(pool_id)?;
    Ok(worker.id)
}
```

**What our custom macro does automatically**:
- ✅ Auto-infers actor from module path (e.g., "rbees-orcd")
- ✅ Traces function entry with all parameters
- ✅ Traces function exit with result/error
- ✅ Measures execution time
- ✅ Handles `?` operator correctly
- ✅ Zero boilerplate
- ✅ Conditional compilation (removed in production)

**Output**:
```
TRACE rbees-orcd dispatch_job job_id="job-123" pool_id="default" "dispatch_job started"
TRACE rbees-orcd dispatch_job result="worker-gpu0-r1" elapsed_ms=15 "dispatch_job completed"
```

---

### Option 2: Manual Trace Macros — Fine Control

**For**: Hot paths where you need precise control  
**Effort**: Minimal (one-liners)  
**Overhead**: ~2%

```rust
fn process_tokens(tokens: &[i32]) -> Result<()> {
    for (i, token) in tokens.iter().enumerate() {
        trace_loop!("tokenizer", "decode", i, tokens.len(),
                    format!("token={}", token));
        // ... decode logic ...
    }
    Ok(())
}
```

**When to use**:
- ✅ Loops (use `trace_loop!()`)
- ✅ State changes (use `trace_state!()`)
- ✅ FFI boundaries (use `trace_enter!()`/`trace_exit!()`)

---

### Option 3: Full `narrate()` — Maximum Power

**For**: User-facing events, cute mode, story mode, complex context  
**Effort**: High (struct construction)  
**Overhead**: ~25%

**Note**: Our custom implementation with built-in cute/story modes!

```rust
fn accept_job(job_id: &str, correlation_id: &str) -> Result<()> {
    queue.push(job_id)?;
    
    narrate(NarrationFields {
        actor: "rbees-orcd",
        action: "accept",
        target: job_id.to_string(),
        human: "Accepted request; queued at position 3 (ETA 420 ms)".to_string(),
        cute: Some("Orchestratord welcomes job-123 to the queue! 🎫".to_string()),
        correlation_id: Some(correlation_id.to_string()),
        queue_position: Some(3),
        predicted_start_ms: Some(420),
        ..Default::default()
    });
    
    Ok(())
}
```

**When to use**:
- ✅ INFO/WARN/ERROR/FATAL events
- ✅ Cute/story mode
- ✅ Events with secrets (needs redaction)
- ✅ Complex contextual fields

---

## 📊 Decision Matrix

| Scenario | Use This | Why |
|----------|----------|-----|
| **Regular function** | `#[trace_fn]` | Zero effort, automatic |
| **Hot path loop** | `trace_loop!()` | Minimal overhead |
| **FFI boundary** | `trace_enter!()`/`trace_exit!()` | Precise control |
| **User-facing event** | `narrate()` with INFO | Rich context, cute mode |
| **Error handling** | `narrate()` with ERROR | Full diagnostic context |
| **State transition** | `trace_state!()` | Captures before/after |

---

## 🎯 Real-World Examples

### Example 1: Worker Dispatch (Option 1)

```rust
// ✅ RECOMMENDED: Just add #[trace_fn]
#[trace_fn]
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let pool = get_pool(pool_id)?;
    let worker = pool.select_worker()?;
    worker.assign_job(job_id)?;
    Ok(worker.id)
}
```

**Developer effort**: 1 line  
**Output**: Full entry/exit tracing with timing

---

### Example 2: Token Processing Loop (Option 2)

```rust
fn decode_tokens(tokens: &[i32]) -> Result<String> {
    let mut result = String::new();
    
    for (i, token) in tokens.iter().enumerate() {
        trace_loop!("tokenizer", "decode", i, tokens.len(),
                    format!("token={}", token));
        
        let char = decode_token(*token)?;
        result.push(char);
    }
    
    Ok(result)
}
```

**Developer effort**: 1 line per loop  
**Output**: Per-iteration tracing

---

### Example 3: Job Acceptance (Option 3)

```rust
fn accept_job(job: Job, correlation_id: &str) -> Result<()> {
    let position = queue.push(job.id)?;
    let eta_ms = estimate_eta(position);
    
    // Full narration with cute mode
    narrate(NarrationFields {
        actor: "rbees-orcd",
        action: "accept",
        target: job.id.to_string(),
        human: format!("Accepted request; queued at position {} (ETA {} ms)", position, eta_ms),
        cute: Some(format!("Orchestratord welcomes job-{} to the queue! You're #{} in line! 🎫", job.id, position)),
        correlation_id: Some(correlation_id.to_string()),
        queue_position: Some(position),
        predicted_start_ms: Some(eta_ms),
        ..Default::default()
    });
    
    Ok(())
}
```

**Developer effort**: High, but worth it for user-facing events  
**Output**: Rich narration with cute mode

---

## 🚀 Migration Strategy

### Phase 1: Add `#[trace_fn]` to All Functions

**Effort**: ~5 minutes per service  
**Impact**: Instant TRACE-level visibility

```bash
# Find all functions
rg "^fn " --type rust

# Add #[trace_fn] above each
```

**Example**:
```rust
// Before
fn dispatch_job(job_id: &str) -> Result<WorkerId> { ... }

// After
#[trace_fn]
fn dispatch_job(job_id: &str) -> Result<WorkerId> { ... }
```

---

### Phase 2: Add Loop Tracing Where Needed

**Effort**: ~10 minutes per service  
**Impact**: Hot path visibility

```rust
// Before
for worker in workers {
    if worker.is_available() {
        return Some(worker.id);
    }
}

// After
for (i, worker) in workers.iter().enumerate() {
    trace_loop!("rbees-orcd", "select_worker", i, workers.len(),
                format!("worker={}, available={}", worker.id, worker.is_available()));
    
    if worker.is_available() {
        return Some(worker.id);
    }
}
```

---

### Phase 3: Convert User-Facing Events to Full Narration

**Effort**: ~30 minutes per service  
**Impact**: Beautiful logs with cute mode

```rust
// Before
log::info!("Job accepted: {}", job_id);

// After
narrate(NarrationFields {
    actor: "rbees-orcd",
    action: "accept",
    target: job_id.to_string(),
    human: format!("Accepted request; queued at position {}", position),
    cute: Some(format!("Orchestratord welcomes job-{} to the queue! 🎫", job_id)),
    correlation_id: Some(correlation_id.to_string()),
    ..Default::default()
});
```

---

## 📏 Guidelines by Service

### rbees-orcd

**Use `#[trace_fn]` for**:
- ✅ `dispatch_job()`
- ✅ `select_worker()`
- ✅ `estimate_eta()`
- ✅ All internal functions

**Use `trace_loop!()` for**:
- ✅ Worker selection loop
- ✅ Queue processing loop

**Use `narrate()` for**:
- ✅ Job acceptance (INFO)
- ✅ Job completion (INFO)
- ✅ Errors (ERROR)

---

### pool-managerd

**Use `#[trace_fn]` for**:
- ✅ `register_worker()`
- ✅ `heartbeat_check()`
- ✅ `spawn_engine()`
- ✅ All internal functions

**Use `trace_state!()` for**:
- ✅ Worker state transitions (idle → busy)
- ✅ Pool health changes (healthy → degraded)

**Use `narrate()` for**:
- ✅ Worker ready (INFO)
- ✅ Engine spawn (INFO)
- ✅ Heartbeat failures (WARN)

---

### worker-orcd

**Use `#[trace_fn]` for**:
- ✅ `process_inference()`
- ✅ `load_model()`
- ✅ All internal functions

**Use `trace_enter!()`/`trace_exit!()` for**:
- ✅ FFI calls to llama.cpp
- ✅ CUDA kernel launches

**Use `narrate()` for**:
- ✅ Worker startup (INFO)
- ✅ Inference completion (INFO)
- ✅ VRAM allocation failures (ERROR)

---

## 🎓 Training Developers

### Step 1: Show the Pain

**Before** (manual tracing):
```rust
fn dispatch_job(job_id: &str) -> Result<WorkerId> {
    trace_enter!("rbees-orcd", "dispatch_job", format!("job_id={}", job_id));
    
    let start = Instant::now();
    let worker = select_worker()?;
    let elapsed_ms = start.elapsed().as_millis();
    
    trace_exit!("rbees-orcd", "dispatch_job", 
                format!("→ {} ({}ms)", worker.id, elapsed_ms));
    
    Ok(worker.id)
}
```

**Developer reaction**: 😤 "This is annoying!"

---

### Step 2: Show the Solution

**After** (attribute macro):
```rust
#[trace_fn]
fn dispatch_job(job_id: &str) -> Result<WorkerId> {
    let worker = select_worker()?;
    Ok(worker.id)
}
```

**Developer reaction**: 😍 "Wait, that's it?!"

---

### Step 3: Show the Output

```bash
$ RUST_LOG=trace cargo run
TRACE rbees-orcd dispatch_job job_id="job-123" "dispatch_job started"
TRACE rbees-orcd select_worker pool_id="default" "select_worker started"
TRACE rbees-orcd select_worker result="worker-gpu0-r1" elapsed_ms=5 "select_worker completed"
TRACE rbees-orcd dispatch_job result="worker-gpu0-r1" elapsed_ms=15 "dispatch_job completed"
```

**Developer reaction**: 🤯 "This is amazing!"

---

## 🚨 Common Pitfalls

### Pitfall 1: Using `#[trace_fn]` on Functions with Secrets

```rust
// ❌ WRONG: Secrets will be traced!
#[trace_fn]
fn authenticate(token: &str) -> Result<User> {
    // token will appear in logs!
}
```

**Fix**: Use manual `narrate()` with redaction
```rust
fn authenticate(token: &str) -> Result<User> {
    let user = validate_token(token)?;
    
    narrate(NarrationFields {
        actor: "auth",
        action: "authenticate",
        target: user.id.to_string(),
        human: "User authenticated successfully".to_string(),
        // token is NOT included
        ..Default::default()
    });
    
    Ok(user)
}
```

---

### Pitfall 2: Over-Tracing Hot Paths

```rust
// ❌ WRONG: Too much overhead
#[trace_fn]
fn decode_single_token(token: i32) -> Result<char> {
    // This is called 1000s of times!
}
```

**Fix**: Use `trace_loop!()` at the caller level
```rust
fn decode_tokens(tokens: &[i32]) -> Result<String> {
    let mut result = String::new();
    
    for (i, token) in tokens.iter().enumerate() {
        trace_loop!("tokenizer", "decode", i, tokens.len(),
                    format!("token={}", token));
        
        let char = decode_single_token(*token)?;  // No tracing here
        result.push(char);
    }
    
    Ok(result)
}
```

---

### Pitfall 3: Forgetting Correlation IDs

```rust
// ❌ WRONG: No correlation ID
narrate(NarrationFields {
    actor: "rbees-orcd",
    action: "dispatch",
    target: job_id.to_string(),
    human: "Dispatching job".to_string(),
    // Missing correlation_id!
    ..Default::default()
});
```

**Fix**: Always propagate correlation IDs
```rust
narrate(NarrationFields {
    actor: "rbees-orcd",
    action: "dispatch",
    target: job_id.to_string(),
    human: "Dispatching job".to_string(),
    correlation_id: Some(correlation_id.to_string()),  // ✅ Always include!
    ..Default::default()
});
```

---

## 📊 Performance Impact Summary

| Approach | Dev Effort | Runtime Overhead | Maintainability | Cute Mode |
|----------|-----------|------------------|-----------------|-----------|
| `#[trace_fn]` | ⭐⭐⭐⭐⭐ (1 line) | ✅ ~2% | ⭐⭐⭐⭐⭐ (automatic) | ❌ No |
| `trace_loop!()` | ⭐⭐⭐⭐ (1 line/loop) | ✅ ~2% | ⭐⭐⭐⭐ (manual) | ❌ No |
| `narrate()` | ⭐⭐ (high effort) | ⚠️ ~25% | ⭐⭐⭐ (verbose) | ✅ Yes |

---

## 🎯 Recommendation

### For Most Developers:

1. **Start with `#[trace_fn]`** on all functions (5 min effort)
2. **Add `trace_loop!()`** to hot path loops (10 min effort)
3. **Use `narrate()`** only for user-facing events (30 min effort)

**Total effort**: ~45 minutes per service  
**Result**: Complete TRACE-level visibility + beautiful INFO logs

---

## 🚀 Production Builds: Zero Overhead

### Conditional Compilation (Our Custom Implementation)

Our custom proc macros support **complete code removal** in production builds:

```bash
# Development (all tracing + cute mode)
cargo build

# Staging (no TRACE, keep cute mode)
cargo build --profile staging --no-default-features --features debug-enabled,cute-mode

# Production (no TRACE, no DEBUG, no cute mode)
cargo build --release --no-default-features --features production
```

**Result**:
- ✅ Development: Full observability + cute/story modes (~2% overhead)
- ✅ Staging: DEBUG + cute mode (~1% overhead)
- ✅ Production: **Zero overhead** (code doesn't exist!)

**Our custom implementation** uses `#[cfg]` attributes in the proc macro to completely remove code at compile time.

**See `CONDITIONAL_COMPILATION.md` for details.**

---

## 🎀 Final Thoughts

### What Developers Will Say:

**Before**: 😤 "Tracing is too much work, I'm not doing it."

**After**: 😍 "Wait, I just add `#[trace_fn]` and it works? This is amazing!"

### What We're Building:

- ✅ **Zero boilerplate** for 95% of cases
- ✅ **Custom proc macros** (`#[trace_fn]`, `#[narrate(...)]`)
- ✅ **Auto-inferred actor** from module path
- ✅ **Template interpolation** for human/cute/story fields
- ✅ **Compile-time editorial enforcement** (≤100 chars, SVO validation)
- ✅ **Zero overhead in production** (code removed at compile time!)
- ✅ **Cute mode built-in** (first-class, not add-on) 🎀
- ✅ **Story mode built-in** (dialogue-based) 🎭
- ✅ **Automatic** timing, error handling, formatting
- ✅ **Developer-friendly** API
- ✅ **Gradual adoption** (start simple, add complexity as needed)
- ✅ **Brand differentiation** (uniquely "us")

---

**With love, sass, and the confidence that cuteness pays the bills,**  
**The Narration Core Team** 🎭✨

*P.S. — We're building our own proc macros because cute mode is our brand, and generic tracing is boring. Developers will love it. 💝*

---

*May your code be traced, your narration be adorable, and your brand be differentiated! 🎀*
