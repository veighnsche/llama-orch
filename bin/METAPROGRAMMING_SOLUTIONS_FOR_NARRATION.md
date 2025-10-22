# Metaprogramming Solutions for Narration Repetition

**Date:** 2025-10-22 03:43 AM  
**Status:** 🤔 EXPLORATION - Not yet implemented

---

## The Problem

Every narration needs `.job_id(&job_id)` and `.correlation_id(&corr_id)`:

```rust
NARRATE.action("hive_start").job_id(&job_id).correlation_id(&corr_id).human("Starting").emit();
NARRATE.action("hive_check").job_id(&job_id).correlation_id(&corr_id).human("Checking").emit();
NARRATE.action("hive_spawn").job_id(&job_id).correlation_id(&corr_id).human("Spawning").emit();
// ... 60+ times in job_router.rs
```

---

## Solution 1: Declarative Macro

### Implementation

```rust
// In narration-core/src/lib.rs
#[macro_export]
macro_rules! narrate_job {
    // With correlation_id
    ($job_id:expr, $corr_id:expr, $action:expr) => {{
        $crate::NARRATE
            .action($action)
            .job_id($job_id)
            .correlation_id($corr_id)
    }};
    
    // Without correlation_id (optional)
    ($job_id:expr, $action:expr) => {{
        $crate::NARRATE
            .action($action)
            .job_id($job_id)
    }};
}

// Usage
narrate_job!(&job_id, &corr_id, "hive_start")
    .human("Starting hive")
    .emit();

narrate_job!(&job_id, &corr_id, "hive_check")
    .human("Checking status")
    .emit();
```

### Pros
- ✅ Reduces repetition from 3 calls to 1
- ✅ Declarative and simple
- ✅ Type-safe
- ✅ Easy to understand

### Cons
- ❌ Still need to call macro on every narration
- ❌ Macro syntax (less discoverable than methods)
- ❌ IDE autocomplete doesn't work as well with macros
- ❌ Doesn't fully solve the problem

### Verdict: **Marginal improvement** (60+ macro calls vs 60+ `.job_id()` calls)

---

## Solution 2: Procedural Macro (Auto-inject)

### Implementation

```rust
// In narration-core-macros (proc-macro crate)
#[proc_macro_attribute]
pub fn auto_narration_context(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse function, find job_id and correlation_id parameters
    // Rewrite all NARRATE calls to automatically include them
}

// Usage
#[auto_narration_context(job_id, correlation_id)]
async fn route_operation(
    job_id: String,
    correlation_id: Option<String>,
    payload: serde_json::Value,
    ...
) -> Result<()> {
    // NARRATE calls automatically include job_id and correlation_id
    NARRATE.action("hive_start").human("Starting").emit();
    NARRATE.action("hive_check").human("Checking").emit();
}
```

### Pros
- ✅ Zero repetition in function body
- ✅ Clean, readable code
- ✅ Centralizes context in one place (function signature)

### Cons
- ❌ Magic! Hard to understand what's happening
- ❌ Requires separate proc-macro crate
- ❌ Complex implementation (AST rewriting)
- ❌ Hard to debug (macro expansion errors)
- ❌ IDE support varies
- ❌ Breaks if you rename parameters

### Verdict: **Too much magic** (makes code harder to understand)

---

## Solution 3: Code Generation

### Implementation

```rust
// In build.rs or xtask
// Generate wrapper functions for common patterns

// Generated code:
pub fn narrate_hive(job_id: &str, corr_id: Option<&str>) -> HiveNarrator {
    HiveNarrator { job_id, corr_id }
}

pub struct HiveNarrator {
    job_id: &str,
    corr_id: Option<&str>,
}

impl HiveNarrator {
    pub fn start(self) -> Narration {
        NARRATE
            .action("hive_start")
            .job_id(self.job_id)
            .correlation_id_opt(self.corr_id)
    }
    
    pub fn check(self) -> Narration {
        NARRATE
            .action("hive_check")
            .job_id(self.job_id)
            .correlation_id_opt(self.corr_id)
    }
}

// Usage
narrate_hive(&job_id, corr_id.as_deref())
    .start()
    .human("Starting hive")
    .emit();

narrate_hive(&job_id, corr_id.as_deref())
    .check()
    .human("Checking status")
    .emit();
```

### Pros
- ✅ Type-safe
- ✅ Good IDE support
- ✅ Reduces repetition

### Cons
- ❌ Still repetitive (call `narrate_hive()` every time)
- ❌ Need to generate code for every domain (hive, worker, model, etc.)
- ❌ Maintenance burden
- ❌ Less flexible

### Verdict: **Not worth the complexity**

---

## Solution 4: Builder with Defaults

### Implementation

```rust
// In narration-core/src/builder.rs
impl Narration {
    pub fn with_defaults(mut self, job_id: &str, corr_id: Option<&str>) -> Self {
        self.fields.job_id = Some(job_id.to_string());
        if let Some(id) = corr_id {
            self.fields.correlation_id = Some(id.to_string());
        }
        self
    }
}

// Usage
NARRATE
    .action("hive_start")
    .with_defaults(&job_id, corr_id.as_deref())
    .human("Starting hive")
    .emit();
```

### Pros
- ✅ Simple
- ✅ No macros
- ✅ Type-safe
- ✅ Flexible

### Cons
- ❌ Still call `.with_defaults()` on every narration
- ❌ Only saves 1 method call (2 → 1)
- ❌ Not a significant improvement

### Verdict: **Slight improvement** (but still repetitive)

---

## Solution 5: Thread-Local Context (Already Tried - FAILED)

### What We Tried

```rust
tokio::task_local! {
    static NARRATION_CONTEXT: RefCell<NarrationContext>;
}

with_narration_context(ctx, async move {
    NARRATE.action("hive_start").emit();  // Auto-inject job_id
}).await;
```

### Why It Failed

- Task-local context lookup in `.emit()` didn't work
- Events were emitted but SSE sink dropped them (no job_id)
- Unclear why the lookup failed
- Too much magic for the benefit

### Verdict: **FAILED** (see `/home/vince/Projects/llama-orch/bin/TASK_LOCAL_CONTEXT_DOESNT_WORK.md`)

---

## Solution 6: Implicit Context via Scoped Guard

### Implementation

```rust
// In narration-core/src/context.rs
pub struct NarrationScope {
    _marker: PhantomData<()>,
}

impl NarrationScope {
    pub fn new(job_id: String, correlation_id: Option<String>) -> Self {
        // Store in thread-local
        CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = Some(NarrationContext { job_id, correlation_id });
        });
        Self { _marker: PhantomData }
    }
}

impl Drop for NarrationScope {
    fn drop(&mut self) {
        // Clear context
        CONTEXT.with(|ctx| {
            *ctx.borrow_mut() = None;
        });
    }
}

// Usage
async fn route_operation(...) -> Result<()> {
    let _scope = NarrationScope::new(job_id.clone(), correlation_id);
    
    // All narrations automatically include job_id/correlation_id
    NARRATE.action("hive_start").human("Starting").emit();
    NARRATE.action("hive_check").human("Checking").emit();
    
    Ok(())
}
```

### Pros
- ✅ Set once, use everywhere
- ✅ RAII cleanup (Drop trait)
- ✅ No repetition in function body

### Cons
- ❌ Thread-local (doesn't work with async/await properly)
- ❌ Similar to solution 5 (which failed)
- ❌ Implicit context (harder to reason about)
- ❌ Might not work across await points

### Verdict: **Likely to fail** (same issues as task-local approach)

---

## Solution 7: Just Accept It

### The Reality

```rust
// This is fine. Really.
NARRATE
    .action("hive_start")
    .job_id(&job_id)
    .correlation_id(&corr_id)
    .human("Starting hive")
    .emit();
```

### Why It's Okay

1. **Explicit is better than implicit** - You can see what's happening
2. **Type-safe** - Compiler checks everything
3. **Easy to debug** - No magic, no macros
4. **IDE support** - Autocomplete works perfectly
5. **It's only 2 extra method calls** - Not a huge burden
6. **The architecture requires it** - SSE security demands job_id

### The Cost of "Fixing" It

Every solution above:
- Adds complexity
- Reduces clarity
- Makes debugging harder
- Introduces magic
- **Doesn't actually eliminate the repetition** (just moves it)

### Comparison

```rust
// Current (explicit)
NARRATE.action("start").job_id(&job_id).correlation_id(&corr_id).human("Starting").emit();

// Macro (still repetitive)
narrate_job!(&job_id, &corr_id, "start").human("Starting").emit();

// Builder helper (still repetitive)
NARRATE.action("start").with_defaults(&job_id, corr_id.as_deref()).human("Starting").emit();

// Proc macro (magic!)
#[auto_narration_context(job_id, correlation_id)]
fn route(...) {
    NARRATE.action("start").human("Starting").emit();  // Where did job_id go?!
}
```

**All solutions still require action on every narration.** The explicit approach is clearest.

---

## Recommendation

### ✅ Keep the current explicit approach

```rust
NARRATE
    .action("hive_start")
    .job_id(&job_id)
    .correlation_id(&corr_id)
    .human("Starting hive")
    .emit();
```

### Why?

1. **Clarity** - You know exactly what's happening
2. **Type safety** - Compiler catches mistakes
3. **Debuggability** - No magic to debug
4. **Maintainability** - Easy for others to understand
5. **Required by architecture** - SSE security needs job_id

### If you MUST reduce repetition...

**Best option: Simple macro helper**

```rust
#[macro_export]
macro_rules! job_narrate {
    ($job_id:expr, $corr_id:expr, $action:expr) => {
        NARRATE.action($action).job_id($job_id).correlation_id($corr_id)
    };
}

// Usage - saves ~20 characters per call
job_narrate!(&job_id, &corr_id, "hive_start")
    .human("Starting hive")
    .emit();
```

**But honestly:** The 20-character savings isn't worth the macro indirection.

---

## Conclusion

**Sometimes repetition is the answer.**

The `.job_id(&job_id)` repetition is:
- Required by the architecture (SSE security)
- Type-safe and explicit
- Easy to understand and debug
- Only 2 extra method calls

**Stop trying to eliminate it. Accept it and move on.**

---

## Sweet Dreams! 😴🌙

The current code is correct. Don't overcomplicate it with metaprogramming.
