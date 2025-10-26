# TEAM-299: Phase 2 - Thread-Local Context Everywhere

**Status:** BLOCKED (Requires TEAM-298 completion)  
**Estimated Duration:** 1 week  
**Dependencies:** TEAM-298 (Phase 1 SSE Optional)  
**Risk Level:** Medium (touches many files)

---

## Mission

Eliminate manual `.job_id()` calls everywhere by using thread-local context. The `n!()` macro already supports this - now we need to set the context properly.

---

## ⚠️ CRITICAL: DO YOUR RESEARCH FIRST!

### Required Research

1. **Read TEAM-298 Handoff** - Understand SSE optional behavior
2. **Find All Manual .job_id() Calls** - Grep and count (should be 100+)
3. **Analyze Job Router Pattern** - How job execution starts
4. **Study Context System** - `src/context.rs` (understand what exists)
5. **Create Research Summary** - `.plan/TEAM_299_RESEARCH_SUMMARY.md`

**DO NOT CODE UNTIL RESEARCH IS COMPLETE!**

---

## Problem: Manual job_id Everywhere

```rust
// Current: Must manually add job_id to EVERY narration
n!("step1", "Step 1");  // ← No job_id, goes to stdout only
n!("step2", "Step 2");  // ← No job_id, goes to stdout only
n!("step3", "Step 3");  // ← No job_id, goes to stdout only
// ↑ These DON'T go to SSE! (no job_id in context)
```

## Solution: Set Context Once

```rust
// After: Set context once, all narration auto-injects job_id
with_narration_context(ctx.with_job_id(job_id), async {
    n!("step1", "Step 1");  // ← job_id auto-injected!
    n!("step2", "Step 2");  // ← job_id auto-injected!
    n!("step3", "Step 3");  // ← job_id auto-injected!
    // ↑ All go to SSE automatically!
}).await
```

---

## Implementation Tasks

### Task 1: Verify Context Auto-Injection Works

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

```rust
// Verify this code already exists from Phase 0:
pub fn macro_emit(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
) {
    // ...
    
    // TEAM-299: This should already work from Phase 0!
    let job_id = context::get_context().and_then(|ctx| ctx.job_id.clone());
    
    let fields = NarrationFields {
        // ...
        job_id,  // ← Automatically from context!
        ..Default::default()
    };
    
    narrate(fields);
}
```

**Test it works:**
```rust
#[tokio::test]
async fn test_context_auto_injection() {
    let ctx = NarrationContext::new().with_job_id("job-123");
    let adapter = CaptureAdapter::install();
    
    with_narration_context(ctx, async {
        n!("test", "Message");
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured[0].job_id, Some("job-123".to_string()));
}
```

### Task 2: Wrap Job Routers with Context

**File:** `bin/10_queen_rbee/src/job_router.rs`

```rust
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    registry: Arc<JobRegistry<String>>,
    hive_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>,
) -> Result<()> {
    let state = JobState { registry, hive_registry };
    
    // TEAM-299: Set thread-local context ONCE
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async move {
        let operation: Operation = serde_json::from_value(payload)?;
        let operation_name = operation.name();

        // TEAM-299: NO manual job_id needed!
        n!("route_job", "Executing operation: {}", operation_name);

        match operation {
            Operation::Status => {
                execute_status(state.hive_registry).await
            }
            
            Operation::Infer(request) => {
                execute_infer(request, state.hive_registry).await
            }
            
            op if op.should_forward_to_hive() => {
                hive_forwarder::forward_to_hive(op, state.hive_registry).await
            }
            
            _ => {
                Err(anyhow::anyhow!("Operation not implemented: {}", operation_name))
            }
        }
    }).await
}
```

**File:** `bin/20_rbee_hive/src/job_router.rs`

```rust
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState,
) -> Result<()> {
    // TEAM-299: Set thread-local context ONCE
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async move {
        let operation: Operation = serde_json::from_value(payload)?;
        
        // TEAM-299: NO manual job_id needed!
        n!("route_job", "Executing operation: {}", operation.name());
        
        match operation {
            // ... all handlers use n!() without job_id
        }
    }).await
}
```

### Task 3: Update Operation Handlers

**Pattern for each handler:**
```rust
// BEFORE (with builder):
NARRATE.action("hive_start")
    .job_id(&job_id)  // ← Manual!
    .context(&alias)
    .human("Starting hive {}")
    .emit();

// AFTER (with macro + context):
n!("hive_start", "Starting hive {}", alias);
// ↑ job_id auto-injected from thread-local context!
```

**Apply this pattern to:**
- [ ] All handlers in `queen-rbee/src/job_router.rs`
- [ ] All handlers in `rbee-hive/src/job_router.rs`
- [ ] Lifecycle handlers (hive-lifecycle, queen-lifecycle, etc.)
- [ ] Worker spawning handlers

### Task 4: Test Context Inheritance in Spawned Tasks

```rust
#[tokio::test]
async fn test_context_inherited_by_spawned_task() {
    let ctx = NarrationContext::new().with_job_id("job-123");
    let adapter = CaptureAdapter::install();
    
    with_narration_context(ctx, async {
        // Spawn a task
        let handle = tokio::spawn(async {
            n!("spawned", "From spawned task");
        });
        
        handle.await.unwrap();
    }).await;
    
    let captured = adapter.captured();
    assert_eq!(captured[0].job_id, Some("job-123".to_string()));
    // Context IS inherited! 🎉
}
```

**If context is NOT inherited (edge case):**
```rust
// Manually propagate context
let ctx_clone = context::get_context().unwrap();

tokio::spawn(with_narration_context(ctx_clone, async {
    n!("spawned", "Manual propagation");
}));
```

---

## Migration Strategy

### Week 3 Day 1-2: Job Routers
- Wrap `route_operation()` with `with_narration_context()`
- Test both queen and hive routers
- Verify SSE still works

### Week 3 Day 3-4: Operation Handlers
- Migrate 50% of handlers to use `n!()`
- Remove manual `.job_id()` calls
- Test each operation

### Week 3 Day 5: Complete Migration
- Migrate remaining 50% of handlers
- Remove all remaining `.job_id()` calls
- Full integration test

---

## Verification Checklist

- [ ] Both job routers wrapped with context
- [ ] All operation handlers use `n!()` macro
- [ ] No manual `.job_id()` calls remain (grep to verify)
- [ ] Context inherited in spawned tasks
- [ ] SSE routing still works correctly
- [ ] All tests pass

---

## Success Criteria

1. **No manual .job_id()** - 100+ calls eliminated
2. **Set once, use everywhere** - Context inheritance works
3. **SSE still works** - No regressions in routing
4. **Simpler code** - Less boilerplate everywhere

---

## Handoff to TEAM-300

Document in `.plan/TEAM_299_HANDOFF.md`:
1. How many `.job_id()` calls removed
2. Which files were updated
3. Any issues with context inheritance
4. Test results
5. Recommendations for Phase 3 (process capture)
