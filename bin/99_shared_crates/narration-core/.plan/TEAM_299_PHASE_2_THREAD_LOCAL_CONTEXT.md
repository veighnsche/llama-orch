# TEAM-299: Phase 2 - Thread-Local Context Everywhere

**Status:** BLOCKED (Requires TEAM-298 completion)  
**Estimated Duration:** 1 week  
**Dependencies:** TEAM-298 (Phase 1)  
**Risk Level:** Medium (touches many files)

---

## Mission

Eliminate manual `.job_id()` calls by using thread-local context consistently. Set job context once at the start of job execution, then all narration automatically includes the job_id.

---

## ⚠️ CRITICAL: DO YOUR RESEARCH FIRST!

### Required Research (Complete ALL before coding)

1. **Read TEAM-298 Handoff** - Understand Phase 1 changes
2. **Read Context System** - `src/context.rs` and `src/builder.rs` (emit methods)
3. **Find All Manual job_id Calls** - Grep for `.job_id()` in all binaries
4. **Analyze Job Router Pattern** - Understand execution flow
5. **Create Research Summary** - Document in `.plan/TEAM_299_RESEARCH_SUMMARY.md`

**DO NOT CODE UNTIL RESEARCH IS COMPLETE!**

---

## Problem: Repetitive Manual job_id

```rust
// Current: Manual .job_id() everywhere (100+ locations)
NARRATE.action("step1").job_id(&job_id).emit();
NARRATE.action("step2").job_id(&job_id).emit();
NARRATE.action("step3").job_id(&job_id).emit();
```

## Solution: Thread-Local Context

```rust
// After: Set once, use everywhere
with_narration_context(ctx.with_job_id(job_id), async {
    NARRATE.action("step1").emit();  // ← Auto-injected!
    NARRATE.action("step2").emit();
    NARRATE.action("step3").emit();
}).await
```

---

## Implementation Tasks

### Task 1: Wrap Job Routers with Context

**Files:**
- `bin/10_queen_rbee/src/job_router.rs`
- `bin/20_rbee_hive/src/job_router.rs`

**Pattern:**
```rust
async fn route_operation(job_id: String, ...) -> Result<()> {
    let ctx = NarrationContext::new().with_job_id(&job_id);
    
    with_narration_context(ctx, async move {
        // All narration here auto-injects job_id
        NARRATE.action("route").emit();
        
        match operation {
            // ... handlers
        }
    }).await
}
```

### Task 2: Remove Manual .job_id() Calls

**Search and replace:**
```bash
# Find all occurrences
rg "\.job_id\(" bin/10_queen_rbee/
rg "\.job_id\(" bin/20_rbee_hive/
rg "\.job_id\(" bin/00_rbee_keeper/

# Remove .job_id(&job_id) from each
```

### Task 3: Remove job_id Parameters

**Pattern:**
```rust
// Before:
async fn execute_hive_start(job_id: &str, alias: String) -> Result<()>

// After:
async fn execute_hive_start(alias: String) -> Result<()>
```

### Task 4: Test Context Inheritance

Verify spawned tasks inherit context:
```rust
#[tokio::test]
async fn test_context_inherited() {
    let ctx = NarrationContext::new().with_job_id("job-123");
    
    with_narration_context(ctx, async {
        tokio::spawn(async {
            NARRATE.action("spawned").emit();
            // Should have job_id!
        }).await.unwrap();
    }).await;
}
```

---

## Verification Checklist

- [ ] All job routers wrapped with `with_narration_context()`
- [ ] All `.job_id()` calls removed
- [ ] All `job_id` parameters removed from handlers
- [ ] Context inherited in spawned tasks
- [ ] All tests pass
- [ ] SSE routing still works

---

## Handoff to TEAM-300

Document in `.plan/TEAM_299_HANDOFF.md`:
1. Files changed
2. Number of `.job_id()` calls removed
3. Any edge cases found
4. Recommendations for Phase 3
