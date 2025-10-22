# Task-Local Context Doesn't Solve the Repetition Problem

**Status:** ❌ FAILED APPROACH

## The Problem We Tried to Solve

Repeating `.job_id(&job_id)` and `.correlation_id(&corr_id)` on every narration:

```rust
NARRATE.action("hive_start").job_id(&job_id).correlation_id(&corr_id).emit();
NARRATE.action("hive_check").job_id(&job_id).correlation_id(&corr_id).emit();
NARRATE.action("hive_spawn").job_id(&job_id).correlation_id(&corr_id).emit();
```

## What We Tried

Task-local context using `tokio::task_local!`:

```rust
with_narration_context(ctx, async move {
    NARRATE.action("hive_start").emit();  // Should auto-include job_id
    NARRATE.action("hive_check").emit();  // Should auto-include job_id
}).await;
```

## Why It Doesn't Work

**The narrations execute in the SAME task** as `with_narration_context`, so the task-local storage is accessible. BUT:

1. The narrations ARE being emitted (visible in queen's stderr)
2. They're NOT being received by keeper via SSE
3. This means `job_id` is NOT being included
4. SSE sink drops events without `job_id`

**Root cause:** The task-local context lookup in `emit()` is failing to find the context!

## The ONLY Solution

**Keep `.job_id(&job_id)` - it's required and can't be eliminated.**

The repetition is unavoidable because:
- `job_id` is LOCAL to each service (needed for SSE routing)
- `correlation_id` is GLOBAL (flows end-to-end)
- Both serve different purposes
- SSE sink REQUIRES `job_id` or events are dropped

## What We Should Do Instead

1. **Accept the repetition** - it's 2 method calls, not that bad
2. **Document WHY it's needed** - SSE routing security
3. **Maybe add a helper** (but it's still verbose):

```rust
// Helper macro (still not great)
macro_rules! narrate_job {
    ($job_id:expr, $corr_id:expr, $action:expr) => {
        NARRATE.action($action).job_id($job_id).correlation_id($corr_id)
    };
}

narrate_job!(&job_id, &corr_id, "hive_start").human("Starting").emit();
```

## Lessons Learned

- Task-local storage looked promising but doesn't solve it
- The architecture REQUIRES job_id for security (SSE isolation)
- Sometimes repetition is the clearest solution
- Don't over-engineer to avoid 2 method calls

## Good Night! 😴

The current approach (keeping `.job_id(&job_id)`) is correct. Don't change it.
