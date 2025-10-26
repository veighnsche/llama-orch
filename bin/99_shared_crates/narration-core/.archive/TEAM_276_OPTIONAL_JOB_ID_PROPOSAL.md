# TEAM-276: Optional Job ID Pattern Improvement

**Status:** PROPOSED  
**Date:** Oct 23, 2025  
**Problem:** Repetitive boilerplate for optional job_id

## Problem

Throughout the codebase, we have this repetitive pattern:

```rust
let mut narration = NARRATE
    .action("daemon_start")
    .context("value")
    .human("Starting daemon");

if let Some(ref job_id) = config.job_id {
    narration = narration.job_id(job_id);
}

narration.emit();
```

This appears **hundreds of times** across:
- daemon-lifecycle (health.rs, shutdown.rs, install.rs, etc.)
- queen-lifecycle (all operations)
- hive-lifecycle (all operations)
- worker-lifecycle (all operations)

**Boilerplate count**: ~5-7 lines per narration with optional job_id

## Proposed Solution

Add a `.maybe_job_id()` method to Narration builder:

```rust
impl Narration {
    /// Set job_id if provided (handles Option<String>)
    ///
    /// TEAM-276: Reduces boilerplate for optional job_id pattern
    ///
    /// # Example
    /// ```rust
    /// // Before (7 lines):
    /// let mut narration = NARRATE.action("start").human("Starting");
    /// if let Some(ref job_id) = config.job_id {
    ///     narration = narration.job_id(job_id);
    /// }
    /// narration.emit();
    ///
    /// // After (3 lines):
    /// NARRATE.action("start")
    ///     .human("Starting")
    ///     .maybe_job_id(config.job_id.as_deref())
    ///     .emit();
    /// ```
    pub fn maybe_job_id(self, id: Option<&str>) -> Self {
        match id {
            Some(jid) => self.job_id(jid),
            None => self,
        }
    }
}
```

## Alternative: Builder Pattern Enhancement

Or even better, make all setters accept Option automatically:

```rust
impl Narration {
    /// Set job_id (accepts Option or &str)
    pub fn job_id(mut self, id: impl Into<Option<String>>) -> Self {
        if let Some(job_id) = id.into() {
            self.fields.job_id = Some(job_id);
        }
        self
    }
}
```

But this would break existing API... So `maybe_job_id` is safer.

## Impact

### Before (current pattern)
```rust
// daemon-lifecycle/src/health.rs (repeated 4 times)
let mut narration = NARRATE
    .action("daemon_health_poll")
    .context(&config.base_url)
    .human(format!("⏳ Waiting for {} to become healthy", daemon_name));
if let Some(ref job_id) = config.job_id {
    narration = narration.job_id(job_id);
}
narration.emit();
// 7 lines
```

### After (with `.maybe_job_id()`)
```rust
NARRATE
    .action("daemon_health_poll")
    .context(&config.base_url)
    .human(format!("⏳ Waiting for {} to become healthy", daemon_name))
    .maybe_job_id(config.job_id.as_deref())
    .emit();
// 5 lines (29% reduction)
```

### Code Reduction

**Per usage:**
- Before: 7 lines
- After: 5 lines
- **Savings: 2 lines (29%)**

**Across codebase:**
- Estimated uses: ~100+ locations
- **Total savings: ~200 lines**

## Implementation

### File: `src/builder.rs`

Add after the existing `.job_id()` method:

```rust
/// Set the job ID if provided (handles Option).
///
/// TEAM-276: Convenience method for optional job_id pattern
///
/// # Example
/// ```rust
/// use observability_narration_core::{Narration, NarrationFactory};
///
/// const NARRATE: NarrationFactory = NarrationFactory::new("dmn-life");
///
/// let job_id: Option<String> = Some("job-123".to_string());
///
/// NARRATE.action("start")
///     .human("Starting daemon")
///     .maybe_job_id(job_id.as_deref())
///     .emit();
/// ```
pub fn maybe_job_id(self, id: Option<&str>) -> Self {
    match id {
        Some(jid) => self.job_id(jid),
        None => self,
    }
}
```

### Alternative Names

1. `maybe_job_id()` - Clear intent ✅
2. `opt_job_id()` - Shorter, but less clear
3. `with_job_id()` - Conflicts with NarrationFactory method
4. `job_id_opt()` - Awkward ordering

**Recommendation: `maybe_job_id()` - Clear and idiomatic**

## Similar Methods?

We could add for all optional fields:

```rust
pub fn maybe_correlation_id(self, id: Option<&str>) -> Self;
pub fn maybe_session_id(self, id: Option<&str>) -> Self;
pub fn maybe_task_id(self, id: Option<&str>) -> Self;
pub fn maybe_error_kind(self, kind: Option<&str>) -> Self;
```

But job_id is by far the most common case (90%+ of optional field usage).

**Recommendation: Start with `maybe_job_id()` only**

## Testing

```rust
#[test]
fn test_maybe_job_id_with_some() {
    let adapter = CaptureAdapter::install();
    
    NARRATE.action("test")
        .human("Test message")
        .maybe_job_id(Some("job-123"))
        .emit();
    
    let captured = adapter.drain();
    assert_eq!(captured[0].job_id, Some("job-123".to_string()));
}

#[test]
fn test_maybe_job_id_with_none() {
    let adapter = CaptureAdapter::install();
    
    NARRATE.action("test")
        .human("Test message")
        .maybe_job_id(None)
        .emit();
    
    let captured = adapter.drain();
    assert_eq!(captured[0].job_id, None);
}
```

## Migration

### No Breaking Changes
- Existing code continues to work
- New method is optional
- Can migrate incrementally

### Migration Script (Optional)
```bash
# Find all instances of the pattern
rg -A 3 "let mut narration.*NARRATE" | grep "if let Some.*job_id"

# Each can be manually refactored to use .maybe_job_id()
```

## Documentation Update

Update builder.rs module docs to include example:

```rust
//! # Optional Job ID
//! ```rust
//! use observability_narration_core::{NarrationFactory, Narration};
//!
//! const NARRATE: NarrationFactory = NarrationFactory::new("my-actor");
//!
//! fn emit_with_optional_job_id(job_id: Option<&str>) {
//!     NARRATE.action("operation")
//!         .human("Performing operation")
//!         .maybe_job_id(job_id)
//!         .emit();
//! }
//! ```
```

## Decision

**Approve?**
- ✅ Reduces boilerplate significantly
- ✅ No breaking changes
- ✅ Clear, idiomatic naming
- ✅ Solves real pain point
- ✅ Small implementation (5 lines)

**Concerns?**
- ⚠️ Adds one more method to API surface
- ⚠️ Could encourage more optional fields (but that's not necessarily bad)

## Recommendation

**APPROVE**: Implement `.maybe_job_id()` in narration-core

This is a high-value, low-risk improvement that will clean up code across the entire codebase.
