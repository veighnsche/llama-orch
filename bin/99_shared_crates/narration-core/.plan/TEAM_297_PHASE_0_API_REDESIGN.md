# TEAM-297: Phase 0 - API Redesign (Foundation)

**Status:** READY FOR IMPLEMENTATION  
**Estimated Duration:** 1 week  
**Dependencies:** None (Foundation phase)  
**Risk Level:** Low (parallel system, non-breaking)

---

## Mission

Create an ultra-concise macro-based narration API that:
1. Uses Rust's `format!()` (not custom string replacement)
2. Supports all 3 narration modes (human, cute, story)
3. Reduces boilerplate from 5 lines to 1 line
4. Works in parallel with existing builder API

---

## ⚠️ CRITICAL: DO YOUR RESEARCH FIRST!

**DO NOT START CODING UNTIL YOU COMPLETE THE RESEARCH PHASE!**

### Required Research (Complete ALL before coding)

#### 1. Read Current Implementation
- [ ] Read `bin/99_shared_crates/narration-core/src/builder.rs` (full file)
- [ ] Read `bin/99_shared_crates/narration-core/src/lib.rs` (narrate functions)
- [ ] Count all `NARRATE.action()` calls in codebase (should be 328)
- [ ] Analyze the `.context()` system (lines 102-155)
- [ ] Find all uses of `.cute()` and `.story()` (should be 0)

#### 2. Understand Current Problems
Document:
- [ ] Why `.context()` with `{0}`, `{1}` is problematic
- [ ] Why cute/story modes are unused
- [ ] Average lines per narration call (should be 4-5)
- [ ] How `.emit()` is always required

#### 3. Study Rust Macros
- [ ] Read Rust macro documentation
- [ ] Understand `macro_rules!` patterns
- [ ] Learn about `format_args!()` integration
- [ ] Study thread-local storage patterns

#### 4. Analyze Usage Patterns
Find the most common narration patterns:
- [ ] Simple message (no variables)
- [ ] Single variable substitution
- [ ] Multiple variables
- [ ] With job_id
- [ ] With error context

#### 5. Create Research Summary
Write document (`.plan/TEAM_297_RESEARCH_SUMMARY.md`) with:
- Current usage analysis (with examples)
- Proposed macro syntax (with examples)
- Migration strategy
- Backward compatibility plan
- Performance considerations

**ONLY AFTER COMPLETING RESEARCH: Proceed to implementation**

---

## Current Problems

### Problem 1: Verbose Builder

```rust
// Current: 328 occurrences of this pattern!
NARRATE.action("worker_spawn")
    .job_id(&job_id)
    .context(&worker_id)
    .context(&device)
    .human("Spawning worker {} on device {}")
    .emit();

// 5 lines, 4 method calls, manual job_id, .emit() always needed
```

### Problem 2: Reinventing format!()

```rust
// Current: Custom {0}, {1} replacement
.context("value1")  // {0}
.context("value2")  // {1}
.human("Message {0} and {1}")

// Problems:
// - Must call .context() for each value
// - {0}, {1} syntax instead of Rust's {}
// - Can't use format!() features (width, precision, etc.)
// - More code to maintain
// - Slower than format!()
```

### Problem 3: Unused Narration Modes

```rust
pub struct NarrationFields {
    pub human: String,         // ✅ Used in all 328 calls
    pub cute: Option<String>,  // ❌ 0 uses!
    pub story: Option<String>, // ❌ 0 uses!
}

// Infrastructure exists but:
// - No way to configure mode
// - No way to select which to display
// - cute/story are dead code
```

---

## Solution: Ultra-Concise Macro

### The Vision

```rust
// Simple message (1 line instead of 5)
n!("startup", "Worker starting");

// With variables (uses Rust's format!())
n!("worker_spawn", "Spawning worker {} on device {}", worker_id, device);

// With all 3 narration modes
n!("deploy",
    human: "Deploying service {}",
    cute: "🚀 Launching {} into the cloud!",
    story: "The orchestrator whispered to {}: 'Time to fly'",
    name
);
```

---

## Implementation Tasks

### Task 1: Add `n!()` Macro

**New File:** `bin/99_shared_crates/narration-core/src/macro.rs`

```rust
/// Ultra-concise narration macro
///
/// # Simple usage:
/// ```
/// n!("action", "message");
/// n!("action", "message {}", var);
/// n!("action", "msg {} and {}", var1, var2);
/// ```
///
/// # With narration type:
/// ```
/// n!(human: "action", "message");
/// n!(cute: "action", "🐝 Cute message");
/// n!(story: "action", "'Hello,' said the system");
/// ```
///
/// # All three modes:
/// ```
/// n!("action",
///     human: "Technical message {}",
///     cute: "🐝 Fun message {}",
///     story: "'Message,' said {}",
///     var
/// );
/// ```
#[macro_export]
macro_rules! n {
    // Simple: n!("action", "message")
    ($action:expr, $msg:expr) => {{
        $crate::macro_emit($action, $msg, None, None);
    }};
    
    // With format: n!("action", "msg {}", arg)
    ($action:expr, $fmt:expr, $($arg:expr),+ $(,)?) => {{
        $crate::macro_emit($action, &format!($fmt, $($arg),+), None, None);
    }};
    
    // Explicit human: n!(human: "action", "msg {}", arg)
    (human: $action:expr, $fmt:expr $(, $arg:expr)* $(,)?) => {{
        $crate::macro_emit($action, &format!($fmt $(, $arg)*), None, None);
    }};
    
    // Explicit cute: n!(cute: "action", "msg {}", arg)
    (cute: $action:expr, $fmt:expr $(, $arg:expr)* $(,)?) => {{
        $crate::macro_emit($action, "", Some(&format!($fmt $(, $arg)*)), None);
    }};
    
    // Explicit story: n!(story: "action", "msg {}", arg)
    (story: $action:expr, $fmt:expr $(, $arg:expr)* $(,)?) => {{
        $crate::macro_emit($action, "", None, Some(&format!($fmt $(, $arg)*)));
    }};
    
    // All three: n!("action", human: "msg", cute: "msg", story: "msg", args...)
    ($action:expr,
     human: $human_fmt:expr,
     cute: $cute_fmt:expr,
     story: $story_fmt:expr
     $(, $arg:expr)* $(,)?
    ) => {{
        $crate::macro_emit(
            $action,
            &format!($human_fmt $(, $arg)*),
            Some(&format!($cute_fmt $(, $arg)*)),
            Some(&format!($story_fmt $(, $arg)*))
        );
    }};
}

/// Long-form alias for those who prefer clarity
#[macro_export]
macro_rules! narrate {
    ($($tt:tt)*) => { $crate::n!($($tt)*) };
}
```

### Task 2: Add Emitter Function

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

```rust
/// Emit narration from macro (internal use)
///
/// This is called by n!() macro, not by users directly.
#[doc(hidden)]
pub fn macro_emit(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
) {
    // Get narration mode from global config
    let mode = get_narration_mode();
    
    // Select which message to display
    let message = match mode {
        NarrationMode::Human => human,
        NarrationMode::Cute => cute.unwrap_or(human),
        NarrationMode::Story => story.unwrap_or(human),
    };
    
    // Get actor from thread-local context (or default)
    let actor = context::get_actor().unwrap_or("unknown");
    
    // Get job_id from thread-local context (if set)
    let job_id = context::get_context().and_then(|ctx| ctx.job_id);
    
    // Build fields
    let fields = NarrationFields {
        actor,
        action,
        target: action.to_string(),
        human: message.to_string(),
        cute: cute.map(|s| s.to_string()),
        story: story.map(|s| s.to_string()),
        job_id,
        ..Default::default()
    };
    
    // Emit using existing narrate() function
    narrate(fields);
}
```

### Task 3: Add Narration Mode Configuration

**File:** `bin/99_shared_crates/narration-core/src/mode.rs`

```rust
use std::sync::atomic::{AtomicU8, Ordering};

/// Global narration mode (thread-safe)
static NARRATION_MODE: AtomicU8 = AtomicU8::new(NarrationMode::Human as u8);

/// Narration display mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NarrationMode {
    /// Standard technical narration (default)
    Human = 0,
    
    /// Cute/whimsical narration (🐝 friendly)
    Cute = 1,
    
    /// Story-mode dialogue narration
    Story = 2,
}

/// Set the global narration mode
///
/// This affects ALL narration from this point forward.
/// Thread-safe and instant.
///
/// # Example
/// ```
/// use observability_narration_core::{set_narration_mode, NarrationMode};
///
/// // Switch to cute mode
/// set_narration_mode(NarrationMode::Cute);
///
/// // All narration now shows cute version (or falls back to human)
/// ```
pub fn set_narration_mode(mode: NarrationMode) {
    NARRATION_MODE.store(mode as u8, Ordering::Relaxed);
}

/// Get the current narration mode
pub fn get_narration_mode() -> NarrationMode {
    match NARRATION_MODE.load(Ordering::Relaxed) {
        0 => NarrationMode::Human,
        1 => NarrationMode::Cute,
        2 => NarrationMode::Story,
        _ => NarrationMode::Human,
    }
}
```

### Task 4: Update Display Logic

**File:** `bin/99_shared_crates/narration-core/src/lib.rs`

```rust
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    let Some(tracing_level) = level.to_tracing_level() else {
        return; // MUTE
    };

    // TEAM-297: Select message based on current mode
    let mode = get_narration_mode();
    let message = match mode {
        NarrationMode::Human => &fields.human,
        NarrationMode::Cute => fields.cute.as_ref().unwrap_or(&fields.human),
        NarrationMode::Story => fields.story.as_ref().unwrap_or(&fields.human),
    };

    // TEAM-297: Always output to stderr (use selected message)
    eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);

    // ... rest of function (SSE, tracing, capture)
}
```

### Task 5: Remove `.context()` System

**File:** `bin/99_shared_crates/narration-core/src/builder.rs`

```rust
// DELETE: The entire .context() system
// - Remove: context_values: Vec<String> field
// - Remove: pub fn context() method
// - Remove: {0}, {1} replacement logic in .human()/.cute()/.story()
//
// WHY: We now use Rust's format!() directly in the macro!

impl Narration {
    pub fn human(mut self, msg: impl Into<String>) -> Self {
        // TEAM-297: SIMPLIFIED - No more {0}, {1} replacement
        self.fields.human = msg.into();
        self
    }
    
    #[cfg(feature = "cute-mode")]
    pub fn cute(mut self, msg: impl Into<String>) -> Self {
        // TEAM-297: SIMPLIFIED - No more {0}, {1} replacement
        self.fields.cute = Some(msg.into());
        self
    }
    
    pub fn story(mut self, msg: impl Into<String>) -> Self {
        // TEAM-297: SIMPLIFIED - No more {0}, {1} replacement
        self.fields.story = Some(msg.into());
        self
    }
}
```

### Task 6: Add Tests

**New File:** `bin/99_shared_crates/narration-core/tests/macro_tests.rs`

```rust
use observability_narration_core::*;

#[test]
fn test_simple_narration() {
    let adapter = CaptureAdapter::install();
    
    n!("test", "Simple message");
    
    let captured = adapter.captured();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].action, "test");
    assert_eq!(captured[0].human, "Simple message");
}

#[test]
fn test_narration_with_format() {
    let adapter = CaptureAdapter::install();
    
    n!("test", "Message with {} and {}", "arg1", "arg2");
    
    let captured = adapter.captured();
    assert_eq!(captured[0].human, "Message with arg1 and arg2");
}

#[test]
fn test_narration_modes() {
    let adapter = CaptureAdapter::install();
    
    n!("test",
        human: "Technical message",
        cute: "🐝 Fun message",
        story: "Story message"
    );
    
    let captured = adapter.captured();
    assert_eq!(captured[0].human, "Technical message");
    assert_eq!(captured[0].cute, Some("🐝 Fun message".to_string()));
    assert_eq!(captured[0].story, Some("Story message".to_string()));
}

#[test]
fn test_mode_selection() {
    let adapter = CaptureAdapter::install();
    
    // Default: human mode
    assert_eq!(get_narration_mode(), NarrationMode::Human);
    
    // Switch to cute
    set_narration_mode(NarrationMode::Cute);
    assert_eq!(get_narration_mode(), NarrationMode::Cute);
    
    // Switch to story
    set_narration_mode(NarrationMode::Story);
    assert_eq!(get_narration_mode(), NarrationMode::Story);
}

#[test]
fn test_fallback_to_human() {
    set_narration_mode(NarrationMode::Cute);
    let adapter = CaptureAdapter::install();
    
    // Only human provided, should fall back
    n!("test", "Only human message");
    
    let captured = adapter.captured();
    // Should use human message even in cute mode
    assert_eq!(captured[0].human, "Only human message");
}
```

---

## Verification Checklist

Before marking this phase complete:

- [ ] `n!()` macro compiles without errors
- [ ] Simple narration works: `n!("action", "message")`
- [ ] Format strings work: `n!("action", "msg {}", var)`
- [ ] Multiple args work: `n!("action", "{} and {}", v1, v2)`
- [ ] Narration modes work: `n!("action", human: "...", cute: "...", story: "...")`
- [ ] Mode selection works: `set_narration_mode(NarrationMode::Cute)`
- [ ] Fallback works: cute mode shows human if cute not provided
- [ ] `.context()` system removed
- [ ] All tests pass
- [ ] Existing builder API still works (backward compatible)

---

## Migration Examples

### Before → After

```rust
// BEFORE (5 lines):
NARRATE.action("startup")
    .context(&port)
    .context(&host)
    .human("Starting on {}:{}")
    .emit();

// AFTER (1 line):
n!("startup", "Starting on {}:{}", host, port);
```

```rust
// BEFORE (4 lines):
NARRATE.action("worker_ready")
    .job_id(&job_id)
    .context(&worker_id)
    .human("Worker {} is ready")
    .emit();

// AFTER (1 line):
n!("worker_ready", "Worker {} is ready", worker_id);
// Note: job_id comes from thread-local context (Phase 2)
```

```rust
// BEFORE (cute/story unusable):
NARRATE.action("deploy")
    .human("Deploying service")
    .emit();
// No way to add cute or story!

// AFTER (all 3 modes):
n!("deploy",
    human: "Deploying service {}",
    cute: "🚀 Launching {} to the stars!",
    story: "The system whispered: 'Fly, {}'",
    name
);
```

---

## Success Criteria

1. **Conciseness** - 1 line for 90% of cases
2. **Power** - Full `format!()` support (width, precision, debug, etc.)
3. **Flexibility** - All 3 narration modes work
4. **Compatibility** - Old builder API still works
5. **Performance** - No regression (< 5% slower)

---

## Handoff to TEAM-298

After completing Phase 0, create `.plan/TEAM_297_HANDOFF.md` with:

1. **What you implemented:**
   - n!() macro syntax and all variants
   - NarrationMode enum and configuration
   - .context() removal details
   
2. **Migration guide:**
   - 10 examples of before/after
   - Common patterns and pitfalls
   
3. **Performance notes:**
   - Benchmark results
   - Any slowdowns found
   
4. **Recommendations for Phase 1:**
   - How SSE optional will work with macro
   - Thread-local context integration points
   
5. **Known issues:**
   - Any edge cases
   - TODO items

---

## Notes

- This phase is **foundational** - everything else builds on this
- Must be **perfect** - 4 teams depend on it
- **Non-breaking** - old API continues working
- **Well-tested** - this is critical infrastructure
