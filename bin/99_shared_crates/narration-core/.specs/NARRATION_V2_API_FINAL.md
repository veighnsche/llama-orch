# Narration V2: Final API Design

**Version:** 2.0.0-final  
**Date:** 2025-10-26  
**Status:** READY FOR IMPLEMENTATION  
**Author:** TEAM-297

---

## Problems with Current API

### Problem 1: Reinventing `format!()`

```rust
// Current: Custom {0}, {1} replacement
NARRATE.action("start")
    .context("http://localhost:8080")  // {0}
    .context("8080")                   // {1}
    .human("Queen started on {0}, port {1}")
    .emit();

// Why not just use Rust's format!() ?
NARRATE.action("start")
    .human(format!("Queen started on {}, port {}", url, port))
    .emit();
```

**Issue:** Custom string replacement is:
- More code to maintain
- Slower than `format!()`
- Confusing (why `{0}` instead of just `{}`?)
- Doesn't support `format!()` features (width, precision, etc.)

### Problem 2: Three Narration Types, But Only One Used

```rust
// Three types defined:
pub struct NarrationFields {
    pub human: String,   // ✅ Used everywhere
    pub cute: Option<String>,   // ❌ Never used (0 occurrences)
    pub story: Option<String>,  // ❌ Never used (0 occurrences)
}

// But only human is displayed!
eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, fields.human);
//                                                              ^^^ Always human
```

**Issue:** We have the infrastructure but:
- Can't switch between narration modes
- No configuration system
- Wasted fields in every struct

### Problem 3: Verbose Builder Pattern

```rust
// Takes 4-5 lines for simple narration
NARRATE.action("startup")
    .context(&port)
    .human("Starting on port {}")
    .emit();
```

---

## Solution: Clean Macro-Based API

### Core Design Principles

1. **Use Rust's `format!()`** - Don't reinvent the wheel
2. **Support all 3 narration modes** - Make it configurable
3. **Default to human for now** - Can switch later
4. **Ultra-concise macros** - 90% of cases = 1 line
5. **Keep builder for complex** - 10% of cases = full control

---

## New API Design

### The Macro: `n!()`

```rust
/// Ultra-concise narration macro
///
/// # Basic usage:
/// n!("action", "message");
/// n!("action", "message with {}", var);
/// n!("action", "multiple {} vars {}", var1, var2);
///
/// # With narration type:
/// n!(human: "action", "message");
/// n!(cute: "action", "🐝 Cutesy message");
/// n!(story: "action", "The system said 'Hello'");
///
/// # All three at once:
/// n!("deploy",
///     human: "Deploying service {}",
///     cute: "🚀 Launching {} into the cloud!",
///     story: "The orchestrator whispered to the service: 'It's time to fly, {}'",
///     service_name
/// );
#[macro_export]
macro_rules! n {
    // Simple: n!("action", "message")
    ($action:expr, $msg:expr) => {{
        $crate::emit_narration($action, $msg, None, None);
    }};
    
    // With format: n!("action", "msg {}", arg)
    ($action:expr, $fmt:expr, $($arg:expr),+) => {{
        $crate::emit_narration($action, &format!($fmt, $($arg),+), None, None);
    }};
    
    // Explicit human: n!(human: "action", "msg {}", arg)
    (human: $action:expr, $fmt:expr $(, $arg:expr)*) => {{
        $crate::emit_narration($action, &format!($fmt $(, $arg)*), None, None);
    }};
    
    // Explicit cute: n!(cute: "action", "msg {}", arg)
    (cute: $action:expr, $fmt:expr $(, $arg:expr)*) => {{
        $crate::emit_narration($action, "", Some(&format!($fmt $(, $arg)*)), None);
    }};
    
    // Explicit story: n!(story: "action", "msg {}", arg)
    (story: $action:expr, $fmt:expr $(, $arg:expr)*) => {{
        $crate::emit_narration($action, "", None, Some(&format!($fmt $(, $arg)*)));
    }};
    
    // All three: n!("action", human: "msg", cute: "msg", story: "msg", args...)
    ($action:expr,
     human: $human_fmt:expr,
     cute: $cute_fmt:expr,
     story: $story_fmt:expr
     $(, $arg:expr)*
    ) => {{
        $crate::emit_narration(
            $action,
            &format!($human_fmt $(, $arg)*),
            Some(&format!($cute_fmt $(, $arg)*)),
            Some(&format!($story_fmt $(, $arg)*))
        );
    }};
}
```

### The Emitter Function

```rust
/// Emit narration with mode selection
///
/// This is called by the macro, not by users directly.
pub fn emit_narration(
    action: &'static str,
    human: &str,
    cute: Option<&str>,
    story: Option<&str>,
) {
    // Get narration mode from global config
    let mode = NARRATION_MODE.load(Ordering::Relaxed);
    
    // Select which message to actually display
    let message = match mode {
        NarrationMode::Human => human,
        NarrationMode::Cute => cute.unwrap_or(human),  // Fall back to human
        NarrationMode::Story => story.unwrap_or(human), // Fall back to human
    };
    
    // Get actor from thread-local context
    let actor = context::get_actor().unwrap_or("unknown");
    
    // Emit the narration
    let fields = NarrationFields {
        actor,
        action,
        target: action.to_string(),
        human: message.to_string(),
        cute: cute.map(|s| s.to_string()),
        story: story.map(|s| s.to_string()),
        ..Default::default()
    };
    
    narrate(fields);
}
```

### Narration Mode Configuration

```rust
use std::sync::atomic::{AtomicU8, Ordering};

/// Global narration mode
static NARRATION_MODE: AtomicU8 = AtomicU8::new(NarrationMode::Human as u8);

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

---

## Usage Examples

### Basic (90% of cases)

```rust
// Simple message
n!("startup", "Worker starting");

// With format string (uses Rust's format!())
n!("model_load", "Loading model from {}", path);

// Multiple args
n!("worker_ready", "Worker {} ready on port {}", id, port);

// Complex formatting (all format!() features work!)
n!("progress", "Downloaded {:.2} GB of {:.2} GB", current, total);
n!("status", "Status: {:?}", status_enum);
n!("hex", "Address: 0x{:08x}", addr);
```

### With Narration Type (Optional)

```rust
// Explicit human (same as default)
n!(human: "deploy", "Deploying service {}", name);

// Cute mode
n!(cute: "deploy", "🚀 Launching {} into the cloud!", name);

// Story mode
n!(story: "deploy", "The orchestrator whispered: 'Fly, {}'", name);
```

### All Three at Once (Rich Narration)

```rust
// Provide all 3 versions, let user choose via config
n!("worker_spawn",
    human: "Spawning worker {} on device {}",
    cute: "🐝 A new worker bee {} is being born on device {}!",
    story: "The hive gently nudged worker {} awake on device {}",
    worker_id,
    device
);

// At runtime, user can switch:
// set_narration_mode(NarrationMode::Cute);
// → Now they see: "🐝 A new worker bee w-123 is being born on device GPU-0!"
```

### Complex Cases (Use Builder)

```rust
// When you need full control, use builder
NARRATE.action("deploy")
    .human(&format!("Deploying {} version {}", name, ver))
    .cute(&format!("🚀 {} v{} is launching!", name, ver))
    .story(&format!("'Ready for launch,' said {} v{}", name, ver))
    .correlation_id(&corr_id)
    .duration_ms(elapsed)
    .emit();
```

---

## Implementation Details

### Remove `.context()` System

```rust
// DELETE: The entire .context() infrastructure
// - context_values: Vec<String>
// - pub fn context(mut self, value: impl Into<String>) -> Self
// - Custom {0}, {1} replacement logic

// WHY: Just use format!() directly in the macro!
```

### Simplify NarrationFields

```rust
pub struct NarrationFields {
    pub actor: &'static str,
    pub action: &'static str,
    pub target: String,
    
    // Three narration types
    pub human: String,         // Always present
    pub cute: Option<String>,  // Optional, for cute mode
    pub story: Option<String>, // Optional, for story mode
    
    // All other fields...
}
```

### Display Logic

```rust
pub fn narrate(fields: NarrationFields) {
    // Select which message to display based on mode
    let mode = get_narration_mode();
    let message = match mode {
        NarrationMode::Human => &fields.human,
        NarrationMode::Cute => fields.cute.as_ref().unwrap_or(&fields.human),
        NarrationMode::Story => fields.story.as_ref().unwrap_or(&fields.human),
    };
    
    // Always display in same format
    eprintln!("[{:<10}] {:<15}: {}", fields.actor, fields.action, message);
    
    // ... SSE, tracing, etc.
}
```

---

## Configuration System (Future)

### Environment Variable

```bash
# Set narration mode via env var
RBEE_NARRATION_MODE=cute cargo run

# Or in code:
if env::var("RBEE_NARRATION_MODE") == Ok("cute".to_string()) {
    set_narration_mode(NarrationMode::Cute);
}
```

### Config File

```toml
# .llorch.toml
[narration]
mode = "cute"  # "human", "cute", or "story"
```

### Runtime API

```rust
// Can be changed at runtime!
set_narration_mode(NarrationMode::Cute);

// All narration from now on uses cute mode
n!("deploy", "Deploying {}", name);
// → If cute version provided: "🚀 Launching service into the cloud!"
// → If not: Falls back to human: "Deploying service"
```

### Per-Module Override

```rust
// Set different modes for different modules
mod worker {
    use narration_core::*;
    
    fn init() {
        set_narration_mode(NarrationMode::Cute);
    }
    
    pub fn spawn() {
        n!("spawn", "Worker spawning");
        // → Uses cute mode for this module
    }
}
```

---

## Migration Strategy

### Phase 0: Add New API (Parallel)

1. Add `n!()` macro
2. Add `emit_narration()` function
3. Add `NarrationMode` enum and `set_narration_mode()`
4. Keep old API working

**Result:** Both systems coexist

### Phase 1: Migrate Simple Cases

```rust
// Before:
NARRATE.action("startup")
    .context(&port)
    .human("Starting on port {}")
    .emit();

// After:
n!("startup", "Starting on port {}", port);
```

**Result:** 80% of narration is now 1 line

### Phase 2: Add Rich Narration

```rust
// Add cute/story versions to key narration points
n!("worker_spawn",
    human: "Spawning worker {}",
    cute: "🐝 New worker bee {} is being born!",
    story: "The hive whispered life into worker {}",
    worker_id
);
```

**Result:** Users can switch modes and see different narration

### Phase 3: Remove `.context()` System

1. Delete `context_values: Vec<String>`
2. Delete `pub fn context()` method
3. Delete custom `{0}`, `{1}` replacement logic

**Result:** Simpler codebase, less code to maintain

---

## Comparison: Before vs After

### Verbosity

```rust
// BEFORE (5 lines):
NARRATE.action("deploy")
    .context(&name)
    .context(&version)
    .human("Deploying {} version {}")
    .emit();

// AFTER (1 line):
n!("deploy", "Deploying {} version {}", name, version);

// Savings: 80% fewer lines!
```

### Features

```rust
// BEFORE: Custom {0}, {1} replacement (limited)
.context("value1")
.context("value2")
.human("Message {0} and {1}")

// AFTER: Full Rust format!() support
n!("action", "Value: {:.2} (hex: 0x{:08x})", val, addr);
//           ^^^^ width, precision, hex, debug, etc. all work!
```

### Narration Modes

```rust
// BEFORE: Only human mode, cute/story unused
.human("Deploying service")
.cute("...")  // Never used!
.story("...")  // Never used!

// AFTER: All modes supported, configurable
n!("deploy",
    human: "Deploying service",
    cute: "🚀 Launching into cloud!",
    story: "'Time to fly,' said the orchestrator"
);
// User sets mode, sees appropriate version!
```

---

## Benefits Summary

### ✅ Simpler Code
- Remove `.context()` system (200+ LOC)
- Remove custom string replacement
- Use Rust's `format!()` directly

### ✅ Shorter Code
- 90% of narration: 1 line instead of 5
- 80% reduction in boilerplate

### ✅ More Features
- Full `format!()` support (width, precision, etc.)
- All 3 narration modes actually work
- Runtime configuration

### ✅ Better UX
- Users can choose narration mode
- Cute mode for fun! 🐝
- Story mode for narrative logs

### ✅ Easier Maintenance
- Less custom code
- Standard Rust patterns
- Fewer edge cases

---

## Open Questions

1. **Should cute/story be features or always available?**
   - Proposal: Always available, controlled by runtime config

2. **Should we require all 3 versions or allow fallback?**
   - Proposal: Always fall back to human if cute/story not provided

3. **Environment variable name?**
   - Proposal: `RBEE_NARRATION_MODE` (matches other RBEE_ vars)

4. **Per-request narration mode (for API)?**
   - Proposal: Later, add `X-Narration-Mode` header support

---

## Next Steps

1. **Implement `n!()` macro** with mode support
2. **Add `NarrationMode` configuration**
3. **Migrate 10 high-frequency narration points**
4. **Test all 3 modes work**
5. **Document migration guide**
6. **Remove `.context()` system** after migration

---

## Conclusion

The new API:
- ✅ Uses Rust's `format!()` (no reinvention)
- ✅ Supports all 3 narration modes
- ✅ Configurable at runtime
- ✅ 80% less code for common cases
- ✅ Backward compatible during migration

**Recommendation:** Implement this BEFORE Phase 1-4, as it simplifies everything!
