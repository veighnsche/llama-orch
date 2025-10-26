# Narration V2: API Redesign - Macro-Based Approach

**Version:** 2.0.0-api  
**Date:** 2025-10-26  
**Status:** DESIGN PROPOSAL  
**Author:** TEAM-297

---

## Problem: Builder Pattern is Verbose

### Current Usage (Repetitive)

```rust
// Every narration requires this pattern:
NARRATE.action("hive_start")
    .job_id(&job_id)  // ← Repetitive
    .context(&alias)
    .human("Starting hive {}")
    .emit();  // ← Always needed

NARRATE.action("hive_check")
    .job_id(&job_id)  // ← Repetitive
    .context(&alias)
    .human("Checking hive {}")
    .emit();  // ← Always needed

NARRATE.action("hive_ready")
    .job_id(&job_id)  // ← Repetitive
    .context(&alias)
    .human("Hive {} is ready")
    .emit();  // ← Always needed
```

**Problems:**
1. `.emit()` is always required (100% of the time)
2. Chain calls are verbose
3. Easy to forget `.emit()`
4. Not DRY (every narration 4-5 lines)

---

## Solution 1: Macro with Format String (Recommended)

### The Vision

```rust
// Ultra-concise: action + format string + args
n!("hive_start", "Starting hive {}", alias);
n!("hive_check", "Checking hive {}", alias);
n!("hive_ready", "Hive {} is ready", alias);

// With explicit actor (when needed)
n!(actor: "custom", "deploy", "Deploying {}", name);

// With job_id (when context not set)
n!(job_id: &job_id, "process", "Processing {}", item);
```

### Implementation

```rust
// In narration-core/src/lib.rs

/// Narrate with format string (ultra-concise)
///
/// # Examples
/// ```
/// n!("hive_start", "Starting hive");
/// n!("deploy", "Deploying {}", name);
/// n!("process", "Processing {} items", count);
/// ```
#[macro_export]
macro_rules! n {
    // Basic: n!("action", "message")
    ($action:expr, $msg:expr) => {{
        $crate::Narration::from_context()
            .action($action)
            .human($msg)
            .emit();
    }};
    
    // With format args: n!("action", "message {}", arg)
    ($action:expr, $fmt:expr, $($arg:expr),+) => {{
        $crate::Narration::from_context()
            .action($action)
            .human(&format!($fmt, $($arg),+))
            .emit();
    }};
    
    // With actor: n!(actor: "name", "action", "message")
    (actor: $actor:expr, $action:expr, $msg:expr) => {{
        $crate::Narration::new($actor, $action, $action)
            .human($msg)
            .emit();
    }};
    
    // With actor + format: n!(actor: "name", "action", "msg {}", arg)
    (actor: $actor:expr, $action:expr, $fmt:expr, $($arg:expr),+) => {{
        $crate::Narration::new($actor, $action, $action)
            .human(&format!($fmt, $($arg),+))
            .emit();
    }};
    
    // With job_id: n!(job_id: &id, "action", "message")
    (job_id: $job_id:expr, $action:expr, $msg:expr) => {{
        $crate::Narration::from_context()
            .action($action)
            .job_id($job_id)
            .human($msg)
            .emit();
    }};
}

/// Longer alias for those who prefer clarity
#[macro_export]
macro_rules! narrate {
    ($($tt:tt)*) => { $crate::n!($($tt)*) };
}
```

### New Method: `from_context()`

```rust
impl Narration {
    /// Create narration from thread-local context
    ///
    /// Automatically uses actor from NARRATE factory if set,
    /// or defaults to "unknown".
    pub fn from_context() -> Self {
        // Try to get actor from thread-local NARRATE factory
        let actor = context::get_actor().unwrap_or("unknown");
        
        Self {
            fields: NarrationFields {
                actor,
                action: "",
                target: String::new(),
                human: String::new(),
                ..Default::default()
            },
            context_values: Vec::new(),
        }
    }
}
```

### Usage Examples

```rust
// Simple message
n!("startup", "Worker starting");

// With format string
n!("model_load", "Loading model from {}", path);

// Multiple args
n!("worker_ready", "Worker {} ready on port {}", id, port);

// With job_id (when context not set)
n!(job_id: &job_id, "process", "Processing item {}", item);

// Custom actor (rare)
n!(actor: "custom-svc", "deploy", "Deploying {}", name);
```

---

## Solution 2: Auto-Emit Builder (Alternative)

### Keep Builder, Remove `.emit()`

```rust
impl Drop for Narration {
    fn drop(&mut self) {
        // Auto-emit when dropped!
        if !self.fields.human.is_empty() {
            crate::narrate(self.fields.clone());
        }
    }
}

// Usage: No .emit() needed!
NARRATE.action("hive_start")
    .context(&alias)
    .human("Starting hive {}");  // ← Auto-emits when dropped!

// Explicit emit still works for clarity
NARRATE.action("hive_check")
    .context(&alias)
    .human("Checking hive {}")
    .emit();  // ← Optional, but explicit
```

**Pros:**
- Backward compatible (`.emit()` still works)
- Less code (no `.emit()` needed)
- Still uses builder pattern

**Cons:**
- Drop-based behavior can be surprising
- Harder to debug (when does it emit?)
- May emit too early if not careful

---

## Solution 3: Factory with Default Messages

### Bake Common Patterns into Factory

```rust
impl NarrationFactory {
    /// Narrate with action only (no custom message)
    ///
    /// Uses default message based on action name.
    pub fn emit(&self, action: &'static str) {
        self.action(action)
            .human(Self::default_message(action))
            .emit();
    }
    
    /// Narrate with format string
    pub fn fmt(&self, action: &'static str, msg: &str, args: &[&dyn ToString]) {
        let formatted = /* format with args */;
        self.action(action)
            .human(&formatted)
            .emit();
    }
    
    fn default_message(action: &str) -> String {
        match action {
            "startup" => "Starting up".to_string(),
            "ready" => "Ready".to_string(),
            "shutdown" => "Shutting down".to_string(),
            _ => format!("Action: {}", action),
        }
    }
}

// Usage:
NARRATE.emit("startup");  // → "Starting up"
NARRATE.fmt("deploy", "Deploying {}", &[&name]);
```

---

## Solution 4: Macro with Named Arguments (Most Flexible)

### Allow Named Arguments

```rust
#[macro_export]
macro_rules! n {
    // Named args: n!(action = "test", human = "message", job_id = &id)
    ($($key:ident = $val:expr),+ $(,)?) => {{
        let mut narration = $crate::Narration::from_context();
        $(
            narration = narration.$key($val);
        )+
        narration.emit();
    }};
}

// Usage:
n!(action = "deploy", human = "Deploying service");
n!(action = "deploy", human = "Deploying {}", context = &name);
n!(action = "process", job_id = &job_id, human = "Processing");
```

---

## Comparison

### Verbosity

```rust
// Current (Builder):
NARRATE.action("deploy")
    .context(&name)
    .human("Deploying {}")
    .emit();

// Solution 1 (Macro Format):
n!("deploy", "Deploying {}", name);

// Solution 2 (Auto-Emit):
NARRATE.action("deploy")
    .context(&name)
    .human("Deploying {}");  // No .emit()

// Solution 3 (Factory Defaults):
NARRATE.fmt("deploy", "Deploying {}", &[&name]);

// Solution 4 (Named Args):
n!(action = "deploy", human = "Deploying {}", context = name);
```

**Winner:** Solution 1 (Macro Format) - Shortest, most intuitive

### Type Safety

| Solution | Compile-Time Checks | Auto-Complete | Error Messages |
|----------|---------------------|---------------|----------------|
| Current Builder | ✅ Excellent | ✅ Full | ✅ Clear |
| Macro Format | ⚠️ Basic | ❌ None | ⚠️ Cryptic |
| Auto-Emit | ✅ Excellent | ✅ Full | ✅ Clear |
| Factory Defaults | ✅ Good | ✅ Partial | ✅ Clear |
| Named Args Macro | ⚠️ Basic | ❌ None | ⚠️ Cryptic |

**Winner:** Current Builder or Auto-Emit (best IDE support)

### Flexibility

| Solution | Custom Fields | Conditional Logic | Builder Chain |
|----------|---------------|-------------------|---------------|
| Current Builder | ✅ Full | ✅ Easy | ✅ Yes |
| Macro Format | ⚠️ Limited | ❌ Hard | ❌ No |
| Auto-Emit | ✅ Full | ✅ Easy | ✅ Yes |
| Factory Defaults | ⚠️ Limited | ❌ Hard | ❌ No |
| Named Args Macro | ✅ Full | ⚠️ Verbose | ❌ No |

**Winner:** Current Builder or Auto-Emit (most flexible)

---

## Recommendation: Hybrid Approach

### Best of Both Worlds

**Keep builder for complex cases:**
```rust
// Complex: use builder
NARRATE.action("deploy")
    .context(&name)
    .context(&version)
    .human("Deploying {} version {}")
    .correlation_id(&corr_id)
    .duration_ms(elapsed)
    .emit();
```

**Add macro for simple cases:**
```rust
// Simple: use macro
n!("startup", "Starting");
n!("ready", "Ready on port {}", port);
n!("shutdown", "Shutting down");
```

### Implementation Plan

#### Step 1: Add `n!()` Macro
```rust
#[macro_export]
macro_rules! n {
    ($action:expr, $msg:expr) => {{
        $crate::NARRATE_CONTEXT
            .with(|factory| {
                factory.action($action)
                    .human($msg)
                    .emit();
            });
    }};
    
    ($action:expr, $fmt:expr, $($arg:expr),+) => {{
        $crate::NARRATE_CONTEXT
            .with(|factory| {
                factory.action($action)
                    .human(&format!($fmt, $($arg),+))
                    .emit();
            });
    }};
}
```

#### Step 2: Add Thread-Local Factory
```rust
thread_local! {
    static NARRATE_CONTEXT: NarrationFactory = NarrationFactory::new("unknown");
}

/// Set the default narration factory for this thread
pub fn set_narration_factory(factory: NarrationFactory) {
    NARRATE_CONTEXT.with(|f| *f = factory);
}
```

#### Step 3: Usage Pattern
```rust
// In each module:
const NARRATE: NarrationFactory = NarrationFactory::new("hive");

// Set thread-local (once per thread)
set_narration_factory(NARRATE);

// Then use macro anywhere:
n!("startup", "Starting hive");  // Uses NARRATE from thread-local
n!("ready", "Hive ready on port {}", port);

// Or use builder for complex cases:
NARRATE.action("deploy")
    .context(&name)
    .human("Deploying {}")
    .correlation_id(&corr_id)
    .emit();
```

---

## Migration Strategy

### Phase 1: Add Macro (Parallel System)
- Add `n!()` macro
- Keep existing builder pattern
- No breaking changes
- Gradually adopt in new code

### Phase 2: Document Patterns
- Simple narration: use `n!()`
- Complex narration: use builder
- Update examples

### Phase 3: Migrate Hot Paths
- Migrate repetitive narration to `n!()`
- Keep complex narration on builder
- Measure LOC savings

### Expected Savings
```rust
// Before (5 lines):
NARRATE.action("startup")
    .context(&port)
    .human("Starting on port {}")
    .emit();

// After (1 line):
n!("startup", "Starting on port {}", port);

// 80% reduction for simple cases!
```

---

## Conclusion

**Recommended:** Hybrid approach with `n!()` macro for simple cases

**Rationale:**
1. ✅ Keeps builder for complex cases (flexibility)
2. ✅ Adds macro for simple cases (conciseness)
3. ✅ Non-breaking (parallel systems)
4. ✅ 80% reduction for common patterns
5. ✅ Still has IDE support for builder
6. ✅ Gradual migration path

**Next Step:** Prototype `n!()` macro and test ergonomics
