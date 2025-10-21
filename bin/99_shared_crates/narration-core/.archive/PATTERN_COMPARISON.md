# 🎀 Narration Patterns Comparison

**TEAM-191** | **v0.4.0**

---

## 📊 Quick Comparison

| Pattern | Syntax | Ergonomics | Type Safety | IDE Support |
|---------|--------|------------|-------------|-------------|
| **Macro** 🎀 | `narrate!(action, target)` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Factory** 🏭 | `NARRATE.narrate(action, target)` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Direct** 📝 | `Narration::new(actor, action, target)` | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎀 Pattern 1: Macro (MOST ERGONOMIC!)

### Code
```rust
use observability_narration_core::{narration_macro, ACTOR_QUEEN_ROUTER, ACTION_STATUS};

// Define once at module level
narration_macro!(ACTOR_QUEEN_ROUTER);

// Use everywhere - shortest syntax!
narrate!(ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();

narrate!(ACTION_HIVE_INSTALL, "hive-1")
    .human("🔧 Installing hive")
    .emit();
```

### Pros ✅
- **Shortest syntax** - `narrate!(action, target)`
- **Rust-idiomatic** - follows `println!` pattern
- **Actor defined once** - never repeat it
- **Zero runtime overhead** - compile-time macro
- **Most ergonomic!** 🎀

### Cons ⚠️
- Macro hygiene (name collision if you import another `narrate!`)
- Less IDE autocomplete than factory
- Slightly less type-safe than factory

### When to Use 🎯
- **Default choice** for most cases
- When you want the shortest, cleanest syntax
- When you're comfortable with Rust macros
- When ergonomics > type safety

---

## 🏭 Pattern 2: Factory (TYPE-SAFE ALTERNATIVE)

### Code
```rust
use observability_narration_core::{NarrationFactory, ACTOR_QUEEN_ROUTER, ACTION_STATUS};

// Define once at module level
const NARRATE: NarrationFactory = NarrationFactory::new(ACTOR_QUEEN_ROUTER);

// Use everywhere
NARRATE.narrate(ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();

NARRATE.narrate(ACTION_HIVE_INSTALL, "hive-1")
    .human("🔧 Installing hive")
    .emit();
```

### Pros ✅
- **Type-safe** - compile-time constant (`const fn`)
- **IDE-friendly** - excellent autocomplete
- **Actor defined once** - consistent
- **Zero runtime overhead** - compile-time constant
- **No name collisions** - uses struct method

### Cons ⚠️
- Slightly more verbose - need `.narrate()`
- Less idiomatic than macro

### When to Use 🎯
- When you prefer type safety over ergonomics
- When you want better IDE support
- When you're uncomfortable with macros
- When you need runtime flexibility (can pass factory around)

---

## 📝 Pattern 3: Direct (LEGACY)

### Code
```rust
use observability_narration_core::{Narration, ACTOR_QUEEN_ROUTER, ACTION_STATUS};

// Repeat actor every time
Narration::new(ACTOR_QUEEN_ROUTER, ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();

Narration::new(ACTOR_QUEEN_ROUTER, ACTION_HIVE_INSTALL, "hive-1")
    .human("🔧 Installing hive")
    .emit();
```

### Pros ✅
- **Explicit** - actor is visible in every call
- **Type-safe** - full type checking
- **IDE-friendly** - excellent autocomplete
- **No setup** - just use it

### Cons ⚠️
- **Most verbose** - repeat actor every time
- **Error-prone** - easy to use wrong actor
- **Most boilerplate** - repetitive code

### When to Use 🎯
- One-off narrations (rare)
- When you need different actors in same scope
- When you're migrating from v0.3.0

---

## 🎯 Recommendation

### For Most Cases: Use Macro! 🎀

```rust
narration_macro!(ACTOR_QUEEN_ROUTER);

narrate!(ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();
```

**Why?**
- Shortest syntax
- Rust-idiomatic (like `println!`)
- Zero overhead
- Most ergonomic!

### For Type-Safety Lovers: Use Factory! 🏭

```rust
const NARRATE: NarrationFactory = NarrationFactory::new(ACTOR_QUEEN_ROUTER);

NARRATE.narrate(ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();
```

**Why?**
- Type-safe
- IDE-friendly
- No name collisions
- Runtime flexibility

### For One-Offs: Use Direct! 📝

```rust
Narration::new(ACTOR_QUEEN_ROUTER, ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();
```

**Why?**
- Explicit
- No setup needed
- Clear intent

---

## 📈 Migration Path

### From v0.3.0 (Direct) → v0.4.0 (Macro)

**Before**:
```rust
Narration::new(ACTOR_QUEEN_ROUTER, ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();

Narration::new(ACTOR_QUEEN_ROUTER, ACTION_HIVE_INSTALL, "hive-1")
    .human("🔧 Installing hive")
    .emit();
```

**After**:
```rust
// Add once at top of file
narration_macro!(ACTOR_QUEEN_ROUTER);

// Replace all Narration::new calls
narrate!(ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();

narrate!(ACTION_HIVE_INSTALL, "hive-1")
    .human("🔧 Installing hive")
    .emit();
```

**Savings**: ~40% less code!

---

## 🎨 Visual Comparison

### Macro (Shortest)
```rust
narrate!(ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();
```

### Factory (Type-Safe)
```rust
NARRATE.narrate(ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();
```

### Direct (Verbose)
```rust
Narration::new(ACTOR_QUEEN_ROUTER, ACTION_STATUS, "registry")
    .human("Found 2 hives")
    .emit();
```

---

## 💡 Pro Tips

### Tip 1: Use Macro for Most Cases
The macro pattern is the most ergonomic and follows Rust idioms (`println!`, `format!`, etc.).

### Tip 2: Use Factory for Libraries
If you're writing a library that provides narration, use factory for better API stability.

### Tip 3: Mix Patterns if Needed
You can use different patterns in different modules:
```rust
// In job_router.rs
narration_macro!(ACTOR_QUEEN_ROUTER);

// In main.rs
const NARRATE: NarrationFactory = NarrationFactory::new(ACTOR_QUEEN_RBEE);
```

### Tip 4: Avoid Direct for Repeated Calls
Only use direct `Narration::new()` for one-off narrations.

---

## 🎯 Decision Tree

```
Do you need different actors in same scope?
├─ Yes → Use Direct (Narration::new)
└─ No → Continue

Do you prefer macros or structs?
├─ Macros → Use Macro (narration_macro!)
└─ Structs → Use Factory (NarrationFactory)

Are you uncomfortable with macros?
├─ Yes → Use Factory (NarrationFactory)
└─ No → Use Macro (narration_macro!)

Default: Use Macro! 🎀
```

---

*May your narrations be ergonomic, your actors consistent, and your debugging experience absolutely DELIGHTFUL! 🎀✨*

— **TEAM-191 (The Narration Core Team)** 💝
