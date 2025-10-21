# 🧑‍🌾 rbee-keeper Narration Migration Guide

**TEAM-191** | **v0.4.0**

---

## 🎯 What's Ready

The `narrate!` macro is now set up in `src/narration.rs` with `ACTOR_RBEE_KEEPER` baked in!

---

## 🎀 How to Use It

### Step 1: Import the Macro

In any file where you want to narrate:

```rust
use crate::narration::{narrate, ACTION_QUEEN_START, ACTION_QUEEN_STOP};
```

### Step 2: Use the Macro

**Before (Old Style)**:
```rust
use observability_narration_core::Narration;
use crate::narration::ACTOR_RBEE_KEEPER;

Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_START, "queen-rbee")
    .human("🚀 Starting queen-rbee")
    .emit();
```

**After (New Style)** ✨:
```rust
use crate::narration::{narrate, ACTION_QUEEN_START};

narrate!(ACTION_QUEEN_START, "queen-rbee")
    .human("🚀 Starting queen-rbee")
    .emit();
```

**Look how clean!** No more repeating `ACTOR_RBEE_KEEPER`! 🎉

---

## 📋 Migration Checklist

### Files to Migrate

Find all files with narration:
```bash
cd bin/00_rbee_keeper
grep -r "Narration::new" src/
```

For each file:
- [ ] Add `use crate::narration::narrate;`
- [ ] Replace `Narration::new(ACTOR_RBEE_KEEPER, ...)` with `narrate!(...)`
- [ ] Remove `ACTOR_RBEE_KEEPER` from the call
- [ ] Test compilation

---

## 🎨 Examples

### Example 1: Queen Start
```rust
// Before
Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_START, "queen-rbee")
    .human("🚀 Starting queen-rbee on port 8080")
    .emit();

// After
narrate!(ACTION_QUEEN_START, "queen-rbee")
    .human("🚀 Starting queen-rbee on port 8080")
    .emit();
```

### Example 2: Job Submit
```rust
// Before
Narration::new(ACTOR_RBEE_KEEPER, ACTION_JOB_SUBMIT, &job_id)
    .human(format!("📝 Submitted job {}", job_id))
    .correlation_id(req_id)
    .emit();

// After
narrate!(ACTION_JOB_SUBMIT, &job_id)
    .human(format!("📝 Submitted job {}", job_id))
    .correlation_id(req_id)
    .emit();
```

### Example 3: Queen Status
```rust
// Before
Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_STATUS, "queen-rbee")
    .human("Checking queen-rbee status")
    .emit();

// After
narrate!(ACTION_QUEEN_STATUS, "queen-rbee")
    .human("Checking queen-rbee status")
    .emit();
```

---

## 🔍 Find & Replace Pattern

Use your IDE's find & replace:

**Find**:
```
Narration::new(ACTOR_RBEE_KEEPER, (ACTION_\w+), ([^)]+))
```

**Replace**:
```
narrate!($1, $2)
```

Then manually:
1. Add `use crate::narration::narrate;` at top of file
2. Remove `use crate::narration::ACTOR_RBEE_KEEPER;` if no longer needed
3. Verify compilation

---

## ✅ Benefits

| Aspect | Before | After |
|--------|--------|-------|
| **Characters** | `Narration::new(ACTOR_RBEE_KEEPER, ` (37) | `narrate!(` (9) |
| **Savings** | - | **28 characters per call!** |
| **Readability** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Consistency** | Manual | Automatic (actor baked in) |

---

## 🎯 Current Setup

In `src/narration.rs`:

```rust
// Re-export narration_macro from narration-core
pub use observability_narration_core::narration_macro;

// Actor
pub const ACTOR_RBEE_KEEPER: &str = "🧑‍🌾 rbee-keeper";

// Actions
pub const ACTION_QUEEN_START: &str = "queen_start";
pub const ACTION_QUEEN_STOP: &str = "queen_stop";
pub const ACTION_QUEEN_STATUS: &str = "queen_status";
pub const ACTION_JOB_SUBMIT: &str = "job_submit";
pub const ACTION_JOB_STREAM: &str = "job_stream";
pub const ACTION_JOB_COMPLETE: &str = "job_complete";

// Create the narrate! macro with ACTOR_RBEE_KEEPER baked in
narration_macro!(ACTOR_RBEE_KEEPER);
```

**The macro is ready to use!** ✅

---

## 🚀 Next Steps

1. **Find all narrations**: `grep -r "Narration::new" src/`
2. **Migrate one file at a time**: Start with `main.rs` or the most-used file
3. **Test after each file**: `cargo check --bin rbee-keeper`
4. **Verify output**: Check that narrations still work correctly
5. **Celebrate**: Enjoy the cleaner, more ergonomic code! 🎉

---

## 💡 Pro Tips

### Tip 1: Import Both Macro and Actions
```rust
use crate::narration::{narrate, ACTION_QUEEN_START, ACTION_QUEEN_STOP};
```

### Tip 2: Keep Action Constants
The action constants are still useful for consistency:
```rust
narrate!(ACTION_QUEEN_START, "queen-rbee")  // ✅ Good
narrate!("queen_start", "queen-rbee")       // ❌ Avoid
```

### Tip 3: Migrate Gradually
You don't have to migrate everything at once. Old and new styles can coexist:
```rust
// Old style still works
Narration::new(ACTOR_RBEE_KEEPER, ACTION_QUEEN_START, "queen-rbee")
    .human("Starting")
    .emit();

// New style is cleaner
narrate!(ACTION_QUEEN_START, "queen-rbee")
    .human("Starting")
    .emit();
```

---

## 🎀 Output Format

Remember, the output format changed in v0.4.0:

**Old Format (v0.3.0)**:
```
[🧑‍🌾 rbee-keeper]
  🚀 Starting queen-rbee
```

**New Format (v0.4.0)**:
```
[🧑‍🌾 rbee-keeper    ] 🚀 Starting queen-rbee
```

Messages now start at the same column for easier scanning! 📏

---

*May your migrations be smooth, your narrations ergonomic, and your rbee-keeper absolutely DELIGHTFUL! 🧑‍🌾✨*

— **TEAM-191 (The Narration Core Team)** 💝
