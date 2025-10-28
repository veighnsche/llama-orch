# TEAM-335: Stack Overflow - All Possible Sources

**Error:** `thread 'tokio-runtime-worker' has overflowed its stack`

**Confirmed Culprit:** `#[with_job_id]` macro

## Possible Sources of Stack Overflow

### 1. Proc Macro Expansion Depth
**Location:** `bin/99_shared_crates/narration-macros/src/lib.rs`

The `#[with_job_id]` macro generates code that wraps the function. If the macro:
- Creates nested async blocks
- Generates recursive type definitions
- Creates deep closure chains
- Uses complex pattern matching in generated code

**Check:**
```bash
cargo expand daemon_lifecycle::start::start_daemon
```

### 2. Tokio Task Local Storage
**Location:** `bin/99_shared_crates/narration-core/src/context.rs`

The macro likely uses `tokio::task_local!` for context propagation. Issues:
- Deep nesting of task-local contexts
- Recursive context inheritance
- Large context structures on the stack

**Check:** Look for `task_local!` usage in narration-core

### 3. Async Function Wrapping
**Location:** Generated code from `#[with_job_id]`

The macro wraps async functions, which can cause:
- Double boxing of futures
- Nested `Pin<Box<dyn Future>>`
- Large future state machines
- Recursive async wrapper chains

**Check:** Look at macro implementation for `async move` blocks

### 4. Type Recursion in Generated Code
**Location:** Macro-generated code

If the macro generates types that reference themselves:
- Recursive struct definitions
- Circular type dependencies
- Infinite type expansion during monomorphization

**Check:** Compiler error messages during expansion

### 5. Large Stack Frames
**Location:** Functions with `#[with_job_id]`

The macro might:
- Capture large amounts of data in closures
- Create large temporary structures
- Duplicate function parameters in wrapper

**Check:** Size of generated wrapper function

### 6. Tokio Runtime Stack Size
**Location:** `bin/00_rbee_keeper/src/main.rs`

The tokio runtime might be configured with:
- Too small stack size for worker threads
- Default stack size insufficient for macro-generated code

**Check:** Tokio runtime builder configuration

### 7. Recursive Macro Application
**Location:** Multiple files using `#[with_job_id]`

If functions with `#[with_job_id]` call other functions with `#[with_job_id]`:
- Nested context wrappers
- Stacked async transformations
- Compounding overhead

**Check:** Call graph of daemon-lifecycle functions

### 8. Interaction with `#[with_timeout]`
**Location:** Functions with both macros

When both macros are applied:
- Double wrapping of async functions
- Nested future transformations
- Combined overhead exceeds stack limit

**Check:** Test with only one macro at a time (already done - with_job_id is the issue)

### 9. Specta Type Generation (Tauri)
**Location:** `bin/00_rbee_keeper/src/tauri_commands.rs`

The Tauri `#[specta::specta]` macro generates TypeScript types. When combined with daemon-lifecycle types:
- Deep type recursion during codegen
- Large type representations
- Specta + with_job_id interaction

**Check:** Remove `#[specta::specta]` temporarily

### 10. Serde Serialization in Configs
**Location:** Config structs in daemon-lifecycle

Config structs with `#[derive(Serialize, Deserialize)]`:
- Large derived implementations
- Nested struct serialization
- Combined with macro-generated code

**Check:** Size of generated serde code

## Investigation Priority

### High Priority (Most Likely)
1. **Proc macro expansion depth** - Check actual generated code
2. **Tokio task_local! usage** - Context propagation mechanism
3. **Async function wrapping** - How the macro transforms functions

### Medium Priority
4. **Recursive macro application** - Call chains
5. **Specta interaction** - Tauri type generation
6. **Tokio runtime config** - Stack size settings

### Low Priority (Less Likely)
7. **Type recursion** - Would cause compile error usually
8. **Large stack frames** - Would need profiling
9. **Serde overhead** - Usually not a stack issue
10. **Timeout interaction** - Already ruled out

## Next Steps

1. **Examine macro implementation:**
   ```bash
   cat bin/99_shared_crates/narration-macros/src/lib.rs
   ```

2. **Check expanded code:**
   ```bash
   cargo expand -p daemon-lifecycle start::start_daemon > expanded.rs
   ```

3. **Look for task_local! usage:**
   ```bash
   grep -r "task_local!" bin/99_shared_crates/narration-core/
   ```

4. **Check tokio runtime config:**
   ```bash
   grep -r "Builder::new" bin/00_rbee_keeper/src/main.rs
   ```

5. **Test without Specta:**
   - Comment out `#[specta::specta]` in tauri_commands.rs
   - See if CLI still crashes

## Files to Examine

1. `bin/99_shared_crates/narration-macros/src/lib.rs` - Macro implementation
2. `bin/99_shared_crates/narration-core/src/context.rs` - Context mechanism
3. `bin/00_rbee_keeper/src/main.rs` - Tokio runtime setup
4. `bin/00_rbee_keeper/src/tauri_commands.rs` - Tauri integration
5. Generated code via `cargo expand`

## Expected Root Cause

Most likely: The `#[with_job_id]` macro creates nested async wrappers that, when combined with:
- Tokio's task-local storage
- Multiple layers of async transformation
- The specific call stack in rbee-keeper

...exceed the default stack size for tokio worker threads.

The fix will likely involve:
- Simplifying the macro implementation
- Reducing async wrapper nesting
- Using explicit parameters instead of task-local storage
- OR increasing tokio worker thread stack size (workaround, not fix)
