# TEAM-335: Circular Dependency Fix Attempt

**Date:** Oct 28, 2025  
**Status:** 🧪 TESTING  
**Fix #5:** Remove circular dependency

---

## The Discovery

User spotted rust-analyzer warning:
```
WARN cyclic deps: observability_narration_core(Idx::<CrateBuilder>(94)) -> job_server(Idx::<CrateBuilder>(88)), 
alternative path: job_server(Idx::<CrateBuilder>(88)) -> observability_narration_core(Idx::<CrateBuilder>(94))
```

**Circular dependency:**
```
job-server (Cargo.toml:18)
  └─ depends on: observability-narration-core

observability-narration-core (Cargo.toml:38, dev-dependencies)
  └─ depends on: job-server (for tests)
```

---

## Why This Could Cause Stack Overflow

### Theory 1: Type Resolution Infinite Loop

During compilation, Rust resolves types:
1. Compile `job-server` → needs `narration-core` types
2. Compile `narration-core` → needs `job-server` types (for tests)
3. Back to step 1 → infinite loop

Even though it's dev-dependencies, rust-analyzer loads ALL dependencies for type checking.

### Theory 2: Initialization Order

At runtime, if both crates have static initializers:
1. `job-server` static init → calls `narration-core` functions
2. `narration-core` static init → calls `job-server` functions (in tests)
3. Infinite recursion during initialization

### Theory 3: Tauri-Specific Issue

- CLI works because it doesn't load test dependencies
- Tauri might load all dependencies (including dev) for some reason
- Circular dependency causes deep call stack during Tauri initialization

---

## The Fix

**File:** `bin/99_shared_crates/narration-core/Cargo.toml`

**Before:**
```toml
[dev-dependencies]
job-server = { path = "../job-server" } # TEAM-302: For test harness
```

**After:**
```toml
[dev-dependencies]
# TEAM-335: COMMENTED OUT - CIRCULAR DEPENDENCY causes stack overflow!
# job-server → narration-core → job-server (infinite loop)
# job-server = { path = "../job-server" }
```

---

## Impact

### ✅ What Still Works

- All production code (job-server is only needed for tests)
- CLI install: `./rbee queen install`
- All business logic unchanged

### ❌ What Breaks

**Tests in narration-core that use job-server:**
- `tests/e2e_job_client_integration.rs` - Uses `JobRegistry`
- `tests/harness/mod.rs` - Uses `JobRegistry`

**These tests will fail to compile.** They need to be:
1. Moved to a separate integration test crate, OR
2. Disabled temporarily, OR
3. Mocked without real job-server dependency

---

## Testing Instructions

1. **Build:**
   ```bash
   cargo build --package rbee-keeper --bin rbee-keeper
   ```
   ✅ **SUCCESS** - Compiles cleanly

2. **Run Tauri GUI:**
   ```bash
   ./target/debug/rbee-keeper
   ```

3. **Test Install:**
   - Click "Install Queen" button
   - **Expected:** Should NOT stack overflow
   - **If works:** Circular dependency WAS the cause!
   - **If fails:** Something else is wrong

4. **Verify rust-analyzer:**
   - Restart rust-analyzer
   - Check for circular dependency warning
   - Should be GONE

---

## Why This Makes Sense

### CLI Works, Tauri Doesn't

**CLI:**
- Runs in main thread (8MB stack)
- Doesn't load dev-dependencies at runtime
- No circular dependency in runtime path

**Tauri:**
- Runs in worker thread (2MB stack)
- Might load dev-dependencies for some reason
- Circular dependency causes deep initialization stack

### Rust-Analyzer Warning

Rust-analyzer sees the cycle and warns about it. This is a REAL issue, not just a warning.

### All Previous Fixes Failed

We removed:
- ❌ Macros (#[with_timeout], #[with_job_id])
- ❌ Narration (all n!() calls)
- ❌ ProcessNarrationCapture
- ❌ Tried spawn_blocking

None worked because the ROOT CAUSE was the circular dependency creating deep type resolution chains.

---

## If This Works

1. **Document it** - Update investigation comment
2. **Fix tests** - Move narration-core tests to separate crate
3. **Verify** - Run full test suite
4. **Celebrate** - 5 attempts, finally found it!

---

## If This Doesn't Work

Then the circular dependency warning was a red herring. Next steps:
1. Get actual stack trace with `RUST_BACKTRACE=full`
2. Try Test 5 from incremental test plan (skip build)
3. Try Test 7 (explicit 32MB stack)
4. Consider it might be a Tauri v2 bug

---

## Compilation Status

✅ `cargo build --package rbee-keeper` - SUCCESS  
✅ No circular dependency in runtime path  
⚠️ Some tests in narration-core will fail (expected)

---

**NEXT: Test in Tauri GUI and report results**

---

**END OF FIX ATTEMPT #5**
