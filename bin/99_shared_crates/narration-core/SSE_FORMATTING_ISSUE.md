# SSE Stream Formatting Issue & Resolution

**TEAM-197** | **Date:** 2025-10-21 | **Status:** RESOLVED

---

## ============================================================
## BUG FIX: TEAM-197 | Narration v0.5.0 format not showing in SSE streams
## ============================================================

### SUSPICION:
- Initially thought the `job-registry` code was using the old `Narration::new()` API
- Suspected caching issues - old compiled code still being used
- Thought maybe narration-core v0.5.0 wasn't being linked properly
- Suspected the daemon process wasn't being restarted with new binary

### INVESTIGATION:

**Step 1: Verified the Code Was Correct**
- Checked `job-registry/src/lib.rs` line 300-304
  - ✅ Code uses `NARRATE.action("execute").context(...).human(...).emit()`
  - ✅ This is the correct v0.5.0 pattern
- Checked narration-core formatting in `lib.rs` line 449
  - ✅ Format string is correct: `"[{:<10}] {:<15}: {}"`
  - ✅ Should produce fixed-width output
- Rebuilt queen-rbee multiple times
  - ❌ Output still shows `[job-exec] Executing job...` (wrong format)

**Step 2: Ruled Out Caching Issues**
- Ran `cargo clean` followed by full rebuild
  - ❌ Still wrong output
- Killed queen-rbee daemon with `pkill -9 queen-rbee`
- Restarted daemon via `./rbee queen start`
  - ❌ Still wrong output
- Checked binary timestamp with `ls -lh target/debug/queen-rbee`
  - ✅ Binary was freshly built (timestamp after rebuild)
  - ✅ Process was using new binary (verified with `ps aux`)
- **CONCLUSION:** Not a caching issue

**Step 3: Searched for Alternative Output Sources**
- Searched for `println!` and `eprintln!` in queen-rbee
  - Found none that would produce `[job-exec]` format
- Searched for hardcoded `[job-exec]` string
  - Found none in the codebase
- Checked if old `Narration::new()` API was being used
  - ❌ Code correctly uses `NARRATE.action()` (v0.5.0 pattern)
- Checked Cargo.lock for narration-core version
  - ✅ Version 0.5.0 is being used
- **CONCLUSION:** Code is correct, but output is wrong

**Step 4: Critical Observation**
- Noticed the output format: `[job-exec] Executing job...`
  - NO spacing after `[job-exec]`
  - NO action column
  - NO colon before message
- This format does NOT match narration-core's `eprintln!` at line 449
- **KEY INSIGHT:** The output is NOT coming from narration-core at all!

**Step 5: Traced the Output Flow**
- Realized job execution happens in `tokio::spawn` (async)
- Narration goes to TWO places:
  1. **stderr** via `eprintln!` (line 449 in narration-core)
  2. **SSE stream** via `sse_sink::send()` (line 452-454 in narration-core)
- Checked keeper's logs - saw CORRECT format from keeper's own narration
- But job execution logs showed WRONG format
- **HYPOTHESIS:** SSE stream consumer is re-formatting the events!

**Step 6: Found the Root Cause**
- Searched for SSE event formatting in queen-rbee
- Located in `/bin/10_queen_rbee/src/http/jobs.rs` line 108:
  ```rust
  // OLD CODE (WRONG):
  let formatted = format!("[{}] {}", event.actor, event.human);
  ```
- This line receives `NarrationEvent` from SSE stream
- It RE-FORMATS the event using OLD pattern (no action, no spacing)
- This formatted string goes to keeper via SSE
- Keeper prints it directly (line 102 in `job_client.rs`)
- **ROOT CAUSE CONFIRMED:** SSE consumer was overriding narration-core's format!

### ROOT CAUSE:

**The Actual Problem:**
- Narration-core v0.5.0 formats events correctly and sends them to TWO destinations:
  1. **stderr** - Formatted with fixed-width columns (CORRECT)
  2. **SSE stream** - Sends raw `NarrationEvent` struct (no formatting)

**The Bug:**
- SSE consumers (like queen-rbee) receive raw `NarrationEvent` structs
- They must format these events themselves for display
- Queen-rbee's SSE endpoint (`/bin/10_queen_rbee/src/http/jobs.rs` line 108) was using OLD format:
  ```rust
  let formatted = format!("[{}] {}", event.actor, event.human);
  ```
- This OLD format has:
  - ❌ No fixed-width padding
  - ❌ No action column
  - ❌ No colon separator

**Why It Happened:**
- When narration-core was upgraded to v0.5.0, the SSE consumer code wasn't updated
- The SSE consumer still used the old formatting pattern from v0.4.0
- This created inconsistent output:
  - Keeper's own logs: CORRECT format (via stderr)
  - Job execution logs: WRONG format (via SSE stream)

**The Flow:**
```
1. job-registry calls NARRATE.action("execute").emit()
2. narration-core formats: "[job-exec  ] execute        : Executing job..."
3. narration-core outputs to stderr (CORRECT)
4. narration-core sends NarrationEvent{actor, action, human} to SSE
5. queen-rbee SSE endpoint receives NarrationEvent
6. queen-rbee RE-FORMATS: "[job-exec] Executing job..." (WRONG!)
7. keeper receives SSE stream and prints it (wrong format)
```

### FIX:

**Location:** `/bin/10_queen_rbee/src/http/jobs.rs` line 108-109

**What Changed:**
```rust
// BEFORE (WRONG):
let formatted = format!("[{}] {}", event.actor, event.human);

// AFTER (CORRECT):
// TEAM-197: Format narration with fixed-width columns for consistency
// Format: "[actor     ] action         : message"
let formatted = format!("[{:<10}] {:<15}: {}", event.actor, event.action, event.human);
```

**Why This Solves the Problem:**
- Now uses the SAME format as narration-core v0.5.0
- `{:<10}` = Left-align actor in 10-character field
- `{:<15}` = Left-align action in 15-character field
- Produces consistent output: `[job-exec  ] execute        : Executing job...`
- Matches stderr output exactly

**Files Modified:**
- `/bin/10_queen_rbee/src/http/jobs.rs` (line 108-109)

### TESTING:

**Verification Steps:**
1. Rebuilt queen-rbee: `cargo build -p queen-rbee`
   - ✅ Compilation successful

2. Killed old daemon: `pkill -9 queen-rbee`
   - ✅ Process terminated

3. Started new daemon: `./rbee queen start`
   - ✅ Daemon started with new binary

4. Triggered job execution: `./rbee hive start`
   - ✅ Output now shows CORRECT format:
   ```
   [job-exec  ] execute        : Executing job job-3a1312a8...
   [qn-router ] route_job      : Executing operation: hive_start
   [job-exec  ] failed         : Job job-3a1312a8... failed: ...
   ```

5. Verified column alignment:
   - ✅ Actor column: 10 characters (left-aligned with padding)
   - ✅ Action column: 15 characters (left-aligned with padding)
   - ✅ Message starts at column 31 (consistent with stderr output)

6. Tested multiple job executions:
   - ✅ All narration events show correct format
   - ✅ Format matches keeper's own logs (stderr)
   - ✅ No more inconsistent formatting

**Edge Cases Tested:**
- Long job IDs (truncation handled correctly)
- Error messages (format preserved)
- Multiple concurrent jobs (all formatted correctly)

**Result:** ✅ Bug fixed, format is now consistent across all output paths

## ============================================================

---

## Why This Bug Was So Hard to Debug

### Multiple Output Paths Masked the Issue:
1. **Narration goes to TWO places:** stderr AND SSE streams
2. **Stderr showed CORRECT format:** Keeper's own logs looked perfect
3. **SSE showed WRONG format:** But we didn't realize it was a separate path
4. **Misleading evidence:** Seeing correct stderr output made us think the code was working

### The SSE Consumer Was Hidden:
1. **Re-formatting happened downstream:** Not in narration-core
2. **Async execution:** Job execution in `tokio::spawn` made tracing harder
3. **Daemon process:** Required restarts, easy to test with stale binary
4. **No obvious connection:** SSE formatting code was in a different crate

### What Made It Click:
- Noticing the EXACT format: `[job-exec]` (no spacing, no colon)
- Realizing this format doesn't match narration-core's `eprintln!`
- Understanding narration has multiple output paths
- Searching for where SSE events get formatted for display

---

## The Core Problem: Decentralized Formatting

**Current Architecture Issue:**

Narration-core formats events and sends them to TWO destinations:
1. **stderr** - Formatted with fixed-width columns (correct)
2. **SSE stream** - Sends raw `NarrationEvent` struct (no formatting)

**The Problem:**
- Each SSE consumer must format events themselves
- If consumers use different formats, output becomes inconsistent
- When narration-core format changes, ALL consumers must be updated manually
- This creates maintenance burden and debugging difficulty

**Example of the Issue:**
- Keeper's own logs (stderr): `[job-exec  ] execute        : Executing job...` ✅
- Job execution logs (SSE): `[job-exec] Executing job...` ❌ (before fix)

---

## Next Steps for Future Team

**TASK:** Investigate and propose a solution for centralizing narration formatting.

**Goal:** Ensure ALL output paths (stderr, SSE, logs) use the SAME format without requiring each consumer to know the formatting rules.

**Current SSE Consumers That Format Events:**
1. `/bin/10_queen_rbee/src/http/jobs.rs` - Line 108-109 (FIXED to match v0.5.0)
2. `/bin/00_rbee_keeper/src/job_client.rs` - Line 102 (just prints)
3. Any other crates that consume SSE narration events

**Questions to Answer:**
- Should SSE stream send pre-formatted strings instead of raw structs?
- Or should consumers use a shared formatting function?
- How to maintain backward compatibility?
- How to ensure format consistency across all output paths?

**Files to Review:**
- `/bin/99_shared_crates/narration-core/src/sse_sink.rs` - SSE event structure
- `/bin/99_shared_crates/narration-core/src/lib.rs` - Main formatting logic (line 449)
- All SSE consumers (search for `NarrationEvent` usage)

---

## Debugging Tips for Future Teams

### If narration format looks wrong:

1. **Check stderr first:** Run the command and look at direct stderr output
   - If stderr is correct but SSE is wrong → SSE consumer is re-formatting
   - If stderr is wrong → narration-core has the bug

2. **Find ALL output paths:**
   ```bash
   # Search for where narration events are printed
   grep -r "println!\|eprintln!" --include="*.rs" | grep -i "event\|narration\|actor"
   ```

3. **Check SSE consumers:**
   ```bash
   # Find all SSE event formatting
   grep -r "format!" --include="*.rs" | grep -i "event\|actor"
   ```

4. **Trace the flow:**
   - Where is `NARRATE.action().emit()` called?
   - Where does `narrate_at_level()` send the event?
   - Where does the SSE consumer receive it?
   - Where does the SSE consumer format/print it?

5. **Use unique markers:**
   Add a unique string to the format to trace where output comes from:
   ```rust
   let formatted = format!("[{:<10}] {:<15}: 🔍DEBUG🔍 {}", ...);
   ```

### Common Pitfalls:

❌ **Assuming stderr = SSE format:** They can be different!  
❌ **Forgetting daemon restarts:** Queen-rbee must be restarted after rebuilds  
❌ **Not checking all consumers:** Every SSE consumer might format differently  
❌ **Caching issues:** Use `cargo clean` if behavior is inexplicable  

---

## Related Files

- **Narration Core:** `/bin/99_shared_crates/narration-core/src/lib.rs`
- **SSE Sink:** `/bin/99_shared_crates/narration-core/src/sse_sink.rs`
- **Job Registry:** `/bin/99_shared_crates/job-registry/src/lib.rs`
- **Queen SSE Endpoint:** `/bin/10_queen_rbee/src/http/jobs.rs`
- **Keeper SSE Consumer:** `/bin/00_rbee_keeper/src/job_client.rs`

---

## Lessons Learned

1. **Multiple output paths can mask bugs** - Check ALL paths (stderr, SSE, logs)
2. **Decentralized formatting creates maintenance burden** - Consider centralizing
3. **Async execution makes tracing harder** - Be aware of `tokio::spawn` boundaries
4. **Daemon processes need explicit restarts** - Old binaries can persist
5. **Document the complete flow** - Make it clear where events go and how they're formatted

---

**Next Team:** Please investigate and propose a solution for centralizing narration formatting to prevent future inconsistencies.

**Priority:** MEDIUM - Current fix works, but architecture could be improved for better DX.

---

## FOLLOW-UP: Complete Solution (TEAMS 199-203)

**TEAM-197's fix was correct but incomplete.** The root cause was decentralized formatting.

**Complete solution implemented by:**
- **TEAM-199:** Security fix (redaction in SSE path)
- **TEAM-200:** Job-scoped SSE broadcaster (isolation)
- **TEAM-201:** Centralized formatting (pre-formatted SSE events)
- **TEAM-202:** Hive narration (using job-scoped SSE)
- **TEAM-203:** Verification and documentation

**See:** START_HERE_TEAMS_199_203.md for complete architecture.

**Status:** ✅ **COMPLETE** - All narration now flows through secure, isolated, consistently-formatted SSE.

---

**TEAM-197** | **2025-10-21** | **Issue Resolved, Architecture Improvement Recommended**  
**TEAM-203** | **2025-10-22** | **Architecture Improvement Completed**
