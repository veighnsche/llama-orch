# START HERE: Narration SSE Architecture Implementation

**Teams:** 199, 200, 201, 202, 203  
**Mission:** Implement centralized, secure, web-UI proof narration  
**Based On:** TEAM-197's critical review of TEAM-198's proposal

---

## ⚠️ CRITICAL: READ THIS FIRST

**TEAM-198 did excellent analysis but has 3 CRITICAL FLAWS:**

1. 🚨 **Security Issue:** Missing redaction in SSE path
2. 🚨 **Isolation Issue:** Global SSE broadcaster (no job separation)
3. 🚨 **Architecture Issue:** Wrong ingestion endpoint design

**TEAM-197 provided fixes. You MUST implement the corrected version.**

---

## Team Assignments

### TEAM-199: Security Fix (Redaction in SSE)
**Document:** `TEAM-199-REDACTION-FIX.md`  
**Priority:** 🚨 **CRITICAL - SECURITY ISSUE**  
**Duration:** 2-3 hours  
**Status:** MUST complete before other teams start

**Mission:** Fix missing redaction in SSE path (FLAW 1 from TEAM-197)

**Deliverables:**
- Fix `From<NarrationFields>` to redact ALL fields
- Add tests for redaction in SSE events
- Verify secrets don't leak through SSE

---

### TEAM-200: Job-Scoped SSE Broadcaster
**Document:** `TEAM-200-JOB-SCOPED-SSE.md`  
**Priority:** 🚨 **CRITICAL - ISOLATION ISSUE**  
**Duration:** 4-6 hours  
**Status:** Start after TEAM-199 completes

**Mission:** Refactor SSE broadcaster to support job-specific channels (OPPORTUNITY 1 from TEAM-197)

**Deliverables:**
- Job-scoped SSE broadcaster
- Thread-local channel support
- Global fallback for non-job narration

---

### TEAM-201: Centralized Formatting
**Document:** `TEAM-201-CENTRALIZED-FORMATTING.md`  
**Priority:** HIGH  
**Duration:** 3-4 hours  
**Status:** Can start after TEAM-199 completes

**Mission:** Add `formatted` field to `NarrationEvent` (TEAM-198 Phase 1, corrected)

**Deliverables:**
- Add `formatted: String` field to `NarrationEvent`
- Pre-format in `From<NarrationFields>`
- Update queen consumer to use `event.formatted`

---

### TEAM-202: Hive Narration
**Document:** `TEAM-202-HIVE-NARRATION.md`  
**Priority:** MEDIUM  
**Duration:** 3-4 hours  
**Status:** Start after TEAM-200 and TEAM-201 complete

**Mission:** Add narration to hive using thread-local pattern (no HTTP ingestion)

**Deliverables:**
- Replace `println!()` with `NARRATE.action().emit()`
- Narration flows through job-scoped SSE
- Test with remote hive

---

### TEAM-203: Verification & Documentation
**Document:** `TEAM-203-VERIFICATION.md`  
**Priority:** HIGH  
**Duration:** 2-3 hours  
**Status:** Start after all other teams complete

**Mission:** End-to-end verification and documentation updates

**Deliverables:**
- E2E test: remote hive/worker narration visible in keeper
- Update architecture docs
- Clean up TEAM-198's incorrect proposals

---

## Implementation Order (CRITICAL)

```
┌─────────────────────────────────────────────────────────────┐
│ IMPLEMENTATION SEQUENCE (DO NOT CHANGE)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  TEAM-199: Security Fix (Redaction)                        │
│     ↓                                                        │
│  ┌────────────────────┬────────────────────┐               │
│  │ TEAM-200:          │ TEAM-201:          │               │
│  │ Job-Scoped SSE     │ Centralized Format │               │
│  └────────────────────┴────────────────────┘               │
│     ↓                                                        │
│  TEAM-202: Hive Narration                                  │
│     ↓                                                        │
│  TEAM-203: Verification & Docs                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Why This Order:**
1. **TEAM-199 FIRST:** Security fix must be in place before SSE changes
2. **TEAM-200 & TEAM-201:** Can work in parallel after TEAM-199
3. **TEAM-202:** Needs both job-scoped SSE and formatted field
4. **TEAM-203:** Final verification after everything is integrated

---

## What TEAM-198 Got Right ✅

Keep these parts of TEAM-198's proposal:
- ✅ Core problem identification (decentralized formatting)
- ✅ Solution direction (centralized formatting)
- ✅ Add `formatted` field to `NarrationEvent`
- ✅ Phased rollout approach

---

## What TEAM-197 Fixed 🚨

Implement these corrections:
- 🚨 **Redact ALL fields** in SSE path (not just `human`)
- 🚨 **Job-scoped SSE** (not global broadcaster)
- 🚨 **Thread-local channels** (not HTTP ingestion endpoint)
- 🚨 **No separate ingestion API** (use worker pattern)

---

## Critical Decisions

### ❌ DO NOT IMPLEMENT (from TEAM-198):

1. **HTTP Ingestion Endpoint** (`POST /v1/narration`)
   - **Why:** No job isolation, authentication issues
   - **Instead:** Use thread-local channels (worker pattern)

2. **send_formatted() Helper**
   - **Why:** Creates events with empty actor/action fields
   - **Instead:** Use normal `From<NarrationFields>` path

3. **Fire-and-Forget HTTP POST**
   - **Why:** Still blocks on .await, no error handling
   - **Instead:** Thread-local channels (no network hop)

### ✅ DO IMPLEMENT (from TEAM-197):

1. **Job-Scoped SSE Broadcaster**
   - Per-job channels with `HashMap<String, broadcast::Sender>`
   - Thread-local channel support
   - Global fallback for non-job narration

2. **Complete Redaction in SSE**
   - Redact: `human`, `cute`, `story`, `target`
   - Same security properties as stderr path

3. **Thread-Local Pattern (Like Worker)**
   - Hive narration uses request-scoped channels
   - No separate HTTP API needed
   - Proven pattern already in worker

---

## Files to Create

### By TEAM-199
- Tests for SSE redaction (in narration-core)

### By TEAM-200
- Job-scoped broadcaster implementation
- Thread-local channel support

### By TEAM-201
- None (modifies existing files only)

### By TEAM-202
- `bin/20_rbee_hive/src/narration.rs`

### By TEAM-203
- E2E test for remote narration
- Updated architecture docs

---

## Files to Modify

### TEAM-199
- `bin/99_shared_crates/narration-core/src/sse_sink.rs` (Fix `From<NarrationFields>`)

### TEAM-200
- `bin/99_shared_crates/narration-core/src/sse_sink.rs` (Job-scoped broadcaster)
- `bin/99_shared_crates/narration-core/src/lib.rs` (Thread-local support)

### TEAM-201
- `bin/99_shared_crates/narration-core/src/sse_sink.rs` (Add formatted field)
- `bin/10_queen_rbee/src/http/jobs.rs` (Use event.formatted)

### TEAM-202
- `bin/20_rbee_hive/src/main.rs` (Replace println! with NARRATE)
- `bin/20_rbee_hive/Cargo.toml` (Add narration-core dependency if missing)

### TEAM-203
- Various docs (architecture, SSE flow, etc.)

---

## Success Criteria

### Security
- [ ] All SSE events have redacted secrets
- [ ] No secrets leak through SSE that are hidden in stderr
- [ ] Test: SSE stream doesn't contain API keys, tokens, passwords

### Isolation
- [ ] Multiple concurrent jobs have separate SSE streams
- [ ] Job A doesn't see Job B's narration
- [ ] Test: Two concurrent keeper commands don't cross-contaminate

### Functionality
- [ ] Keeper sees ALL narration (local and remote)
- [ ] Format is consistent everywhere
- [ ] No manual formatting in consumers
- [ ] Test: `./rbee hive status` shows hive narration

### Performance
- [ ] No extra network hops for narration
- [ ] Thread-local channels work correctly
- [ ] No blocking on queen availability

---

## Common Pitfalls (from TEAM-197)

### ❌ WRONG: Fire-and-Forget HTTP
```rust
// BAD: Still blocks on .await
let _ = client.post(url).send().await;
```

### ❌ WRONG: Empty Actor/Action
```rust
// BAD: Creates invalid events
NarrationEvent {
    actor: String::new(),  // Empty!
    action: String::new(), // Empty!
    // ...
}
```

### ❌ WRONG: Partial Redaction
```rust
// BAD: Only redacts human
let human = redact_secrets(&fields.human, ...);
// cute, story, target NOT redacted!
```

### ✅ CORRECT: Thread-Local Channel
```rust
// GOOD: No network hop, automatic job scoping
if let Some(tx) = THREAD_LOCAL_CHANNEL.with(|c| c.borrow().clone()) {
    tx.send(event).await;
}
```

---

## Testing Strategy

### Unit Tests (Each Team)
- TEAM-199: Redaction tests for all fields
- TEAM-200: Job-scoped broadcaster tests
- TEAM-201: Formatting consistency tests
- TEAM-202: Hive narration emission tests

### Integration Tests (TEAM-203)
- E2E: Remote hive narration visible in keeper
- E2E: Multiple concurrent jobs isolated
- E2E: Format consistency across all components

---

## References

### Primary Documents
1. **TEAM-197-ARCHITECTURE-REVIEW.md** - Critical review (THIS IS YOUR BIBLE)
2. **TEAM-198's NARRATION_SSE_ARCHITECTURE.md** - Original proposal (DO NOT follow blindly)
3. **SSE_FORMATTING_ISSUE.md** - TEAM-197's original bug fix

### Code References
1. **Worker narrate_dual()** - Proven thread-local pattern
2. **Current sse_sink.rs** - Global broadcaster (needs refactoring)
3. **narrate_at_level()** - Redaction logic (must mirror in SSE)

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│ IMPLEMENTATION QUICK REFERENCE                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ TEAM-199 (Security):                                       │
│   Fix: Redact ALL fields in From<NarrationFields>         │
│   Test: SSE events don't leak secrets                      │
│                                                             │
│ TEAM-200 (Isolation):                                      │
│   Add: Job-scoped SSE broadcaster                          │
│   Add: Thread-local channel support                        │
│   Test: Multiple jobs isolated                             │
│                                                             │
│ TEAM-201 (Formatting):                                     │
│   Add: formatted field to NarrationEvent                   │
│   Update: Queen consumer uses event.formatted              │
│   Test: Format consistency                                 │
│                                                             │
│ TEAM-202 (Hive):                                           │
│   Replace: println!() with NARRATE.action().emit()        │
│   Use: Thread-local channel (like worker)                  │
│   Test: Remote hive narration visible                      │
│                                                             │
│ TEAM-203 (Verification):                                   │
│   E2E: All narration visible in keeper                     │
│   E2E: No cross-contamination                              │
│   Docs: Update architecture                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Engineering Rules Reminder

**MANDATORY RULES:**
- ✅ Add TEAM-XXX signature to all modified files
- ✅ Complete previous team's TODO list
- ✅ No TODO markers (implement or delete)
- ✅ Handoff ≤2 pages with code examples
- ✅ Show actual progress (function count, tests added)
- ❌ NO "next team should implement X"
- ❌ NO analysis without implementation
- ❌ NO background testing (use foreground)

---

## Summary

**Mission:** Implement web-UI proof narration with centralized formatting, job isolation, and security.

**Approach:** Fix TEAM-198's proposal based on TEAM-197's critical review.

**Key Changes:**
- Add redaction to SSE path (security)
- Add job-scoped broadcaster (isolation)
- Use thread-local channels (simplicity)
- Pre-format events (consistency)

**Expected Outcome:**
- ✅ All narration visible in keeper (local and remote)
- ✅ No secrets leaked through SSE
- ✅ Jobs properly isolated
- ✅ Simple, ergonomic API

---

**Created by:** TEAM-197  
**For Teams:** 199, 200, 201, 202, 203  
**Status:** READY TO START

**Read your team's specific document, then implement. Questions? Check TEAM-197-ARCHITECTURE-REVIEW.md first.**
