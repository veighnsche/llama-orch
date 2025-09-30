# URGENT: Narration-Core Is Critical for Debugging

**To**: All llama-orch Teams  
**From**: Engineering Leadership  
**Date**: 2025-09-30  
**Priority**: HIGH  
**Subject**: Why `observability/narration-core` Must Be Completed NOW

---

## TL;DR

**You've been asking for better debugging tools. We already built 95% of the spec for it—but never finished the implementation.** This memo explains why `observability/narration-core` is critical, why it was abandoned, and what we need to do to complete it.

**Bottom line**: This crate will solve your debugging pain, but it requires 2-3 weeks of focused work and cross-team adoption.

---

## The Problem You're Experiencing

### What You're Telling Us:
- "I can't trace requests across services"
- "Logs are inconsistent—some use println!, some use tracing"
- "When something breaks, I have to grep through multiple log formats"
- "I don't know which service is causing the delay"
- "Debugging multi-service flows is a nightmare"

### We Hear You. And We Have a Solution.

---

## What Is Narration-Core?

`observability/narration-core` is a **shared logging facade** designed to give you:

### 1. **Human-Readable Story Flow**
Instead of cryptic JSON, you get plain English:
```
"Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'"
"Spawning engine llamacpp-v1 for pool 'default' on GPU0"
"Building llama.cpp with CUDA support for GPU0"
```

### 2. **Correlation IDs Across Services**
Track a single request from orchestratord → pool-managerd → engine-provisioner:
```bash
$ grep "correlation_id=req-xyz" logs/*.log
orchestratord.log: {"correlation_id":"req-xyz","actor":"orchestratord","action":"admission",...}
pool-managerd.log: {"correlation_id":"req-xyz","actor":"pool-managerd","action":"spawn",...}
engine-provisioner.log: {"correlation_id":"req-xyz","actor":"engine-provisioner","action":"build",...}
```

### 3. **Structured Fields**
Every log entry has consistent fields:
- `actor` (who did it: orchestratord, pool-managerd, etc.)
- `action` (what they did: admission, spawn, build, etc.)
- `target` (what they acted on: pool_id, replica_id, session_id, etc.)
- `human` (plain English description)
- `correlation_id` (trace across services)
- Context fields: `pool_id`, `replica_id`, `session_id`, `job_id`, `engine`, `device`, etc.

### 4. **Automatic Secret Redaction**
No more accidentally logging API tokens or sensitive data. Narration-core will automatically mask:
- Bearer tokens
- API keys
- Passwords
- PII

### 5. **Test Capture for BDD**
Your BDD tests can assert:
```rust
assert_narration_includes("spawned engine llamacpp-v1");
assert_correlation_id_present("req-xyz");
```

### 6. **Story Snapshots**
Generate golden files showing the complete flow of a scenario for proof bundles and documentation.

---

## The Current Reality (Why You're Struggling)

### Inconsistent Logging Across Services

**orchestratord**:
```rust
tracing::info!("task created");
```

**pool-managerd**:
```rust
println!("Spawning engine...");
```

**engine-provisioner**:
```rust
eprintln!("Building llama.cpp");
```

**Result**: Three different logging styles, no correlation, no structure, no story.

### No Correlation IDs

When a request touches 3 services, you have **no way to trace it**. You're left grepping for timestamps and guessing which logs belong together.

### Ad-Hoc Debug Logging

From `ROBUSTNESS_FIXES_NEEDED.md`:
```rust
tracing::debug!("expected_tokens={}, checking sentinels", exp);
tracing::debug!("triggering QueueFullDropLru");
```

Every developer reinvents logging. No consistency. No structure.

### Secrets Could Leak

Without automatic redaction, it's easy to accidentally log:
```rust
tracing::info!("Authorization: Bearer {}", token);  // ← LEAKED!
```

### Tests Can't Assert on Logs

BDD tests use a hacky `state.logs` mutex instead of proper log capture. This makes observability testing fragile and inconsistent.

---

## What We Already Have (95% Planned, 5% Implemented)

### ✅ Comprehensive Specs
- 6 spec files under `libs/observability/narration-core/.specs/`
- Accepted proposal with 13 requirements (ORCH-3300..3312)
- Detailed migration plan in `SPEC_CHANGES_NEEDED.md`
- Integration points documented for 8+ crates

### ✅ Minimal Implementation
- One 16-line `human()` function exists
- Used in 4 places in orchestratord:
  - Startup narration
  - HTTP/2 narration
  - Admission narration
  - Cancel narration

### ❌ Missing Components (Why You're Still Struggling)
- **No redaction helpers** (secrets could leak)
- **No test capture adapter** (BDD can't assert on logs)
- **No formatting helpers** (no story snapshots)
- **No field taxonomy expansion** (no correlation IDs, contextual keys)
- **No pretty vs JSON toggle** (stuck with whatever tracing gives you)
- **No sampling controls** (can't rate-limit under load)
- **Zero adoption by other crates** (pool-managerd, provisioners, adapters still use println!)

---

## Why Was This Abandoned?

### 1. No Immediate Crisis
- Tests passed with the `state.logs` mutex hack
- No production incidents forced better observability
- "Works well enough" syndrome

### 2. Spec-First Paralysis
- We wrote 135 lines of proposal and 6 spec files
- Planning felt like progress
- But implementation never followed
- **Classic mistake**: Documentation as procrastination

### 3. Cross-Crate Coordination Overhead
- Needed 8+ crates to adopt it
- No forcing function or mandate
- Orchestratord added it, nobody else followed
- **Result**: Orphaned crate

### 4. Test Capture Adapter Never Built
- BDD tests still use `state.logs` mutex
- No test-driven reason to expand narration-core
- Chicken-egg problem: No tests demanding it, so why build it?

### 5. Redaction Seemed Hard
- Fear of false positives (masking too much) or false negatives (leaking secrets)
- Easier to punt than get it wrong
- **Result**: Security requirement ignored

### 6. YAGNI Mindset
- "You Aren't Gonna Need It"
- Minimal `human()` function "works"
- Advanced features felt like over-engineering
- **Result**: Stopped at 10% implementation

---

## Example: Before vs After

### Debugging Scenario: "My Request Is Stuck"

#### WITHOUT Narration-Core (Current State)

**User reports**: "My request to pool 'default' is stuck. What's happening?"

**Your debugging experience**:
```bash
# Grep orchestratord logs
$ grep "task" orchestratord.log
{"level":"info","msg":"task created"}
{"level":"info","msg":"task created"}
{"level":"info","msg":"task created"}
# Which one is mine? No idea.

# Grep pool-managerd logs
$ grep "Spawning" pool-managerd.log
Spawning engine...
Spawning engine...
# No correlation. No context. No pool_id.

# Grep engine-provisioner logs
$ grep "Building" engine-provisioner.log
Building llama.cpp
Building llama.cpp
# Which build is for my request? No clue.
```

**Result**: You spend 30 minutes correlating timestamps, guessing which logs belong together, and still aren't sure what's stuck.

#### WITH Narration-Core (What You Could Have)

**User reports**: "My request with session_id 'session-abc123' is stuck."

**Your debugging experience**:
```bash
# Grep by session_id across ALL services
$ grep "session-abc123" logs/*.log

orchestratord.log:
{"level":"info","actor":"orchestratord","action":"admission","target":"session-abc123",
 "human":"Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'",
 "correlation_id":"req-xyz","session_id":"session-abc123","pool_id":"default",
 "queue_position":3,"predicted_start_ms":420}

pool-managerd.log:
{"level":"info","actor":"pool-managerd","action":"spawn","target":"GPU0",
 "human":"Spawning engine llamacpp-v1 for pool 'default' on GPU0",
 "correlation_id":"req-xyz","pool_id":"default","replica_id":"r0",
 "engine":"llamacpp-v1","device":"GPU0"}

engine-provisioner.log:
{"level":"info","actor":"engine-provisioner","action":"build","target":"llamacpp-v1",
 "human":"Building llama.cpp with CUDA support for GPU0",
 "correlation_id":"req-xyz","engine":"llamacpp","version":"v1","device":"GPU0"}

# Still stuck? Grep by correlation_id to see EVERYTHING
$ grep "req-xyz" logs/*.log
# See the complete story from admission to current state
```

**Result**: You find the issue in 30 SECONDS, not 30 minutes.

---

## What You Can Do With Narration-Core

### 1. Trace Requests Across Services
```bash
$ grep "correlation_id=req-xyz" logs/*.log
# See the entire flow: orchestratord → pool-managerd → engine-provisioner
```

### 2. Filter by Actor
```bash
$ grep "actor=pool-managerd" logs/*.log
# See everything pool-managerd did
```

### 3. Filter by Action
```bash
$ grep "action=spawn" logs/*.log
# See all engine spawns across all pools
```

### 4. Read the Story
```bash
$ grep "human=" logs/orchestratord.log | jq -r '.human'
Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'
Spawning engine llamacpp-v1 for pool 'default' on GPU0
Building llama.cpp with CUDA support for GPU0
Engine ready; 4 slots available
```

### 5. Assert in Tests
```rust
#[then(regex = "^engine is spawned$")]
pub async fn then_engine_spawned(world: &mut World) {
    world.assert_narration_includes("Spawning engine");
    world.assert_field_present("pool_id", "default");
    world.assert_correlation_id_present();
}
```

### 6. Generate Story Snapshots
```bash
$ cargo test --features story-snapshots
# Generates golden files showing complete scenario flows
# Use in proof bundles and documentation
```

---

## What Needs to Happen

### Phase 1: Complete Core Implementation (Week 1)

**Owner**: TBD  
**Effort**: 5 days  

1. **Redaction Helpers** (1 day)
   - Regex-based secret masking
   - Handle tokens, API keys, passwords, PII
   - Unit tests for edge cases

2. **Test Capture Adapter** (2 days)
   - Replace `state.logs` mutex in BDD
   - Provide `assert_narration_includes()`, `assert_field_present()`, etc.
   - Integration tests

3. **Field Taxonomy Expansion** (1 day)
   - Add correlation ID propagation
   - Add contextual fields: `pool_id`, `replica_id`, `session_id`, `job_id`, `engine`, `device`
   - Update `human()` function signature

4. **Formatting Helpers** (1 day)
   - Stable narrative entries for snapshots
   - Story snapshot generation
   - Golden file support

### Phase 2: Cross-Crate Adoption (Week 2)

**Owner**: All teams (coordinated)  
**Effort**: 5 days  

1. **pool-managerd** (1 day)
   - Replace `println!` with narration calls
   - Add narration for: spawn, health check, supervision, crash recovery
   - Add correlation ID propagation from orchestratord

2. **engine-provisioner** (1 day)
   - Replace `println!` with narration calls
   - Add narration for: preflight, build, CUDA checks, spawn

3. **model-provisioner** (1 day)
   - Replace `println!` with narration calls
   - Add narration for: download, validation, staging

4. **adapter-host** (1 day)
   - Add narration wrappers for: submit, cancel, health, props
   - Propagate correlation IDs

5. **worker-adapters** (1 day)
   - Add narration for: streaming, errors, retries
   - Use redaction helpers for HTTP requests/responses

### Phase 3: Testing & Enforcement (Week 3)

**Owner**: Test team  
**Effort**: 3 days  

1. **BDD Narration Coverage** (1 day)
   - Add narration assertions to all scenarios
   - Generate coverage metrics
   - Set threshold: ≥80% initially, ratchet to 95%

2. **Story Snapshots** (1 day)
   - Generate golden files for key scenarios
   - Include in proof bundles
   - Update `.docs/testing/` guidance

3. **CI Enforcement** (1 day)
   - Add narration coverage gate to CI
   - Fail if coverage drops below threshold
   - Add link checker for correlation IDs

---

## Success Criteria

### Week 1 (Core Implementation)
- ✅ Redaction helpers with tests
- ✅ Test capture adapter working in BDD
- ✅ Field taxonomy expanded (correlation IDs, contextual fields)
- ✅ Formatting helpers and story snapshots

### Week 2 (Cross-Crate Adoption)
- ✅ All services use narration-core (no more println!)
- ✅ Correlation IDs propagate across service boundaries
- ✅ Consistent actor/action/target taxonomy

### Week 3 (Testing & Enforcement)
- ✅ BDD narration coverage ≥80%
- ✅ Story snapshots for key scenarios
- ✅ CI enforces narration coverage
- ✅ Proof bundles include narration excerpts

### Ongoing
- ✅ Debugging time reduced by 50%+ (measured)
- ✅ No more "I can't trace this request" complaints
- ✅ No secret leaks (redaction working)
- ✅ Tests assert on observability (not just behavior)

---

## Why This Matters

### For Developers
- **Faster debugging**: Find issues in seconds, not minutes
- **Better tests**: Assert on observability, not just behavior
- **Less frustration**: No more grepping through inconsistent logs

### For Operations
- **Incident response**: Trace requests across services instantly
- **Performance analysis**: Identify bottlenecks with structured timing data
- **Security**: Automatic redaction prevents secret leaks

### For Management
- **Reduced MTTR**: Mean time to resolution drops significantly
- **Better proof bundles**: Story snapshots show complete flows
- **Spec compliance**: Finally implement what we planned

### For Users
- **Better support**: We can debug their issues faster
- **Transparency**: Generate "what happened" summaries from logs
- **Trust**: We know what our system is doing

---

## The Cost of NOT Doing This

### Technical Debt
- Every crate continues to reinvent logging
- No consistency across services
- Debugging remains painful and slow
- Secrets could leak (no redaction)

### Opportunity Cost
- BDD can't verify observability coverage
- No story snapshots for proof bundles
- Can't generate user-facing summaries
- Debugging time remains high

### Morale Cost
- Teams continue to complain about debugging
- Specs exist but aren't followed (erodes trust)
- "We planned it but never did it" becomes a pattern
- Frustration builds

### Competitive Cost
- Other orchestrators have better observability (e.g., Kubernetes, Nomad)
- We look unprofessional when debugging takes forever
- Users notice when we can't explain what happened

---

## The Ask

### From Leadership
1. **Prioritize this work**: Allocate 2-3 weeks for completion
2. **Assign ownership**: Designate a lead for each phase
3. **Mandate adoption**: All services MUST adopt narration-core (not optional)
4. **Enforce in CI**: Narration coverage becomes a gate

### From Teams
1. **Adopt narration-core**: Replace println!/tracing with narration calls
2. **Propagate correlation IDs**: Pass them across service boundaries
3. **Use structured fields**: Follow actor/action/target taxonomy
4. **Assert in tests**: Use capture adapter in BDD

### From Everyone
1. **Commit to completion**: No more "we'll get to it later"
2. **Hold each other accountable**: Review PRs for narration usage
3. **Celebrate progress**: Track adoption metrics and share wins

---

## FAQ

### Q: Why not just use tracing directly?
**A**: Tracing is low-level. Narration-core provides:
- Consistent taxonomy (actor/action/target)
- Automatic redaction
- Correlation ID propagation
- Test capture adapter
- Story snapshots

### Q: Won't this slow down hot paths?
**A**: No. Narration-core uses tracing under the hood, which is already zero-cost when disabled. Structured fields are just as fast as string interpolation.

### Q: What if I don't want human-readable logs?
**A**: You get both! Structured JSON for machines, human-readable `human` field for humans. Filter by `human=` to read the story, or parse JSON for analysis.

### Q: Can I opt out?
**A**: No. This is a repo-wide standard. Consistent observability is critical for debugging multi-service flows.

### Q: What about existing logs?
**A**: They'll continue to work. Narration-core is additive. We'll migrate incrementally.

### Q: Who maintains this?
**A**: The observability team owns the crate. Each service team owns adoption in their crate.

---

## Conclusion

**You asked for better debugging tools. We already designed them—we just never finished building them.**

Narration-core will:
- ✅ Let you trace requests across services
- ✅ Give you consistent, structured logs
- ✅ Provide human-readable story flow
- ✅ Automatically redact secrets
- ✅ Enable observability testing in BDD
- ✅ Reduce debugging time by 50%+

**But it requires commitment**:
- 2-3 weeks of focused work
- Cross-team adoption
- CI enforcement

**The spec is there. The plan is there. The need is there. Only execution is missing.**

Let's finish what we started.

---

**Next Steps**:
1. Leadership: Approve resource allocation (this week)
2. Assign phase owners (this week)
3. Kick off Phase 1 (next week)
4. Track progress weekly
5. Celebrate 100% adoption (Week 3)

**Questions?** Reply to this memo or join the #observability channel.

---

**Status**: Awaiting leadership approval and resource allocation  
**Owner**: TBD  
**Target Completion**: 2025-10-21 (3 weeks from now)
