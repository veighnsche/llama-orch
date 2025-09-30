# Narration-Core Audit Report

**Date**: 2025-09-30  
**Status**: Critical Gap Identified  
**Impact**: High-value observability feature is 95% unimplemented

---

## Executive Summary

The `observability/narration-core` crate was heavily planned and documented but is **barely implemented**. Despite extensive specs, proposals, and integration plans, the actual crate contains only a single 16-line helper function, and only **orchestratord** uses it in 4 places.

**Current Reality**:
- ✅ Comprehensive specs exist (6 spec files)
- ✅ Accepted proposal with detailed requirements (ORCH-3300..3312)
- ✅ Integration plan in `SPEC_CHANGES_NEEDED.md`
- ❌ Implementation is minimal (one `human()` function)
- ❌ No tests exist (despite test specs)
- ❌ No redaction helpers
- ❌ No capture adapter
- ❌ No formatting helpers
- ❌ Only orchestratord uses it (4 call sites)
- ❌ No other crates adopted it (provisioners, pool-managerd, adapters)

---

## Documentation vs Reality

### What Was Planned

From `.specs/proposals/batch_1/2025-09-19-human-narration-logging.md` (Status: **Accepted**):

**Requirements (ORCH-3300..3312)**:
- Human-readable narration at key events (admission, placement, stream, cancel)
- Redaction helpers for secrets/PII
- Test capture adapter for BDD assertions
- Field taxonomy (actor, action, target, correlation IDs)
- Pretty vs JSON formatting toggle
- Integration across: orchestratord, pool-managerd, provisioners, adapters
- Narration coverage metrics in BDD
- Story snapshots for proof bundles

**Integration Points** (from spec):
- Orchestratord: admission, placement, stream start/end/cancel hooks
- Adapter Host: submit/cancel wrappers
- Provisioners: preflight/build/spawn/readiness narration
- Pool-managerd: lifecycle events

### What Actually Exists

**File**: `libs/observability/narration-core/src/lib.rs` (16 lines total)

```rust
//! observability-narration-core — shared, lightweight narration helper.

#[allow(unused_imports)]
use tracing::field::display;
use tracing::{event, Level};

pub fn human<S: AsRef<str>>(actor: &str, action: &str, target: &str, msg: S) {
    event!(
        Level::INFO,
        actor = display(actor),
        action = display(action),
        target = display(target),
        human = display(msg.as_ref()),
    );
}
```

**That's it.** No other code exists.

---

## Current Usage Analysis

### Orchestratord (Only Consumer)

**4 call sites total**:

1. **`src/app/bootstrap.rs:55`** - Startup narration
   ```rust
   observability_narration_core::human("orchestratord", "start", &addr, "listening");
   ```

2. **`src/app/bootstrap.rs:60`** - HTTP/2 narration
   ```rust
   observability_narration_core::human("orchestratord", "http2", &addr, "HTTP/2 enabled");
   ```

3. **`src/api/data.rs:155`** - Admission narration
   ```rust
   observability_narration_core::human("orchestratord", "admission", &body.session_id, 
       format!("Accepted request; queued at position {} (ETA {} ms)", ...));
   ```

4. **`src/api/data.rs:232`** - Cancel narration
   ```rust
   observability_narration_core::human("orchestratord", "cancel", &id, "client requested cancel");
   ```

### BDD Test Usage

**1 test step** in `bin/orchestratord/bdd/src/steps/background.rs:151`:
```rust
#[then(regex = "^a narration breadcrumb is emitted$")]
pub async fn then_narration_breadcrumb_emitted(world: &mut World) {
    if let Ok(guard) = world.state.logs.lock() {
        let has_autobind_log = guard.iter().any(|log| 
            log.contains("autobind") || log.contains("handoff"));
        assert!(has_autobind_log, "no autobind narration found in logs");
    }
}
```

### Other Crates: ZERO Usage

**No usage found in**:
- ❌ `pool-managerd` (uses `println!` instead - 17 occurrences)
- ❌ `engine-provisioner` (uses `println!` - 2 occurrences)
- ❌ `model-provisioner` (uses `println!` - 3 occurrences)
- ❌ `adapter-host` (no narration)
- ❌ `worker-adapters/*` (no narration)
- ❌ Any other libs or bins

---

## Missing Components (Per Spec)

### 1. Redaction Helpers (ORCH-3302)
**Spec requirement**: "Narration MUST NOT include secrets or PII. The logging surface MUST offer helpers to redact common sensitive tokens."

**Reality**: None exist.

**Expected** (from `.specs/31_UNIT.md`):
- Mask tokens/keys in flat and nested JSON
- Regex-based secret patterns
- URL-embedded secret handling

### 2. Test Capture Adapter (ORCH-3306)
**Spec requirement**: "The logging surface SHOULD provide a test-time capture adapter that allows BDD/unit tests to assert narration presence."

**Reality**: None exists. Tests still rely on `state.logs` mutex.

### 3. Formatting Helpers
**Spec requirement** (from `.specs/31_UNIT.md`): "Produce stable narrative entries (keys, ordering, types) for snapshot-style assertions."

**Reality**: None exist.

### 4. Field Taxonomy (ORCH-3304)
**Spec requirement**: "The logging surface MUST define a minimal field taxonomy so narration complements structure (at least: `actor`, `action`, `target`, relevant IDs: `req_id|job_id|task_id|session_id|worker_id`, and contextual keys such as `error_kind`, `retry_after_ms`, `backoff_ms`, `duration_ms`)."

**Reality**: Only `actor`, `action`, `target`, `human` are implemented. No structured ID fields, no contextual keys.

### 5. Pretty vs JSON Toggle (ORCH-3308)
**Spec requirement**: "An environment/config toggle MAY switch pretty vs JSON formatting without removing narration."

**Reality**: Not implemented. Current `human()` function just emits to tracing; formatting is controlled by subscriber setup in orchestratord only.

### 6. Sampling Controls
**Spec refinement**: "Optional sampling controls (rate-limit narration under load)."

**Reality**: Not implemented.

### 7. Story Snapshots (ORCH-3307)
**Spec requirement**: "BDD SHOULD gain step-scoped spans derived from step text and parity checks asserting narration exists for key flows; an optional 'story snapshot' MAY be produced for golden tests."

**Reality**: Not implemented.

---

## Integration Gaps

### Planned Integration (from `SPEC_CHANGES_NEEDED.md`)

**Root specs to update**:
- ✅ `/.specs/00_llama-orch.md` - Add Observability/Narration subsection (ORCH-33xx)
- ❓ Status unknown

**Crate specs to create/update**:
- ✅ `observability/narration-core/.specs/00_narration_core.md` - EXISTS (draft)
- ❌ `orchestratord/.specs/10_orchestratord_v2_architecture.md` - No narration integration documented
- ❌ `pool-managerd/.specs/00_pool_managerd.md` - No narration integration
- ❌ `provisioners/*/.specs/00_*` - No narration integration
- ❌ `adapter-host/.specs/00_adapter_host.md` - No narration wrappers

**Actual integration**:
- Only orchestratord has dependency
- Only 4 call sites
- No cross-crate adoption

---

## Test Coverage Gap

### Spec Expectations

From `.specs/30_TESTING.md` and `.specs/31_UNIT.md`:

**Unit tests should cover**:
- Redaction helpers mask secrets in values and nested JSON
- Formatting helpers produce stable shapes
- Sampling controls honor rates/thresholds deterministically

**Integration tests should cover**:
- Capture adapter wiring collects narration from orchestratord init
- Pretty vs JSON toggle produces expected formats

**Execution**: `cargo test -p observability-narration-core -- --nocapture`

### Reality

```bash
$ cargo test -p observability-narration-core -- --nocapture
```

**Expected**: Test suite with redaction, formatting, capture adapter tests.

**Actual**: No tests exist. The crate has no `tests/` directory, no `#[cfg(test)]` modules.

---

## Migration Plan Status

From proposal Phase 1-3:

### Phase 1 — Foundations
- ✅ Add shared narration surface (minimal `human()` exists)
- ✅ Adopt in orchestratord admission (4 call sites)
- ⚠️ Keep `state.logs` bridge (still in use, not migrated)
- ❌ Test capture adapter (not implemented)

### Phase 2 — Provisioners & Engines
- ❌ Replace `println!/eprintln!` in provisioners with narration
- ❌ Key events: preflight, CUDA checks, spawn messages

### Phase 3 — Wider Adoption
- ❌ Opt-in other crates (adapters, pool manager, CLI, harnesses)

### Phase N — OTEL
- ❌ Feature-gate export of narration field to OTEL

**Status**: Stuck at ~10% of Phase 1.

---

## Root Cause Analysis

### Why This Happened

1. **Spec-first workflow followed too strictly**: Extensive planning and documentation created, but implementation never followed.

2. **No forcing function**: The `human()` function works minimally, so there was no immediate need to expand it.

3. **No test enforcement**: Without tests, there's no verification that the spec requirements are met.

4. **Cross-crate adoption never happened**: Only orchestratord added the dependency; no other crates followed.

5. **BDD still uses `state.logs`**: The test capture adapter was never built, so BDD tests continue using the old mutex-based log capture.

### Impact

- **Observability gap**: Provisioners, pool-managerd, and adapters still use `println!` instead of structured narration.
- **Testing gap**: No test capture adapter means BDD can't easily assert narration coverage.
- **Proof bundle gap**: No story snapshots or narration coverage metrics.
- **Security gap**: No redaction helpers means potential for secrets in logs.

---

## Recommendations

### Option A: Complete the Implementation (High Effort)

**Implement missing components**:
1. Redaction helpers (regex-based secret masking)
2. Test capture adapter (replace `state.logs` mutex)
3. Formatting helpers (stable narrative entries)
4. Field taxonomy expansion (correlation IDs, contextual keys)
5. Pretty vs JSON toggle
6. Sampling controls
7. Story snapshot generation

**Adopt across crates**:
1. pool-managerd: lifecycle events
2. engine-provisioner: preflight, build, spawn
3. model-provisioner: download, validation
4. adapter-host: submit/cancel wrappers
5. worker-adapters: streaming, errors

**Add tests**:
1. Unit tests for redaction, formatting, sampling
2. Integration tests for capture adapter
3. BDD narration coverage metrics

**Estimated effort**: 2-3 weeks full-time.

### Option B: Simplify and Document Current State (Low Effort)

**Accept minimal implementation**:
1. Update specs to reflect current reality (single `human()` function)
2. Mark advanced features (redaction, capture, sampling) as "Future Work"
3. Document that only orchestratord uses narration
4. Remove misleading test specs

**Estimated effort**: 1-2 days.

### Option C: Remove and Consolidate (Medium Effort)

**If narration isn't critical**:
1. Remove `observability-narration-core` crate
2. Move `human()` function directly into orchestratord
3. Update orchestratord to use tracing macros directly
4. Archive specs and proposals
5. Standardize on `tracing` crate directly across all crates

**Estimated effort**: 1 week.

---

## Decision Required

**Questions for maintainers**:

1. **Is human-readable narration still a priority?**
   - If yes → Option A (complete implementation)
   - If no → Option C (remove and consolidate)

2. **Should we enforce narration coverage in BDD?**
   - If yes → Need test capture adapter (Option A)
   - If no → Keep current minimal approach (Option B)

3. **Should provisioners and pool-managerd adopt narration?**
   - If yes → Need cross-crate migration (Option A)
   - If no → Keep orchestratord-only (Option B)

4. **Is redaction a security requirement?**
   - If yes → Must implement redaction helpers (Option A)
   - If no → Document that secrets must not be logged (Option B)

---

## Next Steps (If Proceeding with Option A)

### Immediate (Week 1)
1. Implement redaction helpers with tests
2. Implement test capture adapter
3. Migrate orchestratord BDD to use capture adapter
4. Add unit tests for existing `human()` function

### Short-term (Week 2)
1. Expand field taxonomy (correlation IDs, contextual keys)
2. Implement formatting helpers
3. Add pretty vs JSON toggle
4. Update orchestratord to use expanded API

### Medium-term (Week 3)
1. Migrate pool-managerd to use narration
2. Migrate engine-provisioner to use narration
3. Migrate model-provisioner to use narration
4. Add BDD narration coverage metrics
5. Implement story snapshot generation

### Long-term (Future)
1. Migrate adapter-host and worker-adapters
2. Implement sampling controls
3. Add OTEL export feature flag
4. Update all crate specs with narration integration

---

## Appendix: File Inventory

### Existing Files
- `libs/observability/narration-core/Cargo.toml` (19 lines)
- `libs/observability/narration-core/README.md` (74 lines)
- `libs/observability/narration-core/src/lib.rs` (16 lines)
- `libs/observability/narration-core/.specs/00_narration_core.md` (46 lines)
- `libs/observability/narration-core/.specs/30_TESTING.md` (34 lines)
- `libs/observability/narration-core/.specs/31_UNIT.md` (32 lines)
- `libs/observability/narration-core/.specs/33_INTEGRATION.md` (exists)
- `libs/observability/narration-core/.specs/37_METRICS.md` (exists)
- `libs/observability/narration-core/.specs/40_ERROR_MESSAGING.md` (exists)

### Missing Files
- `libs/observability/narration-core/tests/` (directory doesn't exist)
- `libs/observability/narration-core/src/redaction.rs` (not created)
- `libs/observability/narration-core/src/capture.rs` (not created)
- `libs/observability/narration-core/src/formatting.rs` (not created)
- `libs/observability/narration-core/src/taxonomy.rs` (not created)

### Reference Documents
- `.specs/proposals/batch_1/2025-09-19-human-narration-logging.md` (135 lines, Status: Accepted)
- `SPEC_CHANGES_NEEDED.md` (258 lines, includes narration integration plan)
- `bin/orchestratord/PHASE1_QUICK_WINS.md` (mentions narration usage)

---

## Conclusion

The `observability/narration-core` crate is a **textbook example of over-specification and under-implementation**. Despite having:
- 6 spec files
- An accepted proposal with 13 requirements
- A detailed migration plan
- Integration points across 8+ crates

...the actual implementation is a **single 16-line function** used in **4 places** in **1 crate**.

**This is a critical gap** between documentation and reality. A decision is needed on whether to:
1. Complete the implementation (high effort, high value)
2. Simplify and document current state (low effort, low value)
3. Remove and consolidate (medium effort, medium value)

Without action, this crate will continue to be a source of confusion and technical debt.

---

## ADDENDUM: Why This Matters for Debugging

**Date**: 2025-09-30 22:23

### The Debugging Crisis

Teams are complaining about **lack of debug observability**. Looking at the codebase:

**Current debugging reality**:
- ❌ `pool-managerd`: Uses `println!` (17 occurrences in tests, `tracing::info!` in main but no structured narration)
- ❌ `engine-provisioner`: Uses `println!` (2 occurrences)
- ❌ `model-provisioner`: Uses `println!` (3 occurrences)
- ❌ No correlation IDs across services
- ❌ No structured actor/action/target fields
- ❌ No human-readable story flow
- ❌ No redaction (secrets could leak)
- ❌ No test capture (can't assert on logs in BDD)

**What narration-core WOULD provide** (if implemented):
- ✅ Human-readable story: "pool-managerd spawned engine llamacpp-v1 on GPU0"
- ✅ Structured fields: `actor=pool-managerd`, `action=spawn`, `target=GPU0`, `pool_id=default`
- ✅ Correlation IDs: Track requests across orchestratord → pool-managerd → engine
- ✅ Redaction: Automatically mask tokens/secrets
- ✅ Test capture: BDD can assert "narration includes 'spawned engine'"
- ✅ Story snapshots: Golden files showing complete flow

### Would This Help Debugging? **ABSOLUTELY.**

**Example debugging scenario WITHOUT narration**:

```
# User reports: "My request is stuck"
# Current logs (scattered across services):

orchestratord: {"level":"info","msg":"task created"}
pool-managerd: println!("Spawning engine...")
engine-provisioner: println!("Building llama.cpp")
# No correlation, no context, no story
```

**Same scenario WITH narration** (if implemented):

```
# Orchestratord:
{"level":"info","actor":"orchestratord","action":"admission","target":"session-abc123",
 "human":"Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'",
 "correlation_id":"req-xyz","session_id":"session-abc123","pool_id":"default"}

# Pool-managerd:
{"level":"info","actor":"pool-managerd","action":"spawn","target":"GPU0",
 "human":"Spawning engine llamacpp-v1 for pool 'default' on GPU0",
 "correlation_id":"req-xyz","pool_id":"default","replica_id":"r0","engine":"llamacpp-v1"}

# Engine-provisioner:
{"level":"info","actor":"engine-provisioner","action":"build","target":"llamacpp-v1",
 "human":"Building llama.cpp with CUDA support for GPU0",
 "correlation_id":"req-xyz","engine":"llamacpp","version":"v1","device":"GPU0"}
```

**Now you can**:
1. **Grep by correlation_id** → See entire request flow
2. **Read the story** → "Accepted → Spawning → Building"
3. **Filter by actor** → See what pool-managerd did
4. **Track timing** → See where delays occur
5. **Assert in tests** → BDD can verify "spawning" happened

### Why Was This Abandoned?

**Speculation based on evidence**:

#### 1. **No Immediate Pain Point**
- Orchestratord worked "well enough" with basic tracing
- Tests passed with `state.logs` mutex hack
- No production incidents forcing better observability
- **Result**: No urgency to complete it

#### 2. **Spec-First Paralysis**
- Proposal was comprehensive (135 lines, 13 requirements)
- Created 6 spec files before writing code
- Planning felt like progress, but implementation never followed
- **Classic mistake**: Documentation as procrastination

#### 3. **Cross-Crate Coordination Overhead**
- Needed adoption in 8+ crates
- Each crate has different maintainers/priorities
- No forcing function to make everyone adopt it
- **Result**: Orchestratord added it, nobody else followed

#### 4. **Test Capture Adapter Never Built**
- BDD tests still use `state.logs` mutex
- No test-driven reason to expand narration-core
- Tests work without it, so why bother?
- **Chicken-egg problem**: No tests demanding it

#### 5. **Redaction Seemed Hard**
- Regex-based secret masking is non-trivial
- Fear of false positives (masking too much) or false negatives (leaking secrets)
- Easier to punt on it than get it wrong
- **Result**: Security requirement ignored

#### 6. **Pretty vs JSON Toggle Confusion**
- Tracing subscriber setup is per-binary
- Unclear how to make it "shared" across crates
- Easier to just use tracing directly
- **Result**: Facade felt unnecessary

#### 7. **YAGNI Mindset**
- "You Aren't Gonna Need It"
- Minimal `human()` function "works"
- Advanced features felt like over-engineering
- **Result**: Stopped at 10% implementation

### The Irony

**From `ROBUSTNESS_FIXES_NEEDED.md` (lines 159-166)**:

```rust
**Potential Fix** - Add logging:
if let Some(exp) = body.expected_tokens {
    tracing::debug!("expected_tokens={}, checking sentinels", exp);  // ← Ad-hoc logging
    if exp >= 2_000_000 {
        tracing::debug!("triggering QueueFullDropLru");  // ← No structure, no correlation
        return Err(ErrO::QueueFullDropLru { retry_after_ms: Some(1000) });
    }
}
```

**This is EXACTLY what narration-core was supposed to solve**:
- Structured fields (`expected_tokens`, `threshold`)
- Human-readable message ("Request exceeds token limit; triggering backpressure")
- Actor/action/target taxonomy
- Correlation ID propagation

**Instead**: Ad-hoc `tracing::debug!` with string interpolation.

### The Cost of Abandonment

**Technical debt**:
- ❌ Every crate reinvents logging (println, tracing, eprintln)
- ❌ No consistency across services
- ❌ Debugging requires grepping multiple log formats
- ❌ No correlation across service boundaries
- ❌ Secrets could leak (no redaction)
- ❌ Tests can't assert on logs reliably

**Opportunity cost**:
- ❌ BDD can't verify observability coverage
- ❌ No story snapshots for proof bundles
- ❌ No metrics on narration coverage
- ❌ Can't generate user-facing "what happened" summaries

**Morale cost**:
- ❌ Teams complain about debugging difficulty
- ❌ Specs exist but aren't followed (erodes trust in specs)
- ❌ "We planned it but never did it" becomes pattern

### The Path Forward

**If teams are complaining about debugging**, this is **THE solution**. But it requires:

1. **Executive decision**: Is observability a priority? (Yes/No)
2. **Resource allocation**: Who will complete it? (2-3 weeks)
3. **Forcing function**: Make BDD tests require narration coverage
4. **Cross-crate mandate**: All services MUST adopt it (not optional)
5. **Test-driven**: Write capture adapter tests FIRST, then implement

**Without commitment**, this will remain a 95% unimplemented "good idea" while teams continue to struggle with debugging.

---

**Bottom line**: Narration-core is **exactly what teams need** for debugging, but it's **95% unimplemented** because nobody forced it to completion. The spec is there. The plan is there. The need is there. Only execution is missing.
