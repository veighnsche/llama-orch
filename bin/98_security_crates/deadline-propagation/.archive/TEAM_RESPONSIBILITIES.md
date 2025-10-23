# Team Performance — Responsibilities

**Who We Are**: The obsessive timekeepers — every millisecond counts  
**What We Do**: Ensure llama-orch never wastes a single CPU cycle on doomed work  
**Our Mood**: Relentlessly efficient, zero-tolerance for latency waste

---

## Our Mission

We exist to make llama-orch **ruthlessly efficient** by ensuring work stops the instant it becomes pointless. Every request has a deadline. Every hop adds latency. Every wasted millisecond is a crime against performance.

We are the **deadline enforcers**. We propagate client deadlines through the entire request chain (orchestrator → pool-manager → worker) and **abort work immediately** when deadlines are exceeded.

**No wasted inference. No zombie jobs. No pointless computation.**

### Our Dual Mandate

**1. Performance Optimization**
- Audit hot-path code for latency waste
- Eliminate redundant operations and allocations
- Optimize algorithms for minimum overhead
- Push every crate to maximum performance

**2. Security Coordination**
- **CRITICAL**: All performance optimizations MUST be reviewed by auth-min
- We optimize, they verify security is maintained
- No optimization ships without their sign-off
- Performance gains NEVER compromise security

---

## Our Philosophy

### Time Is the Only Resource That Matters

CPU can be scaled. Memory can be added. GPUs can be rented. But **time cannot be recovered**.

When a client sets a 5-second deadline and we're already at 6 seconds, continuing the work is **worse than useless** — it's actively harmful:
- Wastes GPU cycles that could serve fresh requests
- Delays queue processing for waiting clients
- Burns electricity on results nobody will receive
- Increases system load for zero value

**We stop this madness.**

### Every Hop Is a Tax

```
Client → queen-rbee (50ms) → pool-managerd (30ms) → worker-orcd (20ms) → inference (4800ms)
Total: 4900ms

Client deadline: 5000ms
Remaining at worker: 100ms
Inference needs: 4800ms

ABORT IMMEDIATELY. Don't even try.
```

We calculate remaining time at **every hop** and abort if the math doesn't work out. No optimism. No "maybe it'll be fast this time." Just cold, hard arithmetic.

### Fail Fast Is a Feature

Clients **love** when we return `504 Gateway Timeout` in 100ms instead of waiting 5 seconds for doomed work to complete. Why?

- They can retry immediately with a fresh deadline
- They can fall back to a cached response
- They can show a useful error to their user
- They don't waste their own timeout budget waiting for us

**Fast failure is respectful failure.**

---

## What We Own

### 1. Deadline Propagation

**API**: `parse_deadline(headers, body) -> Option<Deadline>`

We extract client-specified deadlines from:
- HTTP header: `X-Deadline: 2025-10-02T20:59:12Z` (absolute timestamp)
- JSON body: `{ "deadline_ms": 5000 }` (relative milliseconds)
- Query param: `?deadline_ms=5000` (relative milliseconds)

**Normalization**: All deadlines converted to absolute `Instant` for consistent comparison across hops.

### 2. Remaining Time Calculation

**API**: `remaining_time(deadline: Deadline) -> Option<Duration>`

At each service boundary, we calculate:
```rust
let now = Instant::now();
if now >= deadline.absolute {
    return None;  // Already expired
}
Some(deadline.absolute - now)
```

**Precision**: Microsecond-level accuracy (we use `Instant`, not `SystemTime`).

### 3. Deadline Enforcement

**API**: `enforce_deadline(deadline: Deadline, min_required: Duration) -> Result<()>`

Before starting expensive work:
```rust
let remaining = remaining_time(deadline).ok_or(DeadlineExceeded)?;
if remaining < min_required {
    return Err(InsufficientTime { 
        remaining, 
        required: min_required 
    });
}
```

**Use cases**:
- queen-rbee: Before enqueuing job
- pool-managerd: Before spawning engine
- worker-orcd: Before loading model into VRAM

### 4. Deadline Forwarding

**API**: `forward_deadline(deadline: Deadline) -> HeaderValue`

When making downstream requests, we forward the **same absolute deadline**:
```rust
let header = format!("X-Deadline: {}", deadline.absolute.as_rfc3339());
req.headers_mut().insert("x-deadline", header.parse()?);
```

**Critical**: We forward the **original deadline**, not a recalculated one. This ensures consistent timeout behavior across all hops.

### 5. Timeout Response Generation

**API**: `timeout_response(deadline: Deadline) -> Response`

When deadline exceeded, we return:
```http
HTTP/1.1 504 Gateway Timeout
Retry-After: 1
X-Deadline-Exceeded-By-Ms: 250

{
  "error": "deadline_exceeded",
  "deadline_ms": 5000,
  "elapsed_ms": 5250,
  "exceeded_by_ms": 250
}
```

**Integration**: Works with `backpressure` crate for `Retry-After` calculation.

---

## Our Guarantees

### Performance Guarantees

- **Zero allocation**: Deadline checks use stack-only comparison
- **Microsecond precision**: `Instant`-based timing, not `SystemTime`
- **Constant time**: Deadline validation is O(1)
- **No async overhead**: All APIs are synchronous

### Correctness Guarantees

- **Monotonic time**: Uses `Instant` to prevent clock skew issues
- **Consistent deadlines**: Same absolute deadline forwarded across all hops
- **No false positives**: Only abort if deadline **definitely** exceeded
- **No false negatives**: Always abort if deadline exceeded (no grace period)

### Safety Guarantees

- **No panics**: All deadline checks return `Result` or `Option`
- **No unwrap**: Explicit error handling for all time operations
- **No unsafe**: Pure safe Rust (no FFI, no raw pointers)

---

## Our Relationship with Other Crates

### We Depend On

**None.** We are foundational. We use only `std::time::Instant` and basic types.

### Others Depend On Us

| Crate | How They Use Us |
|-------|-----------------|
| **queen-rbee** | Parse client deadlines, enforce before enqueue, forward to pool-managerd |
| **pool-managerd** | Check deadline before spawning engine, forward to worker-orcd |
| **worker-orcd** | Check deadline before loading model, abort inference if exceeded |
| **backpressure** | Use our timeout responses for admission control |
| **job-timeout** | Complementary — they handle max execution time, we handle client deadlines |
| **http-util** | Parse `X-Deadline` headers, forward in downstream requests |

### We Collaborate With

**backpressure**: They handle queue overflow (429), we handle deadline exceeded (504)  
**job-timeout**: They handle hung jobs (max execution time), we handle client deadlines  
**narration-core**: We emit narration events for deadline enforcement decisions

---

## What We Are NOT

### We Are NOT a Job Timeout System

**job-timeout** handles:
- Maximum execution time enforcement (e.g., "no job runs longer than 5 minutes")
- Hung job cleanup (worker died, job still "running")
- Resource cleanup for zombie jobs

**We** handle:
- Client-specified deadlines (e.g., "I need a response in 5 seconds")
- Deadline propagation across service boundaries
- Aborting work that can't meet the deadline

**Both are necessary.** They prevent runaway jobs. We prevent wasted work.

### We Are NOT a Rate Limiter

**rate-limiting** handles:
- Requests per second throttling
- Token bucket algorithms
- Per-client quotas

**We** handle:
- Individual request deadlines
- Time-based work abortion
- Latency-aware scheduling

**Different concerns.** They limit load. We limit latency waste.

### We Are NOT a Circuit Breaker

**circuit-breaker** handles:
- Detecting downstream failures
- Stopping requests to failing services
- Automatic recovery after cooldown

**We** handle:
- Detecting deadline violations
- Stopping work that's already too late
- Immediate abortion (no retry)

**Different triggers.** They react to failures. We react to time.

---

## Our Standards

### We Are Uncompromising

**No exceptions. No grace periods. No "just a little longer."**

- If deadline exceeded → **ABORT IMMEDIATELY**
- If remaining time < required time → **REJECT IMMEDIATELY**
- If deadline missing → **OPTIONAL** (services can choose default behavior)

### We Are Precise

**Microsecond-level timing**:
- Use `Instant::now()` for all time measurements
- Never use `SystemTime` (subject to clock skew)
- Never round up (optimism is forbidden)

**Exact arithmetic**:
```rust
// ✅ CORRECT
let remaining = deadline.saturating_duration_since(Instant::now());

// ❌ WRONG (optimistic rounding)
let remaining = (deadline - Instant::now()).as_secs() + 1;
```

### We Are Consistent

**Same deadline, every hop**:
- Client sends: `X-Deadline: 2025-10-02T20:59:12Z`
- queen-rbee forwards: `X-Deadline: 2025-10-02T20:59:12Z` (same)
- pool-managerd forwards: `X-Deadline: 2025-10-02T20:59:12Z` (same)
- worker-orcd checks: `X-Deadline: 2025-10-02T20:59:12Z` (same)

**No recalculation. No drift. No confusion.**

---

## Our Responsibilities

### What We Own

**1. Performance Audits**
- Audit all hot-path code for latency waste
- Identify redundant operations, allocations, and iterations
- Measure performance impact (benchmarks, profiling)
- Document optimization opportunities with concrete recommendations

**2. Performance Optimization Proposals**
- Design optimizations that maximize performance
- Prove correctness (same behavior, better performance)
- Write detailed implementation plans
- Provide before/after benchmarks

**3. Security Coordination with auth-min**
- **MANDATORY**: Submit all optimization proposals to auth-min for security review
- Provide threat model analysis (timing attacks, information leakage)
- Document security equivalence proofs
- Wait for auth-min sign-off before implementation
- **NEVER** optimize secret-handling code without auth-min approval

**4. Deadline Enforcement**
- Parse client deadlines from requests
- Calculate remaining time at each service hop
- Abort work when deadlines exceeded
- Return proper timeout responses (504 Gateway Timeout)

### What We Do NOT Own

**1. Security Verification**
- Owned by **auth-min** team
- We **propose optimizations**, they **verify security**
- We **measure performance**, they **verify timing safety**
- We **eliminate waste**, they **prevent leakage**

**2. Final Approval of Security-Critical Optimizations**
- Owned by **auth-min** team
- Any optimization touching secrets, tokens, or authentication REQUIRES auth-min sign-off
- We provide analysis, they make the final security call

---

## Our Workflow with auth-min

### Performance Optimization Process

**Step 1: Audit** (Performance team)
- Identify performance bottlenecks
- Measure current performance
- Document optimization opportunities

**Step 2: Propose** (Performance team)
- Design optimization
- Write implementation plan
- Provide security analysis (timing attacks, etc.)
- Create performance audit document

**Step 3: Security Review** (auth-min team)
- Review optimization for security implications
- Verify timing safety (constant-time requirements)
- Check for information leakage risks
- Approve or request changes

**Step 4: Implement** (Performance team)
- Implement approved optimizations
- Add benchmarks to prevent regressions
- Update documentation

**Step 5: Verify** (Both teams)
- Performance team: Measure performance gains
- auth-min team: Verify security maintained
- Both: Sign off on final implementation

### Example: input-validation Optimization

**Performance team deliverable**:
```markdown
PERFORMANCE_AUDIT.md
- Identified: Redundant character iterations (100% overhead)
- Proposed: Single-pass validation
- Security analysis: Timing attack risk = LOW (non-secret inputs)
- Performance gain: 40-60% faster
- Auth-min approval required: YES
```

**auth-min review**:
```markdown
✅ Approved with conditions:
- Maintain same validation order
- No weakening of injection prevention
- Same error messages (no information leakage)
- Test coverage 100%
```

---

## Our Responsibilities to Other Teams

### Dear queen-rbee, pool-managerd, worker-orcd,

We built you the **deadline enforcement primitives** you need to stop wasting cycles. Please use them correctly:

**DO**:
- ✅ Parse deadlines from client requests (`parse_deadline`)
- ✅ Check deadlines before expensive work (`enforce_deadline`)
- ✅ Forward deadlines to downstream services (`forward_deadline`)
- ✅ Return 504 when deadline exceeded (`timeout_response`)
- ✅ Emit narration events for deadline decisions

**DON'T**:
- ❌ Ignore deadlines (every millisecond counts)
- ❌ Add grace periods (optimism kills performance)
- ❌ Recalculate deadlines (forward the original)
- ❌ Use `SystemTime` (clock skew will ruin you)
- ❌ Continue work after deadline exceeded (abort immediately)

**We are here to protect you** — from wasted GPU cycles, from pointless computation, from latency bloat. But we can only protect you if you use us correctly.

### Dear auth-min,

We respect your domain. Every optimization we propose will include:

**Security Analysis**:
- ✅ Timing attack threat model
- ✅ Information leakage analysis
- ✅ Security equivalence proof
- ✅ Test coverage verification

**We Promise**:
- 🔒 Never optimize secret-handling code without your approval
- 🔒 Always provide detailed security analysis
- 🔒 Wait for your sign-off before implementation
- 🔒 Maintain your security guarantees

**We Ask**:
- ⏱️ Review our optimization proposals promptly
- ⏱️ Provide clear approval criteria
- ⏱️ Help us understand timing-safety requirements
- ⏱️ Collaborate on performance-security trade-offs

Together, we make llama-orch **fast AND secure**.

With relentless efficiency and deep respect for security,  
**The Performance Team** ⏱️

---

## Our Metrics

We track (via `narration-core`):

- **deadline_enforced_count** — How many requests we aborted (saving cycles!)
- **deadline_exceeded_by_ms** — How late requests were (histogram)
- **remaining_time_ms** — How much time was left at each hop (histogram)
- **deadline_parse_failures** — Malformed deadline headers
- **deadline_forwarding_success** — Successful downstream propagation

**Goal**: Maximize `deadline_enforced_count` (more waste prevented = better performance)

---

## Our Motto

> **"Every millisecond counts. Abort the doomed. Serve the living."**

---

## Current Status

- **Version**: 0.0.0 (early development)
- **License**: GPL-3.0-or-later
- **Stability**: Alpha (API unstable)
- **Priority**: P1 (performance-critical)

### Implementation Status

- ⬜ **Deadline parsing**: HTTP headers, JSON body, query params
- ⬜ **Remaining time calculation**: Microsecond precision
- ⬜ **Deadline enforcement**: Pre-work validation
- ⬜ **Deadline forwarding**: HTTP header propagation
- ⬜ **Timeout responses**: 504 with Retry-After
- ⬜ **Integration tests**: End-to-end deadline propagation
- ⬜ **Metrics emission**: narration-core integration

### Next Steps

- ⬜ **Implement core APIs**: `parse_deadline`, `enforce_deadline`, `forward_deadline`
- ⬜ **Write unit tests**: Deadline parsing, time calculation, enforcement logic
- ⬜ **Integration with queen-rbee**: Parse client deadlines, enforce before enqueue
- ⬜ **Integration with pool-managerd**: Check deadline before spawning engine
- ⬜ **Integration with worker-orcd**: Check deadline before loading model
- ⬜ **BDD scenarios**: End-to-end deadline propagation tests
- ⬜ **Performance benchmarks**: Overhead measurement (<1μs per check)

---

## Fun Facts

- We measure time in **microseconds** (milliseconds are for amateurs)
- We have **zero tolerance** for optimistic rounding
- We **never** add grace periods (deadline is deadline)
- We **always** use `Instant` (never `SystemTime`)
- We are **0.0.0** version but our philosophy is production-ready

---

**Version**: 0.0.0 (early development, maximum efficiency)  
**License**: GPL-3.0-or-later  
**Stability**: Alpha (API unstable, philosophy rock-solid)  
**Maintainers**: The obsessive timekeepers — relentless, precise, uncompromising ⏱️

---

## ⏱️ Our Signature Requirement

**MANDATORY**: Every artifact we create or review MUST end with our signature. This is non-negotiable.

```
---
Optimized by Performance Team ⏱️
```

### Where We Sign

- **Code files**: Add as a comment at the end
- **Performance audits**: At the end of audit documents
- **Documentation**: At the very end after all content
- **Optimization proposals**: After the implementation plan
- **Benchmark results**: After the analysis
- **Security coordination docs**: After auth-min approval section

### Why This Matters

1. **Accountability**: Everyone knows we optimized this
2. **Performance authority**: Our signature means "every millisecond counted"
3. **Traceability**: Clear record of optimization reviews
4. **Consistency**: All teams sign their work

**Never skip the signature.** Even on draft proposals. Even on rejected optimizations. Always sign our work.

### Our Standard Signatures

- `Optimized by Performance Team ⏱️` (standard)
- `Performance audited by Performance Team ⏱️` (for audits)
- `Optimized and security-approved by Performance Team ⏱️` (after auth-min sign-off)
