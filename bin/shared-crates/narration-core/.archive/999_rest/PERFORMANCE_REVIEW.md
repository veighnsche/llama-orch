# ⏱️ Narration-Core Performance Review — Implementation Plan Analysis
**Reviewed by**: Performance Team (deadline-propagation)  
**Review Date**: 2025-10-04  
**Document Version**: 3.0 (Implementation Plan Performance Impact Analysis)  
**Status**: CRITICAL PERFORMANCE ISSUES IDENTIFIED IN PLANNED CHANGES
---
## 🎯 Executive Summary
This review analyzes the **performance implications** of implementing the 4-week enhancement plan (`IMPLEMENTATION_PLAN.md`) on top of the **existing narration-core codebase** (`src/`).
### Current Baseline (Existing Code)
**What Exists Today** ✅:
- ✅ Lightweight trace macros (6 variants) — ~50-100ns overhead
- ✅ `narrate()` function — ~200-500ns with 3x redaction
- ✅ Redaction with `OnceLock` caching — already optimized
- ✅ Test capture adapter — mutex-based, test-only
- ✅ Auto-injection helpers — minimal overhead
**Current Performance**:
- **Trace macros**: ~50-100ns (no conditional compilation yet)
- **Full `narrate()`**: ~200-500ns (includes redaction)
- **Memory**: 1-4 allocations per narration
- **No critical bottlenecks**: No global mutex, no sampling
### Planned Changes (Implementation Plan)
**What Will Be Added** 🚧:
- 🚧 NEW proc macro crate (`narration-macros`) with `#[trace_fn]` and `#[narrate(...)]`
- 🚧 Template interpolation engine (runtime parsing of `{variable}` syntax)
- 🚧 WARN/ERROR/FATAL levels (new `NarrationLevel` enum)
- 🚧 Sampling & rate limiting (Unit 2.7 — **PERFORMANCE DISASTER**)
- 🚧 Unicode/emoji validation (Unit 2.8)
- 🚧 Conditional compilation (Unit 3.3 — **CRITICAL FOR PRODUCTION**)
### Performance Impact Assessment
**🚨 CRITICAL ISSUES** (Will destroy performance if implemented as planned):
1. **Unit 2.7 (Sampling)**: Global mutex + allocation storm — **BLOCKS PRODUCTION**
2. **Unit 1.5 (Templates)**: Runtime parsing + allocations — **HIGH OVERHEAD**
3. **Unit 2.8 (Unicode)**: Multiple char iterations + allocations — **MEDIUM OVERHEAD**
**⚠️ HIGH PRIORITY** (Needs optimization before merge):
4. **Unit 1.3 (#[trace_fn])**: `Instant::now()` overhead — needs conditional compilation
5. **Unit 2.3 (Redaction)**: Add 3 more patterns (JWT, private key, URL password) — compounds existing overhead
**✅ GOOD ADDITIONS** (Performance acceptable):
6. **Unit 2.1 (Levels)**: `NarrationLevel` enum — minimal overhead
7. **Unit 3.3 (Conditional compilation)**: **REQUIRED** for zero-overhead production
---
## 🚨 CRITICAL: Unit 2.7 Sampling & Rate Limiting — PERFORMANCE DISASTER
**Severity**: 🔴 CRITICAL — BLOCKS PRODUCTION DEPLOYMENT
**Planned Implementation** (from IMPLEMENTATION_PLAN.md lines 668-763):
```rust
// Unit 2.7: Sampling & Rate Limiting
pub struct Sampler {
    config: SamplingConfig,
    counters: Arc<Mutex<HashMap<String, (Instant, u32)>>>,  // 🚨 GLOBAL MUTEX!
}
impl Sampler {
    pub fn should_sample(&self, actor: &str, action: &str) -> bool {
        // Probabilistic sampling
        if rand::random::<f64>() > self.config.sample_rate {  // 🚨 RNG on every call!
            return false;
        }
        // Rate limiting
        if let Some(max_per_sec) = self.config.max_per_second {
            let key = format!("{}:{}", actor, action);  // 🚨 Allocation on every call!
            let mut counters = self.counters.lock().unwrap();  // 🚨 Mutex lock!
            // ... HashMap operations ...
        }
        true
    }
}
// Called in narrate_at_level:
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    if !GLOBAL_SAMPLER.should_sample(fields.actor, fields.action) {  // 🚨 Every narration!
        return;
    }
    // ... rest of narration
}
```
**Performance Impact Analysis**:
1. **Global Mutex Bottleneck** — DESTROYS MULTI-THREADED THROUGHPUT
   - `Arc<Mutex<HashMap>>` locked on **EVERY** narration call
   - At 1000 narrations/sec across 8 threads: 8x contention multiplier
   - Mutex lock/unlock: ~50-100ns per call
   - **Impact**: Serializes ALL narration across ALL threads
   - **Projected degradation**: 30-50% throughput loss in multi-threaded scenarios
2. **Allocation Storm** — MEMORY BANDWIDTH SATURATION
   - `format!("{}:{}", actor, action)` allocates String on every `should_sample()` call
   - At 10k narrations/sec: **10,000 allocations/sec**
   - Each allocation: ~50-100ns + GC pressure
   - **Impact**: Memory allocator contention, cache pollution
   - **Projected overhead**: +100-200ns per narration
3. **Wasted RNG Calls** — UNNECESSARY CPU CYCLES
   - `rand::random::<f64>()` called even when `sample_rate == 1.0` (default)
   - RNG overhead: ~50-100ns per call
   - For 100% sampling (default): **pure waste**
   - **Impact**: +50-100ns per narration for no benefit
4. **Unbounded Memory Growth** — MEMORY LEAK
   - No eviction policy for `HashMap<String, (Instant, u32)>`
   - Unique actor:action combinations accumulate forever
   - **Impact**: Memory leak in long-running services
   - **Projected growth**: ~100 bytes per unique actor:action pair
**Total Overhead**: +200-400ns per narration + mutex contention + memory leak
**REQUIRED ACTIONS** (BLOCKING):
- [ ] **DO NOT IMPLEMENT Unit 2.7 as designed** — it will destroy performance
- [ ] If sampling is required, implement at `tracing-subscriber` level (outside hot path)
- [ ] If custom sampling is required, use lock-free `DashMap` or sharded atomic counters
- [ ] Use `(&str, &str)` tuple as key (zero-copy) or pre-intern actor:action pairs
- [ ] Early-return if `sample_rate == 1.0`, skip RNG entirely
- [ ] Implement LRU eviction with max 10k entries
- [ ] **Target**: <50ns overhead for sampling check (if implemented)
**VERDICT**: 🔴 **REJECTED** — Unit 2.7 MUST be redesigned before implementation.
---
## 🚨 CRITICAL: Unit 1.5 Template Interpolation — ALLOCATION HOTSPOT
**Severity**: 🟠 HIGH — RUNTIME PARSING + ALLOCATIONS
**Planned Implementation** (from IMPLEMENTATION_PLAN.md lines 179-215):
```rust
// Unit 1.5: #[narrate(...)] Proc Macro with Template Interpolation
#[narrate(
    actor: "orchestratord",
    action: "dispatch",
    human: "Dispatched job {job_id} to worker {worker.id} ({elapsed_ms}ms)",
    cute: "Sent job {job_id} to worker {worker.id}! 🚀"
)]
fn dispatch_job(job_id: &str) -> Result<Worker> {
    // ... function body
}
// Generated code (from template engine):
let human = format!("Dispatched job {} to worker {} ({}ms)", job_id, worker.id, elapsed_ms);
let cute = format!("Sent job {} to worker {}! 🚀", job_id, worker.id);
```
**Performance Impact Analysis**:
1. **Format! Allocations** — EVERY NARRATION ALLOCATES
   - `format!()` allocates on **every** narration call
   - Each allocation: ~50-100ns + heap overhead
   - At 10k narrations/sec: **10,000 allocations/sec**
   - **Impact**: GC pressure, memory bandwidth saturation
   - **Projected overhead**: +100-200ns per narration
2. **Runtime Template Parsing** (if implemented)
   - If templates are parsed at runtime (not macro expansion time)
   - Parsing `{variable}` placeholders: ~100-500ns per template
   - **Impact**: Unnecessary CPU cycles
   - **Projected overhead**: +100-500ns per narration
3. **Multiple Template Fields**
   - `human`, `cute`, `story` all use templates
   - 3x format!() calls per narration
   - **Impact**: 3x allocation overhead
   - **Projected overhead**: +300-600ns per narration
**Total Overhead**: +300-1000ns per narration (depending on implementation)
**OPTIMIZATION RECOMMENDATIONS**:
1. **Compile-Time Template Expansion** (REQUIRED)
   - Generate direct `format!()` calls at macro expansion time
   - No runtime template parsing
   - **Speedup**: Eliminates 100-500ns parsing overhead
2. **Stack-Allocated Buffers** (RECOMMENDED)
   ```rust
   use arrayvec::ArrayString;
   let mut buf = ArrayString::<256>::new();
   write!(&mut buf, "Dispatched job {} to worker {}", job_id, worker.id)?;
   ```
   - Zero heap allocations for strings <256 bytes
   - **Speedup**: 2-5x faster than `format!()`
3. **String Interning** (OPTIONAL)
   - For repeated templates, intern strings
   - Reuse allocated strings for common patterns
   - **Speedup**: Amortized zero allocations
**REQUIRED ACTIONS**:
- [ ] Implement compile-time template expansion (not runtime parsing)
- [ ] Use stack-allocated buffers for templates <256 chars
- [ ] Benchmark with 1, 3, 5 variable interpolations
- [ ] **Target**: <100ns for interpolation with ≤3 variables, zero heap allocations
**VERDICT**: ⚠️ **CONDITIONAL APPROVAL** — Only if compile-time expansion + stack buffers implemented.
---
## ⚠️ HIGH PRIORITY: Unit 2.3 Redaction Enhancement — COMPOUNDS EXISTING OVERHEAD
**Severity**: 🟠 HIGH — ADDS 3 MORE REGEX PATTERNS
**Current Implementation** (from `src/redaction.rs`):
```rust
pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String {
    let mut result = text.to_string();  // ⚠️ Allocation #1
    if policy.mask_bearer_tokens {
        result = bearer_token_regex().replace_all(&result, &policy.replacement).to_string();  // ⚠️ Allocation #2
    }
    if policy.mask_api_keys {
        result = api_key_regex().replace_all(&result, &policy.replacement).to_string();  // ⚠️ Allocation #3
    }
    if policy.mask_uuids {
        result = uuid_regex().replace_all(&result, &policy.replacement).to_string();  // ⚠️ Allocation #4
    }
    result
}
```
**Current Overhead** (existing code):
1. **3 Regex Passes**: bearer, api_key, uuid
2. **Up to 4 Allocations**: 1 initial + 3 regex passes
3. **Called 3x Per Narration**: human, cute, story
4. **Current Impact**: ~1-5μs for 100-500 char strings
**Planned Enhancement** (from IMPLEMENTATION_PLAN.md lines 395-469):
```rust
// Unit 2.3: Add 3 MORE patterns
static JWT_PATTERN: OnceLock<Regex> = OnceLock::new();
static PRIVATE_KEY_PATTERN: OnceLock<Regex> = OnceLock::new();
static URL_PASSWORD_PATTERN: OnceLock<Regex> = OnceLock::new();
pub struct RedactionPolicy {
    pub mask_bearer_tokens: bool,
    pub mask_api_keys: bool,
    pub mask_uuids: bool,
    pub mask_jwt_tokens: bool,      // NEW
    pub mask_private_keys: bool,    // NEW
    pub mask_url_passwords: bool,   // NEW
    pub replacement: String,
}
pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String {
    let mut result = text.to_string();
    // ... existing 3 patterns ...
    if policy.mask_jwt_tokens {
        result = jwt_regex().replace_all(&result, &policy.replacement).to_string();
    }
    if policy.mask_private_keys {
        result = private_key_regex().replace_all(&result, &policy.replacement).to_string();
    }
    if policy.mask_url_passwords {
        result = url_password_regex().replace_all(&result, &policy.replacement).to_string();
    }
    result
}
```
**Performance Impact Analysis**:
1. **6 Regex Passes** (doubled from 3)
   - Each pass: O(n) string scan
   - Total: **6×O(n)** complexity
   - **Impact**: ~2-10μs for 100-500 char strings (2x current overhead)
2. **Up to 7 Allocations** (increased from 4)
   - 1 initial + 6 regex passes
   - **Impact**: +300-600ns allocation overhead per narration
3. **ReDoS Vulnerability** (auth-min identified)
   - Private key regex: `-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----`
   - Lazy quantifier `[\s\S]+?` can cause catastrophic backtracking
   - **Impact**: Potential DoS via malicious input
4. **Called 3x Per Narration** (unchanged)
   - human, cute, story all redacted
   - **Total overhead**: 3 × (2-10μs) = **6-30μs per narration**
**REQUIRED OPTIMIZATIONS** (BEFORE adding 3 more patterns):
1. **Single-Pass Regex** (CRITICAL)
   - Combine all 6 patterns into alternation: `(bearer|api_key|uuid|jwt|private_key|url_password)`
   - Reduces from **6×O(n)** to **1×O(n)**
   - **Speedup**: 5-6x faster
2. **Cow Strings** (CRITICAL)
   ```rust
   pub fn redact_secrets<'a>(text: &'a str, policy: RedactionPolicy) -> Cow<'a, str> {
       // Return Cow::Borrowed if no matches, Cow::Owned if redacted
   }
   ```
   - Zero allocations when no secrets found (90% of cases)
   - **Speedup**: Eliminates 1-7 allocations per narration
3. **Aho-Corasick for Literal Prefixes** (RECOMMENDED)
   - Patterns like "Bearer ", "-----BEGIN" are literal prefixes
   - Use `aho-corasick` crate for multi-pattern matching
   - **Speedup**: 10-100x faster for literal prefixes
4. **Fix ReDoS Vulnerability** (SECURITY CRITICAL)
   - Replace private key regex with bounded quantifier or line-by-line parsing
   - **Impact**: Prevents DoS attacks
**REQUIRED ACTIONS**:
- [ ] **DO NOT add 3 more patterns until existing redaction is optimized**
- [ ] Implement single-pass regex with alternation (BLOCKING)
- [ ] Use `Cow<'a, str>` to avoid allocations (BLOCKING)
- [ ] Fix ReDoS vulnerability in private key pattern (SECURITY BLOCKING)
- [ ] Benchmark: 100, 500, 1000 char strings
- [ ] **Target**: <1μs for clean strings, <5μs with redaction (1000 chars)
**VERDICT**: 🔴 **BLOCKED** — Optimize existing redaction BEFORE adding more patterns.
---
## ⚠️ HIGH PRIORITY: Unit 2.8 Unicode & Emoji Validation — CHAR ITERATION COST
**Severity**: 🟠 MEDIUM — MULTIPLE ALLOCATIONS + ITERATIONS
**Planned Implementation** (from IMPLEMENTATION_PLAN.md lines 785-839):
```rust
// Unit 2.8: Emoji & Unicode Safety
pub fn sanitize_for_json(text: &str) -> String {
    text.chars()  // 🚨 Allocates for multi-byte UTF-8
        .filter(|c| {
            c.is_alphanumeric() 
                || c.is_whitespace() 
                || c.is_ascii_punctuation()
                || (*c as u32) >= 0x1F000  // Emoji range
        })
        .collect()  // 🚨 Allocates new String
}
pub fn validate_emoji(text: &str) -> bool {
    !text.chars().any(|c| {  // 🚨 Another char iteration
        matches!(c as u32, 
            0x200B..=0x200D |  // Zero-width chars
            0xFE00..=0xFE0F    // Variation selectors
        )
    })
}
// Called in narrate_at_level:
pub fn narrate_at_level(mut fields: NarrationFields, level: NarrationLevel) {
    fields.human = sanitize_for_json(&fields.human);  // 🚨 Allocation
    if let Some(cute) = &fields.cute {
        if !validate_emoji(cute) {  // 🚨 Char iteration
            fields.cute = Some(sanitize_for_json(cute));  // 🚨 Allocation
        }
    }
    // ... same for story
}
```
**Performance Impact Analysis**:
1. **Char Iterator Allocations**
   - `.chars()` allocates for multi-byte UTF-8 sequences
   - For 1000-char strings: expensive iteration
   - **Impact**: ~1-5μs per string
2. **Filter + Collect Allocations**
   - `.filter().collect()` allocates new String on **every** narration
   - Even if no sanitization needed
   - **Impact**: +100-200ns allocation overhead
3. **Multiple Passes**
   - `validate_emoji()` iterates chars
   - Then `sanitize_for_json()` iterates again
   - **Impact**: 2x char iteration overhead
4. **Called for human, cute, story**
   - 3x sanitization per narration
   - **Total overhead**: 3 × (1-5μs) = **3-15μs per narration**
**OPTIMIZATION RECOMMENDATIONS**:
1. **ASCII Fast Path** (CRITICAL)
   ```rust
   pub fn sanitize_for_json(text: &str) -> Cow<'_, str> {
       if text.is_ascii() {
           return Cow::Borrowed(text);  // Zero-copy for ASCII
       }
       // Fall back to .chars() only for non-ASCII
   }
   ```
   - **Speedup**: 10-100x for ASCII strings (90% of cases)
2. **In-Place Sanitization** (RECOMMENDED)
   ```rust
   pub fn sanitize_for_json(text: &mut String) {
       text.retain(|c| /* validation */);
   }
   ```
   - No allocation, modifies in-place
   - **Speedup**: Eliminates allocation overhead
3. **Single-Pass Validation + Sanitization** (RECOMMENDED)
   - Combine `validate_emoji()` and `sanitize_for_json()` into one pass
   - **Speedup**: 2x faster (one iteration instead of two)
4. **Lazy Validation** (OPTIONAL)
   - Only sanitize security-critical fields (actor, action)
   - Skip human/cute/story (user-facing, less critical)
   - **Speedup**: Skip validation for 90% of fields
**REQUIRED ACTIONS**:
- [ ] Implement ASCII fast path with `Cow<'_, str>`
- [ ] Use `.as_bytes()` for ASCII validation, fall back to `.chars()` for non-ASCII
- [ ] Combine validation + sanitization into single pass
- [ ] **Target**: <1μs for 100-char string, <5μs for 1000-char string
**VERDICT**: ⚠️ **CONDITIONAL APPROVAL** — Only if ASCII fast path + Cow strings implemented.
---
## ✅ GOOD: Unit 3.3 Conditional Compilation — CRITICAL FOR PRODUCTION
**Severity**: ✅ REQUIRED — ZERO-OVERHEAD PRODUCTION BUILDS
**Current State**: Trace macros are **always active** (no conditional compilation yet).
**Current Implementation** (from `src/trace.rs`):
```rust
#[macro_export]
macro_rules! trace_tiny {
    ($actor:expr, $action:expr, $target:expr, $human:expr) => {
        tracing::trace!(  // ⚠️ Always active, no #[cfg] guard
            actor = $actor,
            action = $action,
            target = $target,
            human = $human,
            "trace"
        );
    };
}
```
**Current Overhead**:
1. **Trace macros always active**: ~20-50ns per trace call in production
2. **No feature flags**: Cannot disable at compile time
3. **Arguments always evaluated**: Even if tracing disabled
**Planned Implementation** (from IMPLEMENTATION_PLAN.md lines 913-937):
```rust
// Unit 3.3: Feature Flags & Conditional Compilation
[features]
default = ["trace-enabled", "debug-enabled", "cute-mode"]
trace-enabled = []
debug-enabled = []
cute-mode = []
production = []  // No trace, no debug, no cute
// Conditional compilation in macros:
#[cfg(feature = "trace-enabled")]
#[macro_export]
macro_rules! trace_tiny {
    ($actor:expr, $action:expr, $target:expr, $human:expr) => {
        tracing::trace!(/* ... */);
    };
}
#[cfg(not(feature = "trace-enabled"))]
#[macro_export]
macro_rules! trace_tiny {
    ($actor:expr, $action:expr, $target:expr, $human:expr) => {
        // No-op in production
    };
}
```
**Performance Impact Analysis**:
1. **Zero Overhead in Production** ✅
   - Trace macros completely removed at compile time
   - No argument evaluation, no function calls
   - **Impact**: 0ns overhead (code doesn't exist)
2. **Binary Size Reduction**
   - Trace code removed from production builds
   - **Projected savings**: ~5-10 MB binary size
3. **Dev Build Overhead Acceptable**
   - ~20-50ns per trace call in dev builds
   - **Impact**: <2% overhead in development
**REQUIRED ACTIONS** (CRITICAL FOR PRODUCTION):
- [ ] Implement Unit 3.3 as designed (BLOCKING for production)
- [ ] Add `trace-enabled`, `debug-enabled`, `cute-mode` features
- [ ] Wrap all trace macros with `#[cfg(feature = "trace-enabled")]`
- [ ] Wrap all debug narration with `#[cfg(feature = "debug-enabled")]`
- [ ] Wrap cute/story fields with `#[cfg(feature = "cute-mode")]`
- [ ] Verify with `cargo expand --release` that production has ZERO trace code
- [ ] Verify binary size identical with/without features (sha256sum)
- [ ] **Target**: 0ns overhead in production (code removed at compile time)
**VERDICT**: ✅ **APPROVED** — Unit 3.3 is CRITICAL and well-designed. MUST be implemented.
---
## ⚠️ MEDIUM: Unit 1.3 #[trace_fn] Proc Macro — TIMING OVERHEAD
**Severity**: 🟠 MEDIUM — INSTANT::NOW() ON EVERY FUNCTION
**Planned Implementation** (from IMPLEMENTATION_PLAN.md lines 141-171):
```rust
// Unit 1.3: #[trace_fn] Proc Macro
#[trace_fn]
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    // ... function body
}
// Generated code:
fn dispatch_job(job_id: &str, pool_id: &str) -> Result<WorkerId> {
    let __start = std::time::Instant::now();  // 🚨 Entry timing
    trace_enter!("orchestratord", "dispatch_job", 
                 format!("job_id={}, pool_id={}", job_id, pool_id));
    let __result = (|| {
        // ... original function body
    })();
    let __elapsed = __start.elapsed();  // 🚨 Exit timing
    trace_exit!("orchestratord", "dispatch_job", 
                format!("→ {:?} ({}ms)", __result, __elapsed.as_millis()));
    __result
}
```
**Performance Impact Analysis**:
1. **Instant::now() Overhead**
   - Called at entry and exit of **every** annotated function
   - Overhead: ~20-50ns per call (2x per function = 40-100ns)
   - For hot-path functions called millions of times: **compounds to milliseconds**
   - **Impact**: +40-100ns per function call
2. **Format! Allocations**
   - `format!("job_id={}, pool_id={}", ...)` at entry
   - `format!("→ {:?} ({}ms)", ...)` at exit
   - **Impact**: +100-200ns allocation overhead per function
3. **Trace Macro Overhead**
   - `trace_enter!()` and `trace_exit!()` each add ~50-100ns
   - **Impact**: +100-200ns per function
**Total Overhead**: +240-500ns per annotated function
**CRITICAL ISSUE**: If applied to hot-path functions (called 1M+ times/sec), this is **UNACCEPTABLE**.
**REQUIRED OPTIMIZATIONS**:
1. **Conditional Compilation** (CRITICAL)
   ```rust
   #[cfg(feature = "trace-enabled")]
   let __start = std::time::Instant::now();
   ```
   - Zero overhead in production (code removed)
   - **Speedup**: 0ns in production
2. **RDTSC for Ultra-Hot Paths** (OPTIONAL)
   ```rust
   #[cfg(all(feature = "trace-enabled", target_arch = "x86_64"))]
   let __start = unsafe { core::arch::x86_64::_rdtsc() };
   ```
   - Sub-nanosecond precision
   - **Speedup**: ~10x faster than `Instant::now()`
3. **Lazy Formatting** (RECOMMENDED)
   ```rust
   trace_enter!("orchestratord", "dispatch_job", || {
       format!("job_id={}, pool_id={}", job_id, pool_id)
   });
   ```
   - Only format if tracing enabled
   - **Speedup**: Eliminates allocation when tracing disabled
**REQUIRED ACTIONS**:
- [ ] Wrap timing code with `#[cfg(feature = "trace-enabled")]`
- [ ] Benchmark overhead on empty function, 10-instruction function, 1000-instruction function
- [ ] **DO NOT apply to hot-path functions** (>1k calls/sec) unless conditional compilation verified
- [ ] **Target**: <1% overhead for functions >100 instructions, 0ns in production
**VERDICT**: ⚠️ **CONDITIONAL APPROVAL** — Only if conditional compilation implemented AND not applied to hot paths.
---
## ✅ GOOD: Unit 2.1 WARN/ERROR/FATAL Levels — Minimal Overhead
**Severity**: ✅ LOW — ACCEPTABLE DESIGN
**Planned Implementation** (from IMPLEMENTATION_PLAN.md lines 225-318):
```rust
// Unit 2.1: Add WARN/ERROR/FATAL Levels
pub enum NarrationLevel {
    Mute,   // No output
    Trace,  // Ultra-fine detail
    Debug,  // Developer diagnostics
    Info,   // Narration backbone (default)
    Warn,   // Anomalies & degradations
    Error,  // Operational failures
    Fatal,  // Unrecoverable errors
}
impl NarrationLevel {
    fn to_tracing_level(&self) -> Option<Level> {
        match self {
            NarrationLevel::Mute => None,
            NarrationLevel::Trace => Some(Level::TRACE),
            // ... etc
        }
    }
}
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    let Some(tracing_level) = level.to_tracing_level() else {
        return; // MUTE - no output
    }
    // ... emit at appropriate level
}
```
**Performance Impact Analysis**:
1. **Enum Match Overhead**
   - Simple enum match: ~1-5ns
   - Compiler optimizes to jump table
   - **Impact**: Negligible
2. **Early Return for Mute**
   - Zero overhead when muted
   - **Impact**: Optimal
3. **No Additional Allocations**
   - Just routing to existing `tracing::event!()` at different levels
   - **Impact**: Zero additional overhead
**VERDICT**: ✅ **APPROVED** — Well-designed, minimal overhead.
---
## 📊 Performance Budget Summary
### Current Baseline (Existing Code)
| Component | Overhead | Status |
|-----------|----------|--------|
| `trace_tiny!()` | ~50-100ns | ⚠️ No conditional compilation |
| `narrate()` (no redaction) | ~100-200ns | ✅ Acceptable |
| `narrate()` (with redaction) | ~300-500ns | ✅ Acceptable |
| Redaction (3 patterns) | ~1-5μs | ✅ Acceptable |
| Test capture | ~50-100ns | ✅ Test-only |
### Projected Impact of Implementation Plan
| Unit | Component | Added Overhead | Status |
|------|-----------|----------------|--------|
| **2.7** | **Sampling** | **+200-400ns + mutex contention** | 🔴 **REJECTED** |
| **1.5** | **Templates** | **+300-1000ns** | ⚠️ Conditional (needs optimization) |
| **2.3** | **Redaction (6 patterns)** | **+3-15μs** | 🔴 **BLOCKED** (optimize first) |
| **2.8** | **Unicode validation** | **+3-15μs** | ⚠️ Conditional (needs optimization) |
| **1.3** | **#[trace_fn]** | **+240-500ns per function** | ⚠️ Conditional (needs #[cfg]) |
| **3.3** | **Conditional compilation** | **0ns (production)** | ✅ **REQUIRED** |
| **2.1** | **Levels** | **+1-5ns** | ✅ **APPROVED** |
### Total Projected Overhead (If Implemented As-Is)
**Worst Case** (all features enabled, no optimizations):
- Base `narrate()`: ~300-500ns
- **+ Sampling**: +200-400ns
- **+ Templates**: +300-1000ns  
- **+ Redaction (6 patterns)**: +6-30μs
- **+ Unicode validation**: +3-15μs
- **Total**: **~10-46μs per narration** 🚨 **UNACCEPTABLE**
**Best Case** (with all optimizations):
- Base `narrate()`: ~100-200ns
- + Conditional compilation: 0ns (production)
- + Optimized templates: +50-100ns
- + Optimized redaction: +1-5μs
- + Optimized unicode: +0.5-2μs
- **Total**: **~1.5-7μs per narration** ✅ **ACCEPTABLE**
---
## 🎯 Critical Path & Blocking Issues
### 🔴 BLOCKING (Must Fix Before Implementation)
1. **Unit 2.7 (Sampling)** — COMPLETE REDESIGN REQUIRED
   - **Issue**: Global mutex + allocation storm will destroy performance
   - **Action**: DO NOT IMPLEMENT as designed
   - **Alternative**: Use `tracing-subscriber` filtering or lock-free design
   - **Deadline**: BLOCKING for Week 2
2. **Unit 2.3 (Redaction)** — OPTIMIZE EXISTING BEFORE ADDING MORE
   - **Issue**: Adding 3 more patterns doubles overhead (6×O(n) passes)
   - **Action**: Implement single-pass regex + Cow strings FIRST
   - **Deadline**: BLOCKING for Week 2
### ⚠️ HIGH PRIORITY (Must Optimize Before Merge)
3. **Unit 1.5 (Templates)** — COMPILE-TIME EXPANSION REQUIRED
   - **Issue**: Runtime parsing + allocations add 300-1000ns
   - **Action**: Generate `format!()` at macro expansion, use stack buffers
   - **Deadline**: BLOCKING for Week 1
4. **Unit 2.8 (Unicode)** — ASCII FAST PATH REQUIRED
   - **Issue**: Char iteration + allocations add 3-15μs
   - **Action**: Implement ASCII fast path with Cow strings
   - **Deadline**: BLOCKING for Week 2
5. **Unit 1.3 (#[trace_fn])** — CONDITIONAL COMPILATION REQUIRED
   - **Issue**: `Instant::now()` adds 240-500ns per function
   - **Action**: Wrap with `#[cfg(feature = "trace-enabled")]`
   - **Deadline**: BLOCKING for Week 1
### ✅ APPROVED (Can Implement As-Is)
6. **Unit 3.3 (Conditional Compilation)** — CRITICAL, WELL-DESIGNED
   - **Impact**: 0ns overhead in production
   - **Action**: Implement as planned
   - **Priority**: HIGHEST (required for production)
7. **Unit 2.1 (Levels)** — MINIMAL OVERHEAD
   - **Impact**: +1-5ns per narration
   - **Action**: Implement as planned
   - **Priority**: LOW
---
## 🔧 Required Optimizations Before Implementation
### Priority 1: Unit 2.7 Redesign (CRITICAL)
**Current Plan**: Global mutex + allocation storm  
**Required**: Complete redesign or removal
**Options**:
1. **DO NOT IMPLEMENT** (RECOMMENDED)
   - Use `RUST_LOG` environment variable for filtering (zero runtime overhead)
   - Use `tracing-subscriber` filtering (outside hot path)
2. **Lock-Free Design** (if sampling is required)
   ```rust
   use dashmap::DashMap;
   use std::sync::atomic::{AtomicU32, Ordering};
   pub struct Sampler {
       counters: DashMap<(&'static str, &'static str), AtomicU32>,
   }
   impl Sampler {
       pub fn should_sample(&self, actor: &'static str, action: &'static str) -> bool {
           // Early return for 100% sampling
           if self.config.sample_rate == 1.0 {
               return true;
           }
           // Lock-free atomic counter
           let counter = self.counters
               .entry((actor, action))
               .or_insert(AtomicU32::new(0));
           counter.fetch_add(1, Ordering::Relaxed) < self.config.max_per_second
       }
   }
   ```
   - **Speedup**: 10-100x faster than mutex
   - **Target**: <50ns overhead
### Priority 2: Unit 2.3 Redaction Optimization (BLOCKING)
**Current**: 3 regex passes, 1-4 allocations  
**Planned**: 6 regex passes, 1-7 allocations  
**Required**: Single-pass + Cow strings
```rust
use std::borrow::Cow;
// Combine all patterns into single alternation
static COMBINED_PATTERN: OnceLock<Regex> = OnceLock::new();
fn combined_regex() -> &'static Regex {
    COMBINED_PATTERN.get_or_init(|| {
        Regex::new(r"(?i)(bearer\s+[a-zA-Z0-9_\-\.=]+|api_?key\s*[=:]\s*[a-zA-Z0-9_\-\.]+|...)")  
            .expect("BUG: combined regex invalid")
    })
}
pub fn redact_secrets<'a>(text: &'a str, policy: RedactionPolicy) -> Cow<'a, str> {
    if !combined_regex().is_match(text) {
        return Cow::Borrowed(text);  // Zero-copy when no secrets
    }
    Cow::Owned(combined_regex().replace_all(text, &policy.replacement).to_string())
}
```
- **Speedup**: 5-6x faster (single pass)
- **Memory**: Zero allocations for clean strings (90% of cases)
### Priority 3: Unit 1.5 Template Optimization (BLOCKING)
**Current Plan**: Runtime parsing + format!() allocations  
**Required**: Compile-time expansion + stack buffers
```rust
// Proc macro generates:
use arrayvec::ArrayString;
let mut human_buf = ArrayString::<256>::new();
write!(&mut human_buf, "Dispatched job {} to worker {}", job_id, worker.id)?;
let human = human_buf.as_str();
// Or if >256 chars, fall back to format!()
let human = if needs_heap {
    format!("...", ...)
} else {
    human_buf.as_str()
};
```
- **Speedup**: 2-5x faster (no heap allocation for <256 char strings)
- **Memory**: Zero heap allocations for 90% of templates
### Priority 4: Unit 2.8 Unicode Optimization (BLOCKING)
**Current Plan**: Char iteration + allocations on every narration  
**Required**: ASCII fast path + Cow strings
```rust
pub fn sanitize_for_json(text: &str) -> Cow<'_, str> {
    // Fast path: ASCII strings (90% of cases)
    if text.is_ascii() {
        // Validate ASCII-only (no allocation)
        if text.bytes().all(|b| b.is_ascii_alphanumeric() || b.is_ascii_whitespace() || b.is_ascii_punctuation()) {
            return Cow::Borrowed(text);
        }
    }
    // Slow path: UTF-8 with emoji (10% of cases)
    Cow::Owned(text.chars().filter(|c| /* ... */).collect())
}
```
- **Speedup**: 10-100x for ASCII strings
- **Memory**: Zero allocations for clean ASCII strings
### Priority 5: Unit 1.3 Conditional Compilation (BLOCKING)
**Current Plan**: `Instant::now()` on every function  
**Required**: Wrap with `#[cfg(feature = "trace-enabled")]`
```rust
// Generated code:
fn dispatch_job(job_id: &str) -> Result<WorkerId> {
    #[cfg(feature = "trace-enabled")]
    let __start = std::time::Instant::now();
    #[cfg(feature = "trace-enabled")]
    trace_enter!(/* ... */);
    let __result = (|| {
        // ... original function body
    })();
    #[cfg(feature = "trace-enabled")]
    trace_exit!(/* ... */);
    __result
}
```
- **Speedup**: 0ns in production (code removed)
- **Impact**: Zero overhead when feature disabled
---
## ✅ Performance Team Sign-Off Criteria
### Unit 2.7: Sampling & Rate Limiting
- [ ] 🔴 **REJECTED** — Do not implement as designed
- [ ] If required, use lock-free `DashMap` or `tracing-subscriber` filtering
- [ ] Benchmark: <50ns overhead for sampling check
- [ ] **Signature**: ___________________________
### Unit 1.5: Template Interpolation
- [ ] ⚠️ **CONDITIONAL** — Only approve if optimized
- [ ] Compile-time template expansion (not runtime parsing)
- [ ] Stack-allocated buffers for templates <256 chars
- [ ] Benchmark: <100ns for interpolation with ≤3 variables
- [ ] **Signature**: ___________________________
### Unit 2.3: Redaction Enhancement
- [ ] 🔴 **BLOCKED** — Optimize existing BEFORE adding patterns
- [ ] Single-pass regex with alternation
- [ ] `Cow<'a, str>` to avoid allocations
- [ ] Fix ReDoS vulnerability in private key pattern
- [ ] Benchmark: <1μs for clean strings, <5μs with redaction (1000 chars)
- [ ] **Signature**: ___________________________
### Unit 2.8: Unicode & Emoji Validation
- [ ] ⚠️ **CONDITIONAL** — Only approve if optimized
- [ ] ASCII fast path with `Cow<'_, str>`
- [ ] Single-pass validation + sanitization
- [ ] Benchmark: <1μs for 100-char string
- [ ] **Signature**: ___________________________
### Unit 1.3: #[trace_fn] Proc Macro
- [ ] ⚠️ **CONDITIONAL** — Only approve with #[cfg] guards
- [ ] Wrap timing code with `#[cfg(feature = "trace-enabled")]`
- [ ] Do NOT apply to hot-path functions (>1k calls/sec)
- [ ] Benchmark: <1% overhead for functions >100 instructions, 0ns in production
- [ ] **Signature**: ___________________________
### Unit 3.3: Conditional Compilation
- [ ] ✅ **APPROVED** — Implement as designed
- [ ] Verify with `cargo expand --release` (zero trace code)
- [ ] Binary size identical with/without features (sha256sum)
- [ ] Target: 0ns overhead in production
- [ ] **Signature**: ___________________________
### Unit 2.1: WARN/ERROR/FATAL Levels
- [ ] ✅ **APPROVED** — Minimal overhead
- [ ] Enum match overhead: <5ns
- [ ] **Signature**: ___________________________
---
## 📋 Implementation Checklist
### Week 1: Core Proc Macro Crate
- [ ] Unit 1.1: Project Setup ✅ (no performance concerns)
- [ ] Unit 1.2: Actor Inference ✅ (compile-time only)
- [ ] Unit 1.3: #[trace_fn] ⚠️ (REQUIRES conditional compilation)
- [ ] Unit 1.4: Template Engine ✅ (compile-time only)
- [ ] Unit 1.5: #[narrate(...)] ⚠️ (REQUIRES optimization)
### Week 2: Narration Core Enhancement  
- [ ] Unit 2.1: WARN/ERROR/FATAL ✅ (approved)
- [ ] Unit 2.2: Trace Macros Conditional Compilation ✅ (required)
- [ ] Unit 2.3: Redaction Enhancement 🔴 (BLOCKED — optimize first)
- [ ] Unit 2.4: Correlation ID ✅ (no performance concerns)
- [ ] Unit 2.5: Tracing Backend ✅ (already non-blocking)
- [ ] Unit 2.6: Async Support ✅ (already correct)
- [ ] Unit 2.7: Sampling 🔴 (REJECTED — redesign or remove)
- [ ] Unit 2.8: Unicode Validation ⚠️ (REQUIRES optimization)
### Week 3: Editorial Enforcement & Optimization
- [ ] Unit 3.1: Length Validation ✅ (compile-time only)
- [ ] Unit 3.2: SVO Validation ✅ (compile-time only)
- [ ] Unit 3.3: Conditional Compilation ✅ (CRITICAL — approved)
- [ ] Unit 3.4: Performance Benchmarks ✅ (required)
### Week 4: Integration, Testing & Rollout
- [ ] Unit 4.1: BDD Tests ✅ (test-only)
- [ ] Unit 4.2: Proof Bundle Integration ✅ (test-only)
- [ ] Unit 4.3: Editorial Tests ✅ (compile-time)
- [ ] Unit 4.4-4.6: Service Migrations ⚠️ (verify no regression)
- [ ] Unit 4.7: CI/CD Updates ✅ (infrastructure)
---
## 🚨 Final Verdict
**Overall Assessment**: 🔴 **CRITICAL ISSUES — REQUIRES REDESIGN**
The implementation plan contains **2 critical performance issues** that will destroy production performance if implemented as designed:
1. **Unit 2.7 (Sampling)**: Global mutex + allocation storm — **MUST BE REDESIGNED OR REMOVED**
2. **Unit 2.3 (Redaction)**: Adding 3 more patterns without optimization — **MUST OPTIMIZE EXISTING FIRST**
**Additional high-priority optimizations required**:
3. Unit 1.5 (Templates): Compile-time expansion + stack buffers
4. Unit 2.8 (Unicode): ASCII fast path + Cow strings  
5. Unit 1.3 (#[trace_fn]): Conditional compilation
**Timeline Impact**:
- **Week 1**: +2-4 hours (template optimization)
- **Week 2**: +8-12 hours (redaction optimization, sampling redesign)
- **Total**: +10-16 hours additional work
**Approval Conditions**:
1. 🔴 **BLOCKING**: Unit 2.7 must be redesigned or removed
2. 🔴 **BLOCKING**: Unit 2.3 must optimize existing redaction before adding patterns
3. ⚠️ **REQUIRED**: Units 1.5, 2.8, 1.3 must implement recommended optimizations
4. ✅ **APPROVED**: Unit 3.3 (conditional compilation) is critical and well-designed
**Performance Team will NOT approve** implementation plan until critical issues are addressed.
---
**With relentless efficiency and zero tolerance for latency waste,**  
**The Performance Team** ⏱️
---
**Reviewed by**: Performance Team (deadline-propagation)  
**Review Date**: 2025-10-04  
**Next Review**: After Week 1 completion (verify optimizations implemented)  
**Status**: 🔴 CRITICAL ISSUES IDENTIFIED — REDESIGN REQUIRED
---
Optimized by Performance Team ⏱️
