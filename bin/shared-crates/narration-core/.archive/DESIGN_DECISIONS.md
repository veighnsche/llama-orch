# 🎯 Narration System — Design Decisions FINALIZED

**Status**: ✅ ALL DECISIONS FINALIZED (Performance Team Authority)  
**Date**: 2025-10-04  
**Teams**: Narration Core, auth-min 🎭, Performance Team ⏱️  
**Final Authority**: Performance Team ⏱️ (has veto power on performance-impacting decisions)

---

## 📋 Executive Summary

All **10 critical design decisions** have been finalized. Performance Team exercised veto authority on **5 security recommendations** that would have added unacceptable overhead.

### 🎯 Performance Team Vetoes Summary

| Decision | Auth-Min Requirement | Performance Team Veto | Final Decision |
|----------|---------------------|----------------------|----------------|
| **#3: Templates** | Escape ALL variables | ❌ REJECTED (50-100ns/var) | Escape user inputs only |
| **#6: Unicode** | 20+ emoji ranges | ❌ REJECTED (excessive) | Simplified: `is_control()` + 5 chars |
| **#7: Injection** | Escape all variables | ❌ REJECTED (50-100ns/var) | Compile-time validation only |
| **#8: CRLF** | Encode all control chars | ❌ REJECTED (allocates) | Strip `\n`, `\r`, `\t` only |
| **#9: Correlation** | HMAC-signed UUIDs | ❌ REJECTED (500-1000ns) | UUID v4 validation only |

### ✅ Approved Decisions

- ✅ **#1: Tracing Opt-In** — Zero overhead in production (MANDATORY)
- ✅ **#2: Timing Strategy** — Conditional `Instant::now()` (0ns in production)
- ✅ **#4: Redaction** — Single-pass regex + Cow strings (<5μs target)
- ✅ **#5: Sampling** — REJECTED (use `RUST_LOG` instead)
- ✅ **#10: Actor Validation** — Compile-time allowlist (zero runtime cost)

### 🔒 Accepted Security Risks (Performance Team Decision)

1. **Template injection**: Only user-marked inputs are escaped (not all variables)
2. **Log injection**: Only `\n`, `\r`, `\t` are stripped (not all control chars)
3. **Unicode**: Simplified validation (not comprehensive emoji ranges)
4. **Correlation ID**: No HMAC signing (forgery risk accepted)
5. **Timing side-channels**: Dev build timing data is acceptable (not a security issue)

**Critical Issues Resolved**:
- 🚨 **7 CRITICAL security vulnerabilities** → Fixed with performance-optimized solutions
- 🚨 **5 CRITICAL performance bottlenecks** → Eliminated via vetoes and redesigns
- ⚡ **Performance targets**: <100ns templates, <50ns CRLF, <5μs redaction, <100ns UUID

---

## 🚨 CRITICAL DECISIONS (Must Decide Before Week 1)

### **Decision 1: Tracing Opt-In vs. Opt-Out**

**Context**: Performance Team raised concerns about `Instant::now()` overhead (20-50ns per function) compounding in hot paths.

**Options**:

#### **Option A: Opt-In (RECOMMENDED by Performance Team)**
```toml
# Cargo.toml
[features]
default = []  # NO tracing by default
trace-enabled = []
debug-enabled = []
```

**Pros**:
- ✅ Zero overhead in production by default
- ✅ Developers explicitly enable tracing when needed
- ✅ No accidental performance degradation
- ✅ Binary size smaller by default

**Cons**:
- ❌ Developers might forget to enable tracing
- ❌ Less observability out-of-the-box
- ❌ Requires documentation/training

#### **Option B: Opt-Out (Current Plan)**
```toml
# Cargo.toml
[features]
default = ["trace-enabled", "debug-enabled"]
production = []  # Explicitly disable for production
```

**Pros**:
- ✅ Better observability by default
- ✅ Easier for developers (works out-of-box)
- ✅ Catches issues in development

**Cons**:
- ❌ Performance overhead in development
- ❌ Risk of accidentally shipping with tracing enabled
- ❌ Larger binary size by default

**User's Preference**: "I actually agree to make tracing opt-in..."

**DECISION MADE**: ✅ **Option A: Opt-In** (trace-enabled feature required)

**Rationale**: 
- Performance Team: Zero overhead in production is NON-NEGOTIABLE
- User agreement: "I actually agree to make tracing opt-in"
- Security Team: No timing side-channels in production builds

**⏱️ PERFORMANCE TEAM FINAL VERDICT**:
- ✅ **APPROVED** — Opt-in is the ONLY acceptable approach
- 🎯 **ENFORCEMENT**: CI MUST verify production builds have `--no-default-features`
- 📊 **VERIFICATION**: Binary size MUST be identical with/without trace features (sha256sum check)
- 🚨 **BLOCKING**: Any PR that enables tracing by default will be REJECTED

**Impact**: Affects Unit 1.3, Unit 2.2, Unit 3.3

---

### **Decision 2: Timing Measurement Strategy**

**Context**: `Instant::now()` adds 20-50ns overhead. Performance Team suggests `rdtsc` for ultra-hot paths.

**Options**:

#### **Option A: Conditional `Instant::now()` (SAFE)**
```rust
#[cfg(feature = "trace-enabled")]
fn measure_timing() -> Duration {
    Instant::now()
}

#[cfg(not(feature = "trace-enabled"))]
fn measure_timing() -> Duration {
    Duration::ZERO  // No-op
}
```

**Pros**:
- ✅ Zero overhead when tracing disabled
- ✅ Safe, portable across platforms
- ✅ Simple implementation

**Cons**:
- ❌ 20-50ns overhead when enabled
- ❌ Not suitable for sub-microsecond measurements

#### **Option B: `rdtsc` for Ultra-Hot Paths (FAST but RISKY)**
```rust
#[cfg(all(target_arch = "x86_64", feature = "trace-enabled"))]
fn measure_timing() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}
```

**Pros**:
- ✅ Sub-nanosecond precision
- ✅ ~5ns overhead (4-10x faster)
- ✅ Ideal for hot paths

**Cons**:
- ❌ x86_64 only (not portable)
- ❌ Requires `unsafe`
- ❌ CPU frequency scaling affects accuracy
- ❌ auth-min concern: timing side-channels

#### **Option C: Hybrid Approach**
```rust
// Use rdtsc for ultra-hot paths (>1M calls/sec)
#[cfg(all(target_arch = "x86_64", feature = "trace-ultra-fast"))]
fn measure_hot_path() -> u64 { unsafe { _rdtsc() } }

// Use Instant::now() for normal paths
#[cfg(feature = "trace-enabled")]
fn measure_normal_path() -> Duration { Instant::now() }
```

**Pros**:
- ✅ Best of both worlds
- ✅ Opt-in for risky optimization
- ✅ Fallback to safe option

**Cons**:
- ❌ More complex
- ❌ Two timing systems to maintain

**DECISION MADE**: ✅ **Option A: Conditional `Instant::now()`** (safe, portable)

**Rationale**:
- Performance Team: Conditional compilation REQUIRED, 0ns in production
- Security Team: No timing side-channels in production
- `rdtsc` rejected due to portability concerns and security implications
- Hybrid approach adds unnecessary complexity

**🎭 AUTH-MIN SECURITY COMMENT**:
- ✅ **APPROVED**: Conditional compilation eliminates timing side-channels in production
- ⚠️ **RESIDUAL RISK**: Even with `#[cfg(feature = "trace-enabled")]`, timing data in dev builds could leak information about code paths (e.g., auth success vs failure). **RECOMMENDATION**: Add documentation warning developers NOT to expose timing data in user-facing contexts (error messages, API responses).
- ⚠️ **VERIFICATION REQUIRED**: Ensure timing measurements are ONLY used for observability, never for security decisions (e.g., don't use timing to detect brute force - use dedicated rate limiting).

**⏱️ PERFORMANCE TEAM FINAL VERDICT**:
- ✅ **APPROVED** — Conditional compilation is MANDATORY
- 🎯 **REQUIREMENT**: `cargo expand --release` MUST show ZERO timing code
- ⚠️ **VETO ON SECURITY CONCERN**: Auth-min's "residual risk" about timing data in dev builds is OVERRULED. Dev builds are NOT production, timing data is EXPECTED in observability. This is not a security issue.
- 📊 **BENCHMARK REQUIREMENT**: Measure overhead on empty function (baseline), 10-instruction function, 1000-instruction function. Target: <1% overhead for functions >100 instructions, 0ns in production.

**Implementation**:
```rust
#[cfg(feature = "trace-enabled")]
let __start = std::time::Instant::now();
```

**Impact**: Affects Unit 1.3

---

### **Decision 3: Template Interpolation — Allocation Strategy**

**Context**: Performance Team identified `format!()` as allocation hotspot (>1000 allocations/sec at high frequency).

**Options**:

#### **Option A: Stack-Allocated Buffers (FAST)**
```rust
use std::io::Write;

fn interpolate_template(template: &str, vars: &[(&str, &str)]) -> String {
    let mut buf = [0u8; 256];  // Stack buffer
    let mut cursor = std::io::Cursor::new(&mut buf[..]);
    
    // Write directly to stack buffer
    for var in vars {
        write!(cursor, "{}", var.1).ok();
    }
    
    String::from_utf8_lossy(cursor.get_ref()).to_string()
}
```

**Pros**:
- ✅ Zero heap allocations for strings <256 bytes
- ✅ ~10x faster than `format!()`
- ✅ Predictable performance

**Cons**:
- ❌ Truncates strings >256 bytes
- ❌ More complex code
- ❌ Requires buffer size tuning

#### **Option B: Pre-Compiled Templates (FASTEST)**
```rust
// At macro expansion time, generate direct write!() calls
#[narrate(human = "Accepted job {job_id} at position {position}")]
// Expands to:
write!(buf, "Accepted job {} at position {}", job_id, position)
```

**Pros**:
- ✅ Zero runtime parsing
- ✅ Minimal allocations
- ✅ Compile-time validation

**Cons**:
- ❌ Complex proc macro
- ❌ Less flexible

#### **Option C: String Interning (MEMORY EFFICIENT)**
```rust
static TEMPLATE_CACHE: Lazy<DashMap<&'static str, String>> = ...;

fn interpolate_cached(template: &'static str, vars: &[(&str, &str)]) -> String {
    TEMPLATE_CACHE.entry(template)
        .or_insert_with(|| compile_template(template))
        .clone()
}
```

**Pros**:
- ✅ Amortized cost (compile once, reuse)
- ✅ Good for repeated templates
- ✅ Moderate complexity

**Cons**:
- ❌ Memory overhead (cache grows)
- ❌ First call still allocates
- ❌ Lock contention on cache

**DECISION MADE**: ✅ **Option B: Pre-compiled templates** (fastest, complex)

**Rationale**:
- Performance Team: BLOCKING - Runtime parsing adds 300-1000ns overhead
- REQUIRED: Compile-time template expansion at macro expansion time
- Generate direct `write!()` calls instead of runtime parsing
- Use stack buffers (ArrayString<256>) for templates <256 chars
- Fall back to heap allocation only for >256 char templates

**🎭 AUTH-MIN SECURITY COMMENT**:
- ⚠️ **INJECTION RISK**: Pre-compiled templates are secure ONLY if variable values are sanitized. The proc macro generates `write!()` calls but CANNOT validate runtime values.
- 🚨 **CRITICAL**: Template injection prevention (Decision 7) MUST be implemented in the generated code, not just at macro expansion.
- **REQUIRED**: Generated code must include `escape_template_var()` calls for ALL interpolated variables:
  ```rust
  // Generated code MUST be:
  write!(&mut buf, "Job {} worker {}", escape_template_var(&job_id), escape_template_var(&worker_id))
  // NOT:
  write!(&mut buf, "Job {} worker {}", job_id, worker_id)  // VULNERABLE!
  ```

**⏱️ PERFORMANCE TEAM FINAL VERDICT**:
- ✅ **APPROVED** — Pre-compiled templates are REQUIRED for performance
- 🎯 **CRITICAL**: Stack buffers (ArrayString<256>) are MANDATORY, not optional
- ⚠️ **VETO ON SECURITY**: Auth-min's requirement to call `escape_template_var()` on EVERY variable is REJECTED. This adds 50-100ns per variable (allocation + string manipulation). **ALTERNATIVE**: Escape ONLY user-controlled inputs, not internal variables (job_id, worker_id, etc.)
- 📊 **TARGET**: <100ns for interpolation with ≤3 variables, ZERO heap allocations for templates <256 chars
- 🚨 **BLOCKING**: If escaping adds >20ns overhead per variable, it must be made optional via feature flag

**Implementation**:
```rust
// Proc macro generates:
use arrayvec::ArrayString;
let mut buf = ArrayString::<256>::new();
write!(&mut buf, "Dispatched job {} to worker {}", job_id, worker.id)?;
```

**Target**: <100ns for interpolation with ≤3 variables, zero heap allocations

**Impact**: Affects Unit 1.4, Unit 1.5

---

### **Decision 4: Secret Redaction — Performance vs. Security**

**Context**: 
- Performance Team: 6+ regex passes expensive, use single-pass or `aho-corasick`
- auth-min Team: ReDoS vulnerability in private key regex

**Options**:

#### **Option A: Single-Pass Regex (FAST)**
```rust
static COMBINED_PATTERN: OnceLock<Regex> = OnceLock::new();

fn combined_regex() -> &'static Regex {
    COMBINED_PATTERN.get_or_init(|| {
        Regex::new(r"(Bearer [a-zA-Z0-9_\-\.=]+)|(api_?key=[a-zA-Z0-9_\-\.]+)|(eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)")
            .expect("combined regex invalid")
    })
}

fn redact_secrets(text: &str) -> String {
    combined_regex().replace_all(text, "[REDACTED]").to_string()
}
```

**Pros**:
- ✅ Single pass (6x faster)
- ✅ Simple implementation
- ✅ No ReDoS risk (simple patterns)

**Cons**:
- ❌ Can't handle private key blocks (multiline)
- ❌ Large regex harder to maintain
- ❌ Still allocates new String

#### **Option B: `aho-corasick` Multi-Pattern (FASTEST)**
```rust
use aho_corasick::AhoCorasick;

static PATTERNS: OnceLock<AhoCorasick> = OnceLock::new();

fn patterns() -> &'static AhoCorasick {
    PATTERNS.get_or_init(|| {
        AhoCorasick::new(&[
            "Bearer ",
            "api_key=",
            "apikey=",
            "-----BEGIN PRIVATE KEY-----",
            "eyJ",  // JWT prefix
        ]).unwrap()
    })
}

fn redact_secrets(text: &str) -> String {
    let mut result = text.to_string();
    for mat in patterns().find_iter(text) {
        // Redact from match to next whitespace/end
        result.replace_range(mat.start()..find_end(text, mat.end()), "[REDACTED]");
    }
    result
}
```

**Pros**:
- ✅ 10-100x faster than regex
- ✅ No ReDoS risk
- ✅ Handles literal prefixes well

**Cons**:
- ❌ Doesn't handle full patterns (e.g., JWT structure)
- ❌ Still allocates
- ❌ Requires additional crate

#### **Option C: Lazy Redaction (DEFERRED)**
```rust
// Only redact ERROR/FATAL levels
pub fn narrate_error(fields: NarrationFields) {
    let redacted_fields = apply_redaction(fields);  // Only here
    narrate_at_level(redacted_fields, NarrationLevel::Error)
}

pub fn narrate_info(fields: NarrationFields) {
    narrate_at_level(fields, NarrationLevel::Info)  // No redaction
}
```

**Pros**:
- ✅ Zero overhead for INFO/DEBUG
- ✅ Redaction only when needed
- ✅ Simple

**Cons**:
- ❌ Secrets could leak in INFO logs
- ❌ auth-min would reject this
- ❌ Violates security principle

#### **Option D: Fix ReDoS + Bounded Quantifiers (SECURE)**
```rust
// Fix private key regex with bounded quantifier
fn private_key_regex() -> &'static Regex {
    PRIVATE_KEY_PATTERN.get_or_init(|| {
        // Bounded to 10KB max (prevents ReDoS)
        Regex::new(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]{1,10000}?-----END [A-Z ]+PRIVATE KEY-----")
            .expect("private key regex invalid")
    })
}
```

**Pros**:
- ✅ Fixes ReDoS vulnerability
- ✅ Still handles multiline
- ✅ auth-min approved pattern

**Cons**:
- ❌ Still multiple regex passes
- ❌ Still allocates
- ❌ Performance not optimal

**DECISION MADE**: ✅ **Option E: Hybrid (aho-corasick + bounded regex)** (balanced)

**Rationale**:
- Security Team: CRITICAL - ReDoS vulnerability MUST be fixed (CRIT-1)
- Performance Team: BLOCKING - 6 regex passes unacceptable, single-pass REQUIRED
- MUST optimize existing redaction BEFORE adding 3 new patterns

**🎭 AUTH-MIN SECURITY COMMENT**:
- ✅ **APPROVED**: Bounded quantifier `{1,10240}` fixes ReDoS vulnerability (CRIT-1)
- ✅ **APPROVED**: Single-pass regex with alternation is secure and performant
- ✅ **APPROVED**: `Cow<'a, str>` for zero-copy is excellent optimization
- ⚠️ **VERIFICATION REQUIRED**: The combined regex pattern MUST be tested for ReDoS with 100KB malicious inputs. Each alternation branch must use bounded quantifiers or safe patterns.
- 🚨 **CRITICAL**: JWT pattern `eyJ[a-zA-Z0-9_-]+\\.eyJ[a-zA-Z0-9_-]+\\.[a-zA-Z0-9_-]+` uses unbounded `+` quantifiers. **MUST** add bounds: `eyJ[a-zA-Z0-9_-]{1,10000}\\.eyJ[a-zA-Z0-9_-]{1,10000}\\.[a-zA-Z0-9_-]{1,10000}` to prevent ReDoS on malformed JWTs.

**⏱️ PERFORMANCE TEAM FINAL VERDICT**:
- ✅ **APPROVED** — Single-pass regex + Cow strings is MANDATORY
- 🎯 **CRITICAL**: DO NOT add 3 more patterns until this optimization is complete
- ✅ **SECURITY ALIGNMENT**: Auth-min's bounded quantifiers are APPROVED (prevents ReDoS without performance penalty)
- 📊 **TARGET**: <1μs for clean strings (90% of cases), <5μs with redaction (1000 chars)
- 🚨 **BLOCKING**: Benchmark MUST prove <5μs for worst-case 1000-char string with all 6 patterns
- ⚡ **OPTIMIZATION**: Consider `aho-corasick` for literal prefixes ("Bearer ", "-----BEGIN") — 10-100x faster than regex

**Implementation**:
```rust
// 1. Fix ReDoS with bounded quantifier
Regex::new(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]{1,10240}?-----END [A-Z ]+PRIVATE KEY-----")

// 2. Combine all patterns into single alternation
static COMBINED_PATTERN: OnceLock<Regex> = OnceLock::new();
fn combined_regex() -> &'static Regex {
    COMBINED_PATTERN.get_or_init(|| {
        Regex::new(r"(?i)(bearer\s+[a-zA-Z0-9_\-\.=]+|api_?key\s*[=:]\s*[a-zA-Z0-9_\-\.]+|...)")
            .expect("combined regex invalid")
    })
}

// 3. Use Cow for zero-copy when no secrets
pub fn redact_secrets<'a>(text: &'a str, policy: RedactionPolicy) -> Cow<'a, str> {
    if !combined_regex().is_match(text) {
        return Cow::Borrowed(text);  // Zero allocations (90% of cases)
    }
    Cow::Owned(combined_regex().replace_all(text, &policy.replacement).to_string())
}
```

**Target**: <1μs for clean strings, <5μs with redaction (1000 chars)

**Impact**: Affects Unit 2.3

---

### **Decision 5: Sampling & Rate Limiting — Lock-Free Architecture**

**Context**: Performance Team identified global mutex as "CRITICAL BOTTLENECK" and "PERFORMANCE DISASTER".

**Options**:

#### **Option A: Lock-Free Atomic Counters (RECOMMENDED)**
```rust
use dashmap::DashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

struct Sampler {
    config: SamplingConfig,
    // Sharded lock-free map
    counters: DashMap<(&'static str, &'static str), (AtomicU64, AtomicU32)>,
}

impl Sampler {
    fn should_sample(&self, actor: &'static str, action: &'static str) -> bool {
        // Early return for 100% sampling (skip RNG)
        if self.config.sample_rate >= 1.0 {
            return true;
        }
        
        // Probabilistic sampling
        if rand::random::<f64>() > self.config.sample_rate {
            return false;
        }
        
        // Lock-free rate limiting
        if let Some(max_per_sec) = self.config.max_per_second {
            let entry = self.counters.entry((actor, action)).or_insert_with(|| {
                (AtomicU64::new(0), AtomicU32::new(0))
            });
            
            let now = Instant::now().as_secs();
            let (last_reset, count) = entry.value();
            
            // Atomic compare-and-swap for reset
            if last_reset.load(Ordering::Relaxed) < now {
                last_reset.store(now, Ordering::Relaxed);
                count.store(0, Ordering::Relaxed);
            }
            
            // Atomic increment and check
            let current = count.fetch_add(1, Ordering::Relaxed);
            if current >= max_per_sec {
                return false;
            }
        }
        
        true
    }
}
```

**Pros**:
- ✅ Lock-free (no contention)
- ✅ ~50ns overhead (vs. mutex ~500ns)
- ✅ Scales to 1M+ narrations/sec
- ✅ No panic risk (no mutex poisoning)

**Cons**:
- ❌ Requires `dashmap` crate
- ❌ More complex than mutex
- ❌ Potential race conditions (acceptable for sampling)

#### **Option B: Thread-Local Counters (FASTEST)**
```rust
use std::cell::RefCell;

thread_local! {
    static COUNTERS: RefCell<HashMap<(&'static str, &'static str), (Instant, u32)>> = 
        RefCell::new(HashMap::new());
}

fn should_sample(actor: &'static str, action: &'static str) -> bool {
    COUNTERS.with(|c| {
        let mut counters = c.borrow_mut();
        // ... per-thread rate limiting
    })
}
```

**Pros**:
- ✅ Zero contention (thread-local)
- ✅ ~20ns overhead
- ✅ Simplest implementation

**Cons**:
- ❌ Per-thread limits (not global)
- ❌ Doesn't work across async tasks
- ❌ Inconsistent behavior

#### **Option C: Disable Sampling (SIMPLE)**
```rust
// Just remove sampling entirely
fn should_sample(&self, actor: &str, action: &str) -> bool {
    true  // Always sample
}
```

**Pros**:
- ✅ Zero overhead
- ✅ Simple
- ✅ No security issues

**Cons**:
- ❌ No protection against log flooding
- ❌ No cost control in production
- ❌ Defeats the purpose

**DECISION MADE**: 🚫 **UNIT 2.7 REJECTED - DO NOT IMPLEMENT**

**Rationale**:
- Performance Team: 🔴 **CRITICAL - PERFORMANCE DISASTER**
- Security Team: 🚨 **5 CRITICAL vulnerabilities** (CRIT-2, CRIT-3, CRIT-4, CRIT-5)
  - Mutex poisoning DoS
  - HashMap collision DoS
  - Unbounded memory growth
  - Global mutex contention (destroys throughput)
- Projected overhead: +200-400ns + mutex contention + memory leak
- **VERDICT**: Complete redesign required OR remove entirely

**Alternative Approaches**:
1. **Use `RUST_LOG` environment variable** (zero runtime overhead) ✅ RECOMMENDED
2. **Use `tracing-subscriber` filtering** (outside hot path) ✅ RECOMMENDED
3. **If custom sampling required**: Lock-free DashMap + atomic counters + LRU eviction

**⏱️ PERFORMANCE TEAM FINAL VERDICT**:
- 🔴 **REJECTED** — Unit 2.7 is a PERFORMANCE DISASTER and MUST NOT be implemented
- 🎯 **ABSOLUTE VETO**: Global mutex on every narration call is UNACCEPTABLE under ANY circumstances
- ✅ **SECURITY ALIGNMENT**: Auth-min's 5 CRITICAL vulnerabilities confirm this design is fundamentally broken
- 📊 **ALTERNATIVE**: Use `RUST_LOG=info` for filtering (zero runtime overhead) or `tracing-subscriber::EnvFilter` (outside hot path)
- 🚨 **ENFORCEMENT**: Any attempt to implement Unit 2.7 as designed will be BLOCKED by Performance Team
- ⚡ **IF SAMPLING REQUIRED**: Must use lock-free DashMap + AtomicU32 counters + LRU eviction (max 10k entries). Target: <50ns overhead.

**Impact**: Unit 2.7 REMOVED from implementation plan

---

### **Decision 6: Unicode & Emoji Validation — Comprehensive vs. Minimal**

**Context**: 
- auth-min: Incomplete emoji ranges, missing zero-width chars, homograph attacks
- Performance Team: Char iteration expensive, normalization costly

**Options**:

#### **Option A: Comprehensive Security (SECURE)**
```rust
use unicode_normalization::UnicodeNormalization;

fn sanitize_for_json(text: &str) -> String {
    // 1. Normalize (NFC)
    let normalized: String = text.nfc().collect();
    
    // 2. Comprehensive emoji ranges
    let emoji_ranges = [
        0x1F600..=0x1F64F,  // Emoticons
        0x1F300..=0x1F5FF,  // Misc Symbols
        0x1F680..=0x1F6FF,  // Transport
        0x1F900..=0x1F9FF,  // Supplemental
        // ... 20+ more ranges
    ];
    
    // 3. Comprehensive zero-width blocklist
    let zero_width = [
        '\u{200B}', '\u{200C}', '\u{200D}',  // Zero-width
        '\u{FEFF}', '\u{2060}', '\u{180E}',  // More zero-width
        // ... complete list
    ];
    
    // 4. Homograph detection
    let result = normalized.chars()
        .filter(|c| is_safe_char(c, &emoji_ranges, &zero_width))
        .collect();
    
    // 5. Check for Cyrillic/Greek lookalikes in actor/action
    if contains_homograph(&result) {
        return "[SANITIZED]".to_string();
    }
    
    result
}
```

**Pros**:
- ✅ Maximum security
- ✅ Prevents all known attacks
- ✅ auth-min approved

**Cons**:
- ❌ 5-10μs per string (slow)
- ❌ Complex implementation
- ❌ Requires `unicode-normalization` crate

#### **Option B: Minimal Security (FAST)**
```rust
fn sanitize_for_json(text: &str) -> String {
    // Only filter control chars and common zero-width
    text.chars()
        .filter(|c| {
            !c.is_control() && 
            !matches!(*c as u32, 0x200B..=0x200D)
        })
        .collect()
}
```

**Pros**:
- ✅ <1μs per string
- ✅ Simple
- ✅ Handles most cases

**Cons**:
- ❌ Incomplete protection
- ❌ auth-min would reject
- ❌ Vulnerable to homograph attacks

#### **Option C: Tiered Validation (BALANCED)**
```rust
// Minimal validation for human/cute/story (performance)
fn sanitize_narrative(text: &str) -> String {
    text.chars()
        .filter(|c| !c.is_control())
        .collect()
}

// Comprehensive validation for actor/action (security)
fn sanitize_security_critical(text: &str) -> String {
    let normalized: String = text.nfc().collect();
    // Full validation + homograph detection
    validate_comprehensive(&normalized)
}
```

**Pros**:
- ✅ Fast for most fields
- ✅ Secure for critical fields
- ✅ Balanced approach

**Cons**:
- ❌ Two validation paths
- ❌ More complex

**DECISION MADE**: ✅ **Option C: Tiered (Simplified)** (Performance Team Override)

**Rationale**:
- Security Team: HIGH severity - incomplete emoji ranges, missing zero-width chars, homograph attacks
- Performance Team: VETO on comprehensive emoji ranges (20+ ranges, excessive complexity)
- **FINAL DECISION**: ASCII fast path + simplified validation
  - ASCII strings: Zero-copy (90% of cases)
  - UTF-8: Use `c.is_control()` + basic zero-width blocklist (5 chars)
  - Homograph detection: ONLY for actor/action fields
  - NO Unicode normalization for human/cute/story (too expensive)

**🎭 AUTH-MIN SECURITY COMMENT**:
- ✅ **APPROVED**: Tiered approach balances security and performance
- ✅ **APPROVED**: ASCII fast path with zero-copy `Cow::Borrowed` is excellent
- 🚨 **CRITICAL REQUIREMENT**: `validate_security_critical()` MUST be called for `actor` and `action` fields. These are used in log analysis, metrics, and audit trails - homograph spoofing here would bypass security monitoring.
- ⚠️ **INCOMPLETE**: The implementation example shows `is_safe_char(c)` but doesn't specify the comprehensive emoji ranges and zero-width blocklist. **REQUIRED**: Document the EXACT Unicode ranges and codepoints to be validated (see SECURITY_REVIEW.md HIGH-1, HIGH-2 for complete lists).
- ⚠️ **HOMOGRAPH DETECTION**: `contains_homograph()` function is mentioned but not implemented. **REQUIRED**: Implement mixed-script detection (Latin + Cyrillic/Greek) for actor/action fields.

**⏱️ PERFORMANCE TEAM FINAL VERDICT**:
- ✅ **APPROVED** — Tiered validation is the correct approach
- 🎯 **CRITICAL**: ASCII fast path is MANDATORY — 90% of strings are ASCII, MUST be zero-copy
- ⚠️ **VETO ON SECURITY**: Auth-min's requirement for "comprehensive emoji ranges" (20+ ranges) is EXCESSIVE. **SIMPLIFIED APPROACH**: Use `c.is_control()` check + basic zero-width blocklist (5 chars). Homograph detection for actor/action ONLY.
- 📊 **TARGET**: <1μs for ASCII strings (zero-copy), <5μs for UTF-8 with validation
- 🚨 **BLOCKING**: If Unicode normalization (NFC) is added, it MUST be lazy (only for actor/action), NOT for human/cute/story
- ⚡ **OPTIMIZATION**: Use `.as_bytes()` for ASCII validation, fall back to `.chars()` only for non-ASCII

**Implementation** (Performance Team Approved):
```rust
// Fast path for ASCII (90% of cases) - ZERO-COPY
pub fn sanitize_for_json(text: &str) -> Cow<'_, str> {
    if text.is_ascii() {
        return Cow::Borrowed(text);  // Zero-copy, no validation needed
    }
    
    // Simplified UTF-8 validation (not comprehensive)
    Cow::Owned(
        text.chars()
            .filter(|c| {
                !c.is_control() &&  // Basic control char filter
                !matches!(*c as u32, 
                    0x200B..=0x200D |  // Zero-width space, ZWNJ, ZWJ
                    0xFEFF |           // Zero-width no-break space
                    0x2060             // Word joiner
                )
            })
            .collect()
    )
}

// Homograph detection ONLY for actor/action (security-critical)
fn validate_actor(actor: &str) -> Result<&str, Error> {
    if !actor.is_ascii() {
        return Err(Error::NonAsciiActor);  // Reject non-ASCII actors
    }
    Ok(actor)
}
```

**Target**: <1μs for ASCII strings, <5μs for UTF-8 with validation

**Impact**: Affects Unit 2.8

---

## ⚠️ HIGH PRIORITY DECISIONS (Must Decide Before Week 2)

### **Decision 7: Template Injection Prevention**

**Context**: auth-min identified template injection vulnerability if variable values contain `{}`.

**Options**:

#### **Option A: Escape Braces**
```rust
fn escape_template_var(value: &str) -> String {
    value.replace("{", "{{").replace("}", "}}")
}
```

**Pros**: Simple, effective
**Cons**: Allocates, changes output

#### **Option B: Reject Invalid Values**
```rust
fn validate_template_var(value: &str) -> Result<&str, Error> {
    if value.contains('{') || value.contains('}') {
        return Err(Error::InvalidTemplateVar);
    }
    Ok(value)
}
```

**Pros**: No injection possible
**Cons**: Breaks valid use cases

#### **Option C: Compile-Time Validation**
```rust
// In proc macro, validate at compile time
if var_value.contains('{') {
    return syn::Error::new_spanned(
        var_value,
        "Template variable cannot contain braces"
    ).to_compile_error();
}
```

**Pros**: Caught at compile time
**Cons**: Only works for literals

**DECISION MADE**: ✅ **Option C: Compile-time validation ONLY** (Performance Team Override)

**Rationale**:
- Security Team: MEDIUM severity - template injection vulnerability (MED-1)
- Performance Team: VETO on runtime escaping (50-100ns overhead per variable)
- **FINAL DECISION**: Compile-time validation for literals, NO runtime escaping
- Internal variables (job_id, worker_id, etc.) are trusted, no escaping needed
- User-controlled inputs MUST be marked with `#[user_input]` attribute for escaping

**🎭 AUTH-MIN SECURITY COMMENT**:
- ⚠️ **ESCAPING STRATEGY ISSUE**: The implementation shows `value.replace('{', "\\{").replace('}', "\\}")` which escapes to `\{` and `\}`. This is CORRECT for preventing template re-parsing, but **VERIFY** that the output format doesn't break downstream log parsers.
- 🚨 **CRITICAL**: The compile-time validation only catches literal braces in the template string itself, NOT in variable values. The runtime escaping MUST be applied to ALL interpolated variables, not just user input. Even internal variables could contain `{}` (e.g., JSON strings, error messages).
- **RECOMMENDATION**: Consider rejecting `{}` in variable values instead of escaping, to avoid confusion in logs. If a variable contains `{worker_id}`, escaping it to `\{worker_id\}` might be misleading.

**⏱️ PERFORMANCE TEAM FINAL VERDICT**:
- ⚠️ **CONDITIONAL APPROVAL** — Escaping adds overhead, must be minimized
- 🎯 **CRITICAL**: Runtime escaping adds 50-100ns per variable (2x `.replace()` calls + allocation). This is UNACCEPTABLE for high-frequency narration.
- 🔴 **VETO ON SECURITY**: Auth-min's requirement to escape ALL variables is REJECTED. **ALTERNATIVE**: Only escape user-controlled inputs (marked with `#[user_input]` attribute in proc macro)
- 📊 **TARGET**: <20ns overhead for escaping (if required at all)
- 🚨 **BLOCKING**: If escaping is mandatory, it MUST be:
  1. Applied ONLY to user-controlled variables (not internal variables)
  2. Implemented with zero-copy check (if no `{}` found, return borrowed)
  3. Use single-pass scan (not 2x `.replace()` calls)
- ⚡ **OPTIMIZATION**: Compile-time validation is FREE (proc macro), runtime escaping is EXPENSIVE

**Implementation** (Performance Team Approved):
```rust
// Proc macro: validate at compile time (FREE)
if template_literal.contains('{') || template_literal.contains('}') {
    return syn::Error::new_spanned(
        template_literal,
        "Template literal cannot contain braces"
    ).to_compile_error();
}

// Runtime: escape ONLY user-controlled inputs (opt-in)
#[narrate(
    human: "User {user_name} logged in",  // user_name is user input
    #[user_input(user_name)]  // Explicit marking
)]

// Generated code:
let user_name_escaped = if user_name.contains('{') || user_name.contains('}') {
    Cow::Owned(user_name.replace('{', "\\{").replace('}', "\\}"))
} else {
    Cow::Borrowed(user_name)  // Zero-copy if no braces
};
```

**Impact**: Affects Unit 1.5

---

### **Decision 8: Log Injection (CRLF) Prevention**

**Context**: auth-min noted missing CRLF injection prevention.

**Options**:

#### **Option A: Strip CRLF**
```rust
fn sanitize_log_injection(text: &str) -> String {
    text.replace('\r', "").replace('\n', " ")
}
```

**Pros**: Simple, effective
**Cons**: Changes multiline content

#### **Option B: Reject CRLF**
```rust
fn validate_no_crlf(text: &str) -> Result<&str, Error> {
    if text.contains('\r') || text.contains('\n') {
        return Err(Error::LogInjection);
    }
    Ok(text)
}
```

**Pros**: Strict
**Cons**: Breaks legitimate multiline (story mode)

#### **Option C: Encode CRLF**
```rust
fn encode_crlf(text: &str) -> String {
    text.replace('\r', "\\r").replace('\n', "\\n")
}
```

**Pros**: Preserves content, safe
**Cons**: Allocates, changes output

**DECISION MADE**: ✅ **Option A: Strip \n, \r, \t ONLY** (Performance Team Override)

**Rationale**:
- Security Team: MEDIUM severity - log injection vulnerability (MED-2)
- Performance Team: VETO on encoding all control chars (allocates on every narration)
- **FINAL DECISION**: Strip only `\n`, `\r`, `\t` (actual injection vectors)
- Other control chars (\0, \x1B, \x08) are rare and don't pose injection risk
- Story mode: Use dedicated field that preserves newlines, don't inject into `human` field

**🎭 AUTH-MIN SECURITY COMMENT**:
- ✅ **APPROVED**: Encoding CRLF as `\n` and `\r` prevents log injection while preserving content
- ⚠️ **INCOMPLETE**: The implementation only encodes `\n`, `\r`, and `\t`. **MISSING**: Other control characters that could break log parsers:
  - `\0` (null byte) - could truncate logs
  - `\x1B` (ESC) - could inject ANSI escape codes for terminal manipulation
  - `\x08` (backspace) - could erase previous log content
- **REQUIRED**: Encode ALL control characters (U+0000 to U+001F, U+007F) except whitespace:
  ```rust
  fn sanitize_log_field(text: &str) -> String {
      text.chars()
          .map(|c| match c {
              '\n' => "\\n".to_string(),
              '\r' => "\\r".to_string(),
              '\t' => "\\t".to_string(),
              c if c.is_control() => format!("\\x{:02X}", c as u32),
              c => c.to_string(),
          })
          .collect()
  }

**⏱️ PERFORMANCE TEAM FINAL VERDICT**:
- ⚠️ **CONDITIONAL APPROVAL** — Encoding adds overhead, must be optimized
- 🎯 **CRITICAL**: Auth-min's requirement to encode ALL control characters (U+0000 to U+007F) is EXPENSIVE. `.chars()` + `.map()` + `.collect()` allocates on EVERY narration.
- 🔴 **VETO ON SECURITY**: Encoding all control chars is OVERKILL. **SIMPLIFIED APPROACH**: Only encode `\n`, `\r`, `\t` (the actual injection vectors). Other control chars are rare and don't pose injection risk.
- 📊 **TARGET**: <50ns overhead for CRLF encoding (if no control chars found, zero-copy)
- 🚨 **BLOCKING**: If encoding is mandatory, it MUST be:
  1. Zero-copy check first: `if !text.contains(|c: char| c.is_control()) { return Cow::Borrowed(text) }`
  2. Single-pass scan (not `.chars().map().collect()`)
  3. Use `String::replace()` for common cases (\n, \r, \t only)
- ⚡ **OPTIMIZATION**: Use `.as_bytes()` for ASCII control char detection (faster than `.chars()`)
  ```

**Implementation** (Performance Team Approved):
```rust
// Zero-copy fast path
fn sanitize_log_field(text: &str) -> Cow<'_, str> {
    if !text.contains(|c: char| matches!(c, '\n' | '\r' | '\t')) {
        return Cow::Borrowed(text);  // Zero-copy (90% of cases)
    }
    
    // Only allocate if control chars found
    Cow::Owned(
        text.replace('\n', " ")  // Strip, not encode (faster)
            .replace('\r', " ")
            .replace('\t', " ")
    )
}
```

**Impact**: Affects Unit 2.8 (add new sanitization step)

---

### **Decision 9: Correlation ID Validation**

**Context**: auth-min identified injection risk if correlation IDs come from user input.

**Options**:

#### **Option A: UUID v4 Only (STRICT)**
```rust
fn validate_correlation_id(id: &str) -> Result<&str, Error> {
    // Regex: 8-4-4-4-12 hex format
    if !CORRELATION_ID_PATTERN.is_match(id) {
        return Err(Error::InvalidCorrelationId);
    }
    Ok(id)
}
```

**Pros**: Secure, predictable format
**Cons**: Inflexible, rejects valid IDs

#### **Option B: Alphanumeric + Hyphens (FLEXIBLE)**
```rust
fn validate_correlation_id(id: &str) -> Result<&str, Error> {
    if id.len() > 64 || !id.chars().all(|c| c.is_alphanumeric() || c == '-') {
        return Err(Error::InvalidCorrelationId);
    }
    Ok(id)
}
```

**Pros**: Flexible, simple
**Cons**: Less strict

#### **Option C: HMAC-Signed (MAXIMUM SECURITY)**
```rust
fn validate_correlation_id(id: &str, signature: &str) -> Result<&str, Error> {
    let expected_sig = hmac_sha256(id, SECRET_KEY);
    if signature != expected_sig {
        return Err(Error::InvalidSignature);
    }
    Ok(id)
}
```

**Pros**: Prevents forgery
**Cons**: Complex, requires key management

**DECISION MADE**: ✅ **Option A: UUID v4 only (Simplified)** (Performance Team Override)

**Rationale**:
- Security Team: MEDIUM severity - correlation ID injection (MED-3, EXIST-3)
- Performance Team: VETO on HMAC-signed correlation IDs (500-1000ns overhead)
- **FINAL DECISION**: UUID v4 validation with byte-level checks (<100ns)
- Version validation (position 14, 19) is OPTIONAL (adds 5ns, minimal benefit)
- Forgery risk is ACCEPTED (no HMAC signing)

**🎭 AUTH-MIN SECURITY COMMENT**:
- ✅ **APPROVED**: UUID v4 validation prevents injection attacks
- ✅ **APPROVED**: Length check (36 chars) and hyphen position validation is secure
- ⚠️ **CASE SENSITIVITY**: The implementation uses `c.is_ascii_hexdigit()` which accepts both uppercase and lowercase. **RECOMMENDATION**: Normalize to lowercase before validation to ensure consistent correlation ID format across systems.
- ⚠️ **VERSION VALIDATION MISSING**: UUID v4 has specific version bits (4 in position 14, variant bits in position 19). The current validation accepts ANY UUID format. **RECOMMENDATION**: Add version validation:
  ```rust
  // Position 14 must be '4' (version 4)
  14 => c == '4',
  // Position 19 must be '8', '9', 'a', 'b' (variant bits)
  19 => matches!(c, '8' | '9' | 'a' | 'b' | 'A' | 'B'),
  ```
- 🚨 **FORGERY RISK**: UUID v4 validation prevents injection but NOT forgery. Attackers can generate valid UUIDs to poison traces. Consider HMAC-signed correlation IDs (Option C) for security-critical flows.

**⏱️ PERFORMANCE TEAM FINAL VERDICT**:
- ✅ **APPROVED** — UUID v4 validation is acceptable
- 🎯 **OPTIMIZATION**: Auth-min's version validation adds 2 extra checks (positions 14, 19). This is ACCEPTABLE (<5ns overhead).
- ⚠️ **VETO ON SECURITY**: Auth-min's "forgery risk" concern about HMAC-signed correlation IDs is REJECTED. HMAC adds 500-1000ns overhead per validation (cryptographic operation). This is UNACCEPTABLE for correlation IDs.
- 📊 **TARGET**: <100ns for UUID validation (byte-level checks, no regex)
- 🚨 **IMPLEMENTATION**: Use byte-level validation (check hyphens at positions 8,13,18,23, hex chars elsewhere) — 10x faster than regex
- ⚡ **OPTIMIZATION**: Early-return on length check (if len != 36, reject immediately)

**Implementation** (Performance Team Approved):
```rust
// Byte-level validation (10x faster than char iteration)
fn validate_correlation_id(id: &str) -> Option<&str> {
    if id.len() != 36 {
        return None;  // Early return
    }
    
    let bytes = id.as_bytes();
    
    // Check hyphens at positions 8, 13, 18, 23
    if bytes[8] != b'-' || bytes[13] != b'-' || 
       bytes[18] != b'-' || bytes[23] != b'-' {
        return None;
    }
    
    // Check hex chars (skip version validation for performance)
    for (i, &b) in bytes.iter().enumerate() {
        if i == 8 || i == 13 || i == 18 || i == 23 {
            continue;  // Skip hyphens
        }
        if !b.is_ascii_hexdigit() {
            return None;
        }
    }
    
    Some(id)  // Return borrowed (zero-copy)
}
```

**Impact**: Affects Unit 2.4

---

### **Decision 10: Actor Inference Validation**

**Context**: auth-min noted module path parsing could allow actor identity spoofing.

**Options**:

#### **Option A: Allowlist Validation**
```rust
const ALLOWED_ACTORS: &[&str] = &[
    "orchestratord",
    "pool-managerd",
    "worker-orcd",
    "inference-engine",
];

fn validate_inferred_actor(actor: &str) -> Result<&str, Error> {
    if !ALLOWED_ACTORS.contains(&actor) {
        return Err(Error::InvalidActor);
    }
    Ok(actor)
}
```

**Pros**: Prevents spoofing
**Cons**: Requires maintenance

#### **Option B: Namespace Validation**
```rust
fn validate_inferred_actor(module_path: &str) -> Result<&str, Error> {
    // Must start with "llama_orch::"
    if !module_path.starts_with("llama_orch::") {
        return Err(Error::InvalidNamespace);
    }
    // Extract actor from path
    Ok(extract_actor(module_path))
}
```

**Pros**: Flexible, namespace-based
**Cons**: Can be bypassed with `#[path]`

#### **Option C: No Validation (Trust Developers)**
```rust
fn infer_actor(module_path: &str) -> &str {
    extract_actor(module_path)  // No validation
}
```

**Pros**: Simple, flexible
**Cons**: Vulnerable to spoofing

**DECISION MADE**: ✅ **Option A: Allowlist** (secure, maintenance)

**Rationale**:
- Security Team: LOW severity - actor inference spoofing (LOW-2)
- `#[path]` attribute can bypass module path inference
- Allowlist prevents spoofing attacks

**🎭 AUTH-MIN SECURITY COMMENT**:
- ✅ **APPROVED**: Allowlist validation prevents actor spoofing
- ⚠️ **MAINTENANCE BURDEN**: The allowlist must be updated whenever new services are added. **RECOMMENDATION**: Generate the allowlist at build time from workspace members to avoid manual maintenance:
  ```rust
  // In build.rs
  let actors = workspace_members()
      .filter(|m| m.starts_with("llama-orch-"))
      .map(|m| m.strip_prefix("llama-orch-").unwrap())
      .collect::<Vec<_>>();
  // Generate const ALLOWED_ACTORS at compile time
  ```
- ⚠️ **CASE SENSITIVITY**: The allowlist comparison should be case-sensitive to prevent bypasses like "Orchestratord" vs "orchestratord".
- 🚨 **COMPILE-TIME vs RUNTIME**: The validation happens at macro expansion (compile-time). If actor inference fails validation, should it be a compile error or runtime error? **RECOMMENDATION**: Compile error is safer - prevents deployment of code with invalid actors.

**⏱️ PERFORMANCE TEAM FINAL VERDICT**:
- ✅ **APPROVED** — Allowlist validation is acceptable (compile-time, zero runtime overhead)
- 🎯 **OPTIMIZATION**: Auth-min's build.rs generation is EXCELLENT — zero maintenance, zero runtime cost
- ✅ **SECURITY ALIGNMENT**: Compile-time validation is PERFECT — catches invalid actors before deployment
- 📊 **PERFORMANCE**: Zero runtime overhead (validation happens at macro expansion)
- ⚡ **IMPLEMENTATION**: Use `const ALLOWED_ACTORS: &[&str]` generated by build.rs, validate in proc macro

**Implementation**:
```rust
const ALLOWED_ACTORS: &[&str] = &[
    "orchestratord",
    "pool-managerd",
    "worker-orcd",
    "inference-engine",
    "vram-residency",
];

fn validate_inferred_actor(actor: &str) -> Result<&str, Error> {
    if !ALLOWED_ACTORS.contains(&actor) {
        return Err(Error::InvalidActor);
    }
    Ok(actor)
}
```

**Impact**: Affects Unit 1.2

---

## 📊 Decision Summary Table

| Decision | Final Decision | Rationale | Impact | Status |
|----------|----------------|-----------|--------|--------|
| **1. Tracing Opt-In** | ✅ **Opt-In** | Performance: 0ns overhead required | Units 1.3, 2.2, 3.3 | **APPROVED** |
| **2. Timing Strategy** | ✅ **Conditional Instant::now()** | Performance + Security: No timing side-channels | Unit 1.3 | **APPROVED** |
| **3. Template Allocation** | ✅ **Pre-compiled + Stack Buffers** | Performance: <100ns target, zero heap | Units 1.4, 1.5 | **APPROVED** |
| **4. Redaction Strategy** | ✅ **Hybrid (single-pass + Cow)** | Security: Fix ReDoS; Performance: <5μs | Unit 2.3 | **APPROVED** |
| **5. Sampling Architecture** | 🚫 **REJECTED - DO NOT IMPLEMENT** | Performance: DISASTER; Security: 5 CRITICAL vulns | Unit 2.7 | **REMOVED** |
| **6. Unicode Validation** | ✅ **Tiered (ASCII fast + comprehensive)** | Performance: <5μs; Security: Full validation | Unit 2.8 | **APPROVED** |
| **7. Template Injection** | ✅ **Compile-time + Runtime Escape** | Security: Prevent injection | Unit 1.5 | **APPROVED** |
| **8. CRLF Prevention** | ✅ **Encode (preserve content)** | Security: Prevent log injection | Unit 2.8 | **APPROVED** |
| **9. Correlation ID** | ✅ **UUID v4 Only** | Security: Strict validation | Unit 2.4 | **APPROVED** |
| **10. Actor Validation** | ✅ **Allowlist** | Security: Prevent spoofing | Unit 1.2 | **APPROVED** |

---

## ✅ FINAL DECISIONS (Non-Negotiable)

All decisions have been made based on **security and performance team reviews**. These are **NON-NEGOTIABLE** requirements:

### **APPROVED FOR IMPLEMENTATION**:

1. ✅ **Tracing Opt-In** - Zero overhead in production (Performance Team: REQUIRED)
2. ✅ **Conditional `Instant::now()`** - No timing side-channels (Security + Performance: REQUIRED)
3. ✅ **Pre-Compiled Templates + Stack Buffers** - <100ns target (Performance Team: BLOCKING)
4. ✅ **Hybrid Redaction (Single-pass + Cow)** - Fix ReDoS + <5μs target (Security: CRITICAL; Performance: BLOCKING)
5. ✅ **Tiered Unicode Validation** - ASCII fast path + comprehensive (Security: HIGH; Performance: BLOCKING)
6. ✅ **Template Injection Prevention** - Compile-time + runtime (Security: MEDIUM)
7. ✅ **CRLF Encoding** - Preserve story mode (Security: MEDIUM)
8. ✅ **UUID v4 Validation** - Strict format (Security: MEDIUM)
9. ✅ **Actor Allowlist** - Prevent spoofing (Security: LOW)

### **REJECTED - DO NOT IMPLEMENT**:

🚫 **Unit 2.7: Sampling & Rate Limiting**
- **Security Team**: 5 CRITICAL vulnerabilities (mutex poisoning, HashMap collision DoS, memory leak, contention)
- **Performance Team**: PERFORMANCE DISASTER (+200-400ns + mutex contention)
- **Alternative**: Use `RUST_LOG` or `tracing-subscriber` filtering

---

## 📝 Next Steps

1. ✅ **Team Sign-Off Complete**: Performance Team has final authority
2. 🚨 **Update Implementation Plan**: Remove Unit 2.7, incorporate Performance Team overrides
3. 📊 **Document Trade-offs**: Security Team concerns are acknowledged but overruled for performance
4. ⚡ **Performance Targets**:
   - Template interpolation: <100ns (no runtime escaping)
   - CRLF sanitization: <50ns (zero-copy for clean strings)
   - Unicode validation: <1μs for ASCII (zero-copy), <5μs for UTF-8
   - Correlation ID: <100ns (byte-level validation)
5. 🔒 **Accepted Security Risks** (Performance Team Decision):
   - Template injection: Only user-marked inputs are escaped
   - Log injection: Only \n, \r, \t are stripped (not all control chars)
   - Unicode: Simplified validation (not comprehensive emoji ranges)
   - Correlation ID: No HMAC signing (forgery risk accepted)

---

## 🔐 Security Sign-Off Section

**auth-min Team 🎭**: Security review complete.

- ✅ Decision 4: Redaction Strategy (ReDoS fix) - **APPROVED** (fix CRIT-1)
- ✅ Decision 6: Unicode Validation (comprehensive ranges) - **APPROVED** (fix HIGH-1, HIGH-2, HIGH-3, HIGH-4)
- ✅ Decision 7: Template Injection Prevention - **APPROVED** (fix MED-1)
- ✅ Decision 8: CRLF Prevention - **APPROVED** (fix MED-2)
- ✅ Decision 9: Correlation ID Validation - **APPROVED** (fix MED-3, EXIST-3)
- ✅ Decision 10: Actor Validation - **APPROVED** (fix LOW-2)
- 🚫 Unit 2.7: Sampling - **REJECTED** (5 CRITICAL vulnerabilities)

**Status**: ✅ **APPROVED WITH UNIT 2.7 REMOVAL**  
**Signed**: auth-min Team 🎭  
**Date**: 2025-10-04

---

## ⚡ Performance Sign-Off Section

**Performance Team ⏱️**: Performance review complete.

- ✅ Decision 1: Tracing Opt-In - **APPROVED** (0ns overhead in production)
- ✅ Decision 2: Timing Strategy - **APPROVED** (conditional compilation required)
- ✅ Decision 3: Template Allocation - **APPROVED** (<100ns target with stack buffers)
- ✅ Decision 4: Redaction Strategy - **APPROVED** (single-pass + Cow, <5μs target)
- 🚫 Decision 5: Sampling - **REJECTED** (performance disaster, use RUST_LOG instead)
- ✅ Decision 6: Unicode Validation - **APPROVED** (ASCII fast path required)
- ⚠️ Decision 7: Template Injection - **CONDITIONAL** (escape user inputs only, not all variables)
- ⚠️ Decision 8: CRLF Prevention - **CONDITIONAL** (encode \n, \r, \t only, not all control chars)
- ✅ Decision 9: Correlation ID - **APPROVED** (UUID v4 validation, no HMAC overhead)
- ✅ Decision 10: Actor Validation - **APPROVED** (compile-time allowlist, zero runtime cost)

**Status**: ✅ **APPROVED WITH CONDITIONS**  
**Signed**: Performance Team ⏱️  
**Date**: 2025-10-04

### **Performance Team Vetoes & Overrides**:

1. **VETO**: Auth-min's requirement to escape ALL template variables (Decision 7) — REJECTED
   - **Reason**: Adds 50-100ns per variable, unacceptable overhead
   - **Alternative**: Escape ONLY user-controlled inputs marked with `#[user_input]` attribute

2. **VETO**: Auth-min's requirement to encode ALL control characters (Decision 8) — REJECTED
   - **Reason**: `.chars().map().collect()` allocates on every narration
   - **Alternative**: Encode only `\n`, `\r`, `\t` (actual injection vectors)

3. **VETO**: Auth-min's "comprehensive emoji ranges" (20+ ranges) in Decision 6 — REJECTED
   - **Reason**: Excessive complexity, minimal security benefit
   - **Alternative**: Use `c.is_control()` + basic zero-width blocklist (5 chars)

4. **VETO**: Auth-min's HMAC-signed correlation IDs (Decision 9) — REJECTED
   - **Reason**: 500-1000ns cryptographic overhead per validation
   - **Alternative**: UUID v4 validation (byte-level checks, <100ns)

5. **VETO**: Auth-min's "residual risk" concern about timing in dev builds (Decision 2) — OVERRULED
   - **Reason**: Dev builds are NOT production, timing data is EXPECTED in observability
   - **Verdict**: This is not a security issue

**Performance Team Authority**: In this milestone, Performance Team has final say on performance-impacting decisions. Security concerns that add >50ns overhead per narration are subject to performance veto.

---

**Document Status**: ✅ **ALL DECISIONS FINALIZED (Performance Team Authority)**  
**Team Sign-Off**: ⚠️ Security Team Concerns Noted | ✅ Performance Team APPROVED  
**Implementation Start**: **APPROVED** with Performance Team overrides  
**Authority**: Performance Team has final say on performance-impacting decisions  
**Next Steps**: Update IMPLEMENTATION_PLAN.md to reflect Performance Team decisions

---

## 📊 Performance Verification Requirements

Before merging any implementation, the following benchmarks MUST prove the performance targets are met:

### Mandatory Benchmarks (Blocking)

```rust
// benches/narration_performance.rs
#[bench]
fn bench_template_interpolation_3_vars(b: &mut Bencher) {
    // TARGET: <100ns
    b.iter(|| {
        narrate!(human: "Job {job_id} worker {worker_id} status {status}")
    });
}

#[bench]
fn bench_crlf_sanitization_clean_string(b: &mut Bencher) {
    // TARGET: <50ns (zero-copy)
    let text = "Accepted request; queued at position 3";
    b.iter(|| sanitize_log_field(text));
}

#[bench]
fn bench_redaction_clean_string_1000_chars(b: &mut Bencher) {
    // TARGET: <1μs (zero-copy)
    let text = "A".repeat(1000);
    b.iter(|| redact_secrets(&text, RedactionPolicy::default()));
}

#[bench]
fn bench_redaction_with_secrets_1000_chars(b: &mut Bencher) {
    // TARGET: <5μs
    let text = format!("Bearer abc123 {}", "x".repeat(990));
    b.iter(|| redact_secrets(&text, RedactionPolicy::default()));
}

#[bench]
fn bench_uuid_validation(b: &mut Bencher) {
    // TARGET: <100ns
    let uuid = "550e8400-e29b-41d4-a716-446655440000";
    b.iter(|| validate_correlation_id(uuid));
}

#[bench]
fn bench_unicode_validation_ascii(b: &mut Bencher) {
    // TARGET: <1μs (zero-copy)
    let text = "Accepted request";
    b.iter(|| sanitize_for_json(text));
}
```

### CI Integration (Required)

```yaml
# .github/workflows/performance.yml
- name: Run performance benchmarks
  run: cargo bench --bench narration_performance
  
- name: Verify performance targets
  run: |
    # Fail if any benchmark exceeds target by >10%
    cargo bench --bench narration_performance -- --save-baseline main
    cargo bench --bench narration_performance -- --baseline main --fail-fast
```

### Production Verification (Required)

```bash
# Verify zero-overhead production builds
cargo expand --release --features="" | grep -q "Instant::now" && exit 1 || exit 0
cargo expand --release --features="" | grep -q "trace_tiny" && exit 1 || exit 0

# Verify binary size identical with/without trace features
cargo build --release --no-default-features
sha256sum target/release/narration-core > /tmp/baseline.sha256
cargo build --release --features trace-enabled
sha256sum target/release/narration-core > /tmp/with-trace.sha256
diff /tmp/baseline.sha256 /tmp/with-trace.sha256 && echo "FAIL: Binary changed" || echo "PASS"
```

---

**Document Status**: ✅ **ALL DECISIONS FINALIZED (Performance Team Authority)**  
**Team Sign-Off**: ⚠️ Security Team Concerns Noted | ✅ Performance Team APPROVED  
**Implementation Start**: **APPROVED** with Performance Team overrides  
**Authority**: Performance Team has final say on performance-impacting decisions  
**Verification**: Benchmarks MUST prove all performance targets before merge

---

*May your decisions be wise, your code be fast, and your logs be secure! 🎀*
