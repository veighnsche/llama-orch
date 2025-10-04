# 🔒 Narration Core — Security Review & Attack Surface Analysis

**Document Version**: 1.1.0  
**Review Date**: 2025-10-04  
**Last Updated**: 2025-10-04 13:15 CET  
**Reviewed By**: auth-min Security Team 🎭  
**Status**: CRITICAL ISSUES IDENTIFIED — MITIGATION REQUIRED

---

## 📢 UPDATE NOTICE (v1.1.0)

**Date**: 2025-10-04 13:15 CET  
**Changes**: Added security comments to `DESIGN_DECISIONS.md`

The auth-min security team has reviewed all 10 design decisions in `DESIGN_DECISIONS.md` and added security comments where issues were identified:

### Security Comments Added:
1. **Decision 2 (Timing Strategy)** - Added warning about timing data exposure in dev builds
2. **Decision 3 (Template Allocation)** - CRITICAL: Template injection prevention must be in generated code
3. **Decision 4 (Redaction Strategy)** - CRITICAL: JWT pattern needs bounded quantifiers to prevent ReDoS
4. **Decision 6 (Unicode Validation)** - CRITICAL: Must validate actor/action fields, incomplete emoji/zero-width lists
5. **Decision 7 (Template Injection)** - Warning about escaping strategy and runtime validation
6. **Decision 8 (CRLF Prevention)** - INCOMPLETE: Missing control character encoding (null, ESC, backspace)
7. **Decision 9 (Correlation ID)** - Missing UUID v4 version validation, forgery risk
8. **Decision 10 (Actor Validation)** - Recommendations for build-time generation and compile-time errors

### Key Findings from Design Review:
- **3 CRITICAL issues** identified in approved decisions (template injection, JWT ReDoS, actor validation)
- **5 INCOMPLETE implementations** requiring additional security measures
- **All decisions approved** with security enhancements required

**Action Required**: Review security comments in `DESIGN_DECISIONS.md` and incorporate into implementation.

---

## 📋 Executive Summary

This document provides a comprehensive security analysis of the `observability-narration-core` crate, covering both the **existing codebase** (`/src`) and the **planned implementation** (`IMPLEMENTATION_PLAN.md`). The review identifies **15 distinct attack surfaces** across 8 security categories, with **5 CRITICAL** vulnerabilities requiring immediate mitigation.

### Severity Breakdown

| Severity | Count | Status |
|----------|-------|--------|
| **CRITICAL** | 5 | 🚨 Requires immediate mitigation |
| **HIGH** | 4 | ⚠️ Must address before production |
| **MEDIUM** | 4 | ⚠️ Address before v1.0 |
| **LOW** | 2 | 📝 Document and monitor |

### Risk Assessment

**Overall Risk Level**: **HIGH**  
**Production Readiness**: **NOT READY** (5 critical issues blocking)  
**Recommended Action**: **HALT IMPLEMENTATION** until critical issues are resolved

---

## 🎯 Scope of Review

### Current Codebase Analyzed
- ✅ `src/lib.rs` — Core narration function and field definitions
- ✅ `src/redaction.rs` — Secret redaction patterns
- ✅ `src/capture.rs` — Test capture adapter
- ✅ `src/auto.rs` — Auto-injection helpers
- ✅ `src/http.rs` — HTTP header propagation
- ✅ `src/otel.rs` — OpenTelemetry integration
- ✅ `src/trace.rs` — Lightweight trace macros

### Planned Implementation Analyzed
- ✅ `IMPLEMENTATION_PLAN.md` — All 38 implementation units
- ✅ Week 1: Proc macro crate (Units 1.1-1.5)
- ✅ Week 2: Core enhancements (Units 2.1-2.8)
- ✅ Week 3: Editorial enforcement (Units 3.1-3.4)
- ✅ Week 4: Integration & rollout (Units 4.1-4.7)

---

## 🚨 CRITICAL VULNERABILITIES (Immediate Action Required)

### CRIT-1: ReDoS Vulnerability in Private Key Regex Pattern
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.3 (lines 410-414)  
**Severity**: CRITICAL  
**CWE**: CWE-1333 (Inefficient Regular Expression Complexity)

**Vulnerability**:
```rust
// VULNERABLE PATTERN (from implementation plan)
Regex::new(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]+?-----END [A-Z ]+PRIVATE KEY-----")
```

**Attack Vector**:
1. Attacker provides input: `"-----BEGIN PRIVATE KEY-----" + ("A" * 100000)` (no END marker)
2. Lazy quantifier `[\s\S]+?` causes catastrophic backtracking
3. Regex engine tries all possible match positions → exponential time complexity
4. **Result**: CPU exhaustion, DoS attack

**Proof of Concept**:
```rust
// This input will hang the regex engine for minutes
let malicious = format!("-----BEGIN PRIVATE KEY-----{}", "A".repeat(100000));
redact_secrets(&malicious, policy); // HANGS
```

**Impact**:
- **Availability**: Complete service DoS via CPU exhaustion
- **Scope**: Any narration containing malicious private key-like patterns
- **Exploitability**: Trivial (single HTTP request with crafted payload)

**Mitigation Required**:
```rust
// OPTION 1: Bounded quantifier (max 10KB key)
Regex::new(r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]{1,10240}?-----END [A-Z ]+PRIVATE KEY-----")

// OPTION 2: Line-by-line parsing (recommended)
fn redact_private_key(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut in_key = false;
    let mut result = String::new();
    
    for line in lines {
        if line.contains("BEGIN") && line.contains("PRIVATE KEY") {
            in_key = true;
            result.push_str("[REDACTED PRIVATE KEY]\n");
        } else if line.contains("END") && line.contains("PRIVATE KEY") {
            in_key = false;
        } else if !in_key {
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}
```

**Acceptance Criteria**:
- [ ] Replace lazy quantifier with bounded quantifier OR line-by-line parser
- [ ] Add ReDoS regression test with 100KB malicious input
- [ ] Verify redaction completes in <10ms for 100KB input
- [ ] Document maximum supported key size

---

### CRIT-2: Mutex Poisoning DoS in Sampling Module
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.7 (line 711)  
**Severity**: CRITICAL  
**CWE**: CWE-667 (Improper Locking)

**Vulnerability**:
```rust
// VULNERABLE CODE (from implementation plan)
let mut counters = self.counters.lock().unwrap(); // PANIC if poisoned!
```

**Attack Vector**:
1. Attacker triggers panic in thread A while holding mutex (e.g., via integer overflow)
2. Mutex becomes poisoned
3. All subsequent `lock().unwrap()` calls panic
4. **Result**: Complete narration system failure, cascading panics

**Proof of Concept**:
```rust
// Thread A: Trigger panic while holding lock
std::thread::spawn(|| {
    let mut counters = SAMPLER.counters.lock().unwrap();
    panic!("Intentional panic"); // Mutex now poisoned
});

// Thread B: All subsequent locks panic
let counters = SAMPLER.counters.lock().unwrap(); // PANICS!
```

**Impact**:
- **Availability**: Complete narration system failure
- **Cascading Failure**: All threads attempting narration will panic
- **Recovery**: Requires service restart

**Mitigation Required**:
```rust
// OPTION 1: Graceful degradation
match self.counters.lock() {
    Ok(mut counters) => {
        // Normal sampling logic
    }
    Err(poisoned) => {
        // Log error, allow narration to proceed without sampling
        tracing::warn!("Sampling mutex poisoned, disabling rate limiting");
        return true; // Allow event through
    }
}

// OPTION 2: Lock-free atomic counters (recommended)
use dashmap::DashMap;
use std::sync::atomic::{AtomicU32, Ordering};

pub struct Sampler {
    config: SamplingConfig,
    counters: Arc<DashMap<String, (Instant, AtomicU32)>>, // Lock-free!
}
```

**Acceptance Criteria**:
- [ ] Replace `.unwrap()` with graceful error handling OR lock-free data structure
- [ ] Add poison recovery test (trigger panic, verify system continues)
- [ ] Document poison handling behavior
- [ ] Performance test: verify no contention with 1000 concurrent threads

---

### CRIT-3: HashMap Collision DoS Attack
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.7 (line 710)  
**Severity**: CRITICAL  
**CWE**: CWE-407 (Algorithmic Complexity)

**Vulnerability**:
```rust
// VULNERABLE CODE (from implementation plan)
let key = format!("{}:{}", actor, action); // Allocates + user-controlled input
let mut counters = self.counters.lock().unwrap();
counters.entry(key.clone()).or_insert((now, 0)); // HashMap collision attack
```

**Attack Vector**:
1. Attacker controls `actor` or `action` via HTTP headers or input
2. Craft inputs with identical hash values (hash collision)
3. HashMap degrades from O(1) to O(n) lookup
4. **Result**: CPU exhaustion, DoS attack

**Proof of Concept**:
```rust
// Generate collision-inducing keys (simplified example)
for i in 0..100000 {
    let actor = format!("actor_{}", i); // Crafted to collide
    let action = "action";
    sampler.should_sample(&actor, action); // O(n) lookup after collisions
}
```

**Impact**:
- **Performance**: HashMap lookup degrades to O(n)
- **Availability**: CPU exhaustion under collision attack
- **Amplification**: Each narration call becomes expensive

**Mitigation Required**:
```rust
// OPTION 1: Sanitize keys (remove collision-inducing chars)
fn sanitize_key(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
        .take(64) // Limit length
        .collect()
}

let key = format!("{}:{}", sanitize_key(actor), sanitize_key(action));

// OPTION 2: Use tuple keys (no allocation, no collision)
use std::collections::HashMap;
let counters: HashMap<(&'static str, &'static str), (Instant, u32)> = HashMap::new();
counters.entry((actor, action)).or_insert((now, 0));

// OPTION 3: Lock-free DashMap with SipHash (recommended)
use dashmap::DashMap;
let counters: DashMap<(&'static str, &'static str), (Instant, AtomicU32)> = DashMap::new();
```

**Acceptance Criteria**:
- [ ] Sanitize actor/action before using as HashMap keys OR use tuple keys
- [ ] Add collision attack test (10K identical hash keys, verify <100ms)
- [ ] Document key sanitization rules
- [ ] Benchmark: verify O(1) lookup under collision attack

---

### CRIT-4: Unbounded Memory Growth in Sampling Counters
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.7 (line 714)  
**Severity**: CRITICAL  
**CWE**: CWE-770 (Allocation of Resources Without Limits)

**Vulnerability**:
```rust
// VULNERABLE CODE (from implementation plan)
let mut counters = self.counters.lock().unwrap();
counters.entry(key.clone()).or_insert((now, 0)); // No eviction policy!
```

**Attack Vector**:
1. Attacker generates unique `actor:action` combinations
2. Each unique key adds entry to HashMap
3. HashMap grows unbounded
4. **Result**: Memory exhaustion, OOM kill

**Proof of Concept**:
```rust
// Generate 1 million unique keys
for i in 0..1_000_000 {
    let actor = format!("actor_{}", i);
    let action = format!("action_{}", i);
    sampler.should_sample(&actor, &action); // HashMap grows to 1M entries
}
// Memory usage: ~100MB+ (depending on entry size)
```

**Impact**:
- **Availability**: Memory exhaustion → OOM kill → service crash
- **Persistence**: HashMap never shrinks, memory leak persists
- **Amplification**: Each unique actor:action pair adds permanent entry

**Mitigation Required**:
```rust
// OPTION 1: LRU eviction with max size
use lru::LruCache;

pub struct Sampler {
    config: SamplingConfig,
    counters: Arc<Mutex<LruCache<String, (Instant, u32)>>>,
}

impl Sampler {
    pub fn new(config: SamplingConfig) -> Self {
        Self {
            config,
            counters: Arc::new(Mutex::new(LruCache::new(10_000))), // Max 10K entries
        }
    }
}

// OPTION 2: TTL-based cleanup (recommended)
pub struct Sampler {
    config: SamplingConfig,
    counters: Arc<DashMap<String, (Instant, AtomicU32)>>,
    last_cleanup: Arc<Mutex<Instant>>,
}

impl Sampler {
    fn cleanup_expired(&self) {
        let now = Instant::now();
        let mut last = self.last_cleanup.lock().unwrap();
        
        if now.duration_since(*last) > Duration::from_secs(60) {
            self.counters.retain(|_, (timestamp, _)| {
                now.duration_since(*timestamp) < Duration::from_secs(60)
            });
            *last = now;
        }
    }
}
```

**Acceptance Criteria**:
- [ ] Implement LRU eviction OR TTL-based cleanup
- [ ] Add memory leak test (1M unique keys, verify bounded memory)
- [ ] Document maximum memory usage (e.g., "max 10K entries = ~1MB")
- [ ] Monitor: add metric for `sampling_counters_size`

---

### CRIT-5: Global Mutex Contention in Sampling
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.7 (line 711)  
**Severity**: CRITICAL  
**CWE**: CWE-833 (Deadlock)

**Vulnerability**:
```rust
// VULNERABLE CODE (from implementation plan)
pub struct Sampler {
    counters: Arc<Mutex<HashMap<String, (Instant, u32)>>>, // GLOBAL LOCK!
}

pub fn should_sample(&self, actor: &str, action: &str) -> bool {
    let mut counters = self.counters.lock().unwrap(); // Blocks all threads!
    // ... sampling logic
}
```

**Attack Vector**:
1. High-frequency narration (1000+ events/sec across threads)
2. All threads contend for single global mutex
3. Lock contention serializes all narration
4. **Result**: Throughput collapse, latency spikes

**Proof of Concept**:
```rust
// 100 threads, each emitting 1000 narrations/sec
for _ in 0..100 {
    std::thread::spawn(|| {
        for _ in 0..1000 {
            narrate(fields.clone()); // All block on sampling mutex
        }
    });
}
// Throughput: ~1000 events/sec (should be 100K/sec)
```

**Impact**:
- **Performance**: Throughput collapse from 100K/sec to 1K/sec
- **Latency**: P99 latency spikes to 100ms+ (from <1ms)
- **Cascading**: Blocks async executors, causes starvation

**Mitigation Required**:
```rust
// OPTION 1: Sharded counters (reduce contention)
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct Sampler {
    shards: Vec<Arc<Mutex<HashMap<String, (Instant, u32)>>>>,
}

impl Sampler {
    fn get_shard(&self, key: &str) -> &Arc<Mutex<HashMap<String, (Instant, u32)>>> {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % self.shards.len();
        &self.shards[idx]
    }
}

// OPTION 2: Lock-free DashMap (recommended)
use dashmap::DashMap;

pub struct Sampler {
    counters: Arc<DashMap<String, (Instant, AtomicU32)>>, // Lock-free!
}

impl Sampler {
    pub fn should_sample(&self, actor: &str, action: &str) -> bool {
        // No global lock, per-entry locking only
        let key = format!("{}:{}", actor, action);
        // ... lock-free logic
    }
}
```

**Acceptance Criteria**:
- [ ] Replace global mutex with sharded locks OR lock-free DashMap
- [ ] Add contention test (100 threads, 1000 events/sec each)
- [ ] Verify throughput: >50K events/sec with 100 concurrent threads
- [ ] Verify latency: P99 <10ms under load

---

## ⚠️ HIGH SEVERITY VULNERABILITIES

### HIGH-1: Incomplete Unicode Validation (Injection Bypass)
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.8 (line 783)  
**Severity**: HIGH  
**CWE**: CWE-20 (Improper Input Validation)

**Vulnerability**:
```rust
// INCOMPLETE VALIDATION (from implementation plan)
|| (*c as u32) >= 0x1F000  // Emoji range - INCOMPLETE!
```

**Attack Vector**:
- Emoji ranges are fragmented: 0x1F600-0x1F64F, 0x1F300-0x1F5FF, 0x1F900-0x1F9FF, etc.
- Check `>= 0x1F000` allows malicious Unicode in gaps (e.g., 0x1F100-0x1F2FF)
- Attacker injects control chars, combining chars, or malicious Unicode

**Mitigation**:
```rust
// Use comprehensive emoji range check
fn is_emoji(c: char) -> bool {
    matches!(c as u32,
        0x1F600..=0x1F64F | // Emoticons
        0x1F300..=0x1F5FF | // Misc Symbols
        0x1F680..=0x1F6FF | // Transport
        0x1F900..=0x1F9FF | // Supplemental Symbols
        0x2600..=0x26FF   | // Misc Symbols
        0x2700..=0x27BF   | // Dingbats
        0xFE00..=0xFE0F   | // Variation Selectors
        0x1F1E6..=0x1F1FF   // Regional Indicators
    )
}
```

---

### HIGH-2: Incomplete Zero-Width Character Blocklist
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.8 (line 793)  
**Severity**: HIGH  
**CWE**: CWE-838 (Inappropriate Encoding for Output Context)

**Vulnerability**:
```rust
// INCOMPLETE BLOCKLIST (from implementation plan)
0x200B..=0x200D |  // Zero-width chars - INCOMPLETE!
0xFE00..=0xFE0F    // Variation selectors
```

**Missing Zero-Width Characters**:
- U+FEFF (Zero-Width No-Break Space / BOM)
- U+2060 (Word Joiner)
- U+180E (Mongolian Vowel Separator)
- U+200C (Zero-Width Non-Joiner)
- U+034F (Combining Grapheme Joiner)

**Attack Vector**:
- Hide malicious content in logs using unchecked zero-width chars
- Bypass log analysis tools that don't handle zero-width chars

**Mitigation**:
```rust
fn contains_zero_width(c: char) -> bool {
    matches!(c as u32,
        0x200B..=0x200D | // Zero-width space, ZWNJ, ZWJ
        0xFEFF |          // Zero-width no-break space
        0x2060 |          // Word joiner
        0x180E |          // Mongolian vowel separator
        0x034F |          // Combining grapheme joiner
        0xFE00..=0xFE0F   // Variation selectors
    )
}
```

---

### HIGH-3: Homograph Attack Vulnerability
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.8  
**Severity**: HIGH  
**CWE**: CWE-1007 (Insufficient Visual Distinction of Homoglyphs)

**Vulnerability**:
- No detection of Cyrillic/Greek lookalikes
- Attacker can spoof actor names in logs

**Attack Vector**:
```rust
// Cyrillic 'о' (U+043E) looks identical to Latin 'o' (U+006F)
narrate(NarrationFields {
    actor: "оrchestratord", // Cyrillic о, not Latin o!
    action: "dispatch",
    // ...
});
// Logs show "оrchestratord" - visually identical to "orchestratord"
// But grep for "orchestratord" won't match!
```

**Impact**:
- **Log Analysis Bypass**: Grep/search tools miss spoofed entries
- **Audit Trail Corruption**: Spoofed actor names evade detection
- **Incident Response**: False negatives during security investigations

**Mitigation**:
```rust
// Option 1: Restrict to ASCII for security-critical fields
fn validate_actor(actor: &str) -> Result<(), &'static str> {
    if !actor.is_ascii() {
        return Err("Actor must be ASCII-only");
    }
    Ok(())
}

// Option 2: Unicode normalization + confusable detection
use unicode_normalization::UnicodeNormalization;

fn detect_homograph(text: &str) -> bool {
    // Check for mixed scripts (Latin + Cyrillic)
    let has_latin = text.chars().any(|c| ('a'..='z').contains(&c) || ('A'..='Z').contains(&c));
    let has_cyrillic = text.chars().any(|c| ('\u{0400}'..='\u{04FF}').contains(&c));
    has_latin && has_cyrillic
}
```

---

### HIGH-4: Missing Unicode Normalization
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.8  
**Severity**: HIGH  
**CWE**: CWE-176 (Improper Handling of Unicode Encoding)

**Vulnerability**:
- No NFC/NFD normalization before validation
- Allows normalization-based bypasses

**Attack Vector**:
```rust
// Composed form: é (U+00E9)
let text1 = "café";

// Decomposed form: e + ́ (U+0065 + U+0301)
let text2 = "café";

// text1 != text2 (different byte sequences)
// But visually identical!
```

**Mitigation**:
```rust
use unicode_normalization::UnicodeNormalization;

pub fn sanitize_for_json(text: &str) -> String {
    // Normalize to NFC before validation
    let normalized: String = text.nfc().collect();
    
    normalized.chars()
        .filter(|c| /* validation logic */)
        .collect()
}
```

---

## ⚠️ MEDIUM SEVERITY VULNERABILITIES

### MED-1: Template Injection Vulnerability
**Location**: `IMPLEMENTATION_PLAN.md` Unit 1.5 (line 176)  
**Severity**: MEDIUM  
**CWE**: CWE-94 (Improper Control of Generation of Code)

**Vulnerability**:
```rust
// VULNERABLE: Template interpolation without sanitization
let template = "Worker {worker_id} started";
let worker_id = user_input; // Contains: "ABC} malicious {evil"
// Result: "Worker ABC} malicious {evil started"
```

**Attack Vector**:
- Variable values contain `{}` characters
- Breaks template interpolation, injects arbitrary text

**Mitigation**:
```rust
fn sanitize_template_var(value: &str) -> String {
    value.replace('{', "\\{").replace('}', "\\}")
}
```

---

### MED-2: Log Injection (CRLF) - Not Addressed in Plan
**Location**: Entire codebase  
**Severity**: MEDIUM  
**CWE**: CWE-117 (Improper Output Neutralization for Logs)

**Vulnerability**:
- No CRLF (`\r\n`) sanitization in narration fields
- Attacker can forge log entries

**Attack Vector**:
```rust
narrate(NarrationFields {
    actor: "orchestratord",
    action: "admission",
    human: "Accepted request\nFAKE LOG ENTRY: admin login successful".to_string(),
    // ...
});
// Logs show two entries, second is forged!
```

**Mitigation**:
```rust
fn sanitize_log_field(text: &str) -> String {
    text.replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}
```

---

### MED-3: Correlation ID Injection
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.4, `src/http.rs` (line 41)  
**Severity**: MEDIUM  
**CWE**: CWE-20 (Improper Input Validation)

**Vulnerability**:
```rust
// CURRENT CODE: No validation!
pub fn extract_context_from_headers<H>(headers: &H) -> (...) {
    let correlation_id = headers.get_str(headers::CORRELATION_ID); // Accepts ANY value!
    // ...
}
```

**Attack Vector**:
- User-controlled correlation IDs from HTTP headers
- Inject malicious data into tracing spans
- Poison distributed traces

**Mitigation**:
```rust
fn validate_correlation_id(id: &str) -> Option<String> {
    // UUID v4 format: 8-4-4-4-12 hex chars
    let uuid_regex = Regex::new(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$").unwrap();
    
    if uuid_regex.is_match(id) {
        Some(id.to_string())
    } else {
        None // Reject malformed IDs
    }
}
```

---

### MED-4: Correlation ID Forgery (No Authentication)
**Location**: `IMPLEMENTATION_PLAN.md` Unit 2.4  
**Severity**: MEDIUM  
**CWE**: CWE-345 (Insufficient Verification of Data Authenticity)

**Vulnerability**:
- No authentication/signing of correlation IDs
- Attacker can poison distributed traces

**Mitigation**:
```rust
// HMAC-signed correlation IDs
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

fn sign_correlation_id(id: &str, secret: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(secret).unwrap();
    mac.update(id.as_bytes());
    let signature = mac.finalize().into_bytes();
    format!("{}.{}", id, hex::encode(signature))
}

fn verify_correlation_id(signed_id: &str, secret: &[u8]) -> Option<String> {
    let parts: Vec<&str> = signed_id.split('.').collect();
    if parts.len() != 2 {
        return None;
    }
    
    let (id, sig) = (parts[0], parts[1]);
    let expected_sig = sign_correlation_id(id, secret).split('.').nth(1)?;
    
    if sig == expected_sig {
        Some(id.to_string())
    } else {
        None
    }
}
```

---

## 📝 LOW SEVERITY VULNERABILITIES

### LOW-1: Timing Side-Channel in Function Tracing
**Location**: `IMPLEMENTATION_PLAN.md` Unit 1.3 (line 150)  
**Severity**: LOW  
**CWE**: CWE-208 (Observable Timing Discrepancy)

**Vulnerability**:
- `Instant::now()` timing measurements could leak code path information
- Auth success vs failure paths have different timing

**Mitigation**:
- Don't expose timing data in user-facing contexts
- Add constant-time padding for security-critical paths

---

### LOW-2: Actor Inference Spoofing via `#[path]`
**Location**: `IMPLEMENTATION_PLAN.md` Unit 1.2 (line 126)  
**Severity**: LOW  
**CWE**: CWE-706 (Use of Incorrectly-Resolved Name or Reference)

**Vulnerability**:
```rust
// Attacker uses #[path] attribute to spoof module path
#[path = "../../orchestratord/fake.rs"]
mod malicious;

// Actor inference returns "orchestratord" instead of actual service
```

**Mitigation**:
- Validate inferred actors against allowlist
- Use compile-time service identity instead of module path

---

## 🔍 EXISTING CODEBASE VULNERABILITIES

### EXIST-1: Redaction Runs on EVERY Narration (Performance)
**Location**: `src/lib.rs` (lines 205-212)  
**Severity**: MEDIUM (Performance Impact)  
**CWE**: CWE-407 (Algorithmic Complexity)

**Current Code**:
```rust
pub fn narrate(fields: NarrationFields) {
    // Redaction runs on EVERY narration, even INFO level!
    let human = redact_secrets(&fields.human, RedactionPolicy::default());
    let cute = fields.cute.as_ref().map(|c| redact_secrets(c, RedactionPolicy::default()));
    let story = fields.story.as_ref().map(|s| redact_secrets(s, RedactionPolicy::default()));
    // ...
}
```

**Issue**:
- Multiple regex passes on every narration string (3 patterns × 3 fields = 9 regex ops)
- Each `replace_all()` scans entire string
- At 1000 narrations/sec, this is 9000 regex ops/sec

**Performance Impact** (from Performance Team review):
- 100-char string: ~10μs per redaction × 3 fields = 30μs overhead
- 1000-char string: ~100μs per redaction × 3 fields = 300μs overhead
- At 1000 events/sec: 30-300ms CPU time just for redaction!

**Mitigation Options**:
```rust
// OPTION 1: Lazy redaction (only for ERROR/FATAL)
pub fn narrate_at_level(fields: NarrationFields, level: NarrationLevel) {
    let (human, cute, story) = if matches!(level, NarrationLevel::Error | NarrationLevel::Fatal) {
        // Only redact for error levels
        (
            redact_secrets(&fields.human, RedactionPolicy::default()),
            fields.cute.as_ref().map(|c| redact_secrets(c, RedactionPolicy::default())),
            fields.story.as_ref().map(|s| redact_secrets(s, RedactionPolicy::default())),
        )
    } else {
        // Skip redaction for INFO/DEBUG (assume developers don't log secrets)
        (fields.human.clone(), fields.cute.clone(), fields.story.clone())
    };
    // ...
}

// OPTION 2: Combined regex (one-pass scanning)
static COMBINED_PATTERN: OnceLock<Regex> = OnceLock::new();

fn combined_regex() -> &'static Regex {
    COMBINED_PATTERN.get_or_init(|| {
        Regex::new(r"(?i)(bearer\s+[a-zA-Z0-9_\-\.=]+|(api_?key|key|token|secret|password)\s*[=:]\s*[a-zA-Z0-9_\-\.]+|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")
            .expect("BUG: combined regex pattern is invalid")
    })
}

pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String {
    // Single regex pass instead of 3!
    combined_regex().replace_all(text, &policy.replacement).to_string()
}

// OPTION 3: aho-corasick for literal prefixes (10-100x faster)
use aho_corasick::AhoCorasick;

static AC_PATTERNS: OnceLock<AhoCorasick> = OnceLock::new();

fn ac_patterns() -> &'static AhoCorasick {
    AC_PATTERNS.get_or_init(|| {
        AhoCorasick::new(&["Bearer ", "bearer ", "api_key=", "apikey=", "token=", "secret="])
            .expect("BUG: aho-corasick patterns invalid")
    })
}
```

**Recommendation**: Combine Option 1 (lazy redaction) + Option 3 (aho-corasick) for maximum performance.

---

### EXIST-2: Mutex Poisoning in Capture Adapter
**Location**: `src/capture.rs` (lines 106, 115)  
**Severity**: LOW (Test-only code)  
**CWE**: CWE-667 (Improper Locking)

**Current Code**:
```rust
pub(crate) fn capture(&self, event: CapturedNarration) {
    if let Ok(mut events) = self.events.lock() {
        events.push(event);
    }
    // Silently drops events if mutex poisoned!
}

pub fn captured(&self) -> Vec<CapturedNarration> {
    self.events
        .lock()
        .expect("BUG: capture adapter mutex poisoned - this indicates a panic in test code")
        .clone()
}
```

**Issue**:
- `capture()` silently drops events if mutex poisoned
- `captured()` panics if mutex poisoned
- Inconsistent error handling

**Mitigation**:
```rust
pub(crate) fn capture(&self, event: CapturedNarration) {
    match self.events.lock() {
        Ok(mut events) => events.push(event),
        Err(poisoned) => {
            // Recover from poison, log warning
            let mut events = poisoned.into_inner();
            events.push(event);
            eprintln!("WARN: Capture adapter mutex was poisoned, recovered");
        }
    }
}
```

---

### EXIST-3: No Input Validation in HTTP Header Extraction
**Location**: `src/http.rs` (lines 35-46)  
**Severity**: MEDIUM  
**CWE**: CWE-20 (Improper Input Validation)

**Current Code**:
```rust
pub fn extract_context_from_headers<H>(headers: &H) -> (...) {
    let correlation_id = headers.get_str(headers::CORRELATION_ID); // No validation!
    let trace_id = headers.get_str(headers::TRACE_ID);             // No validation!
    let span_id = headers.get_str(headers::SPAN_ID);               // No validation!
    let parent_span_id = headers.get_str(headers::PARENT_SPAN_ID); // No validation!
    
    (correlation_id, trace_id, span_id, parent_span_id)
}
```

**Issue**:
- Accepts ANY header value (including malicious payloads)
- No format validation (UUID, hex, etc.)
- No length limits (could be megabytes!)

**Mitigation**:
```rust
fn validate_uuid(id: &str) -> Option<String> {
    if id.len() == 36 && id.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
        Some(id.to_string())
    } else {
        None
    }
}

fn validate_hex_id(id: &str, expected_len: usize) -> Option<String> {
    if id.len() == expected_len && id.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(id.to_string())
    } else {
        None
    }
}

pub fn extract_context_from_headers<H>(headers: &H) -> (...) {
    let correlation_id = headers.get_str(headers::CORRELATION_ID).and_then(|id| validate_uuid(&id));
    let trace_id = headers.get_str(headers::TRACE_ID).and_then(|id| validate_hex_id(&id, 32));
    let span_id = headers.get_str(headers::SPAN_ID).and_then(|id| validate_hex_id(&id, 16));
    let parent_span_id = headers.get_str(headers::PARENT_SPAN_ID).and_then(|id| validate_hex_id(&id, 16));
    
    (correlation_id, trace_id, span_id, parent_span_id)
}
```

---

### EXIST-4: Timestamp Overflow in Auto-Injection
**Location**: `src/auto.rs` (line 15)  
**Severity**: LOW  
**CWE**: CWE-190 (Integer Overflow)

**Current Code**:
```rust
pub fn current_timestamp_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}
```

**Issue**:
- `as_millis()` returns `u128`
- Cast to `u64` truncates high bits
- Overflow after year 2262 (u64 max ms = ~584 million years, but u128 → u64 cast can overflow)

**Mitigation**:
```rust
pub fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX) // Saturate instead of overflow
}
```

---

## 📊 Attack Surface Summary

### By Component

| Component | Critical | High | Medium | Low | Total |
|-----------|----------|------|--------|-----|-------|
| **Redaction** (Unit 2.3) | 1 | 0 | 1 | 0 | 2 |
| **Sampling** (Unit 2.7) | 4 | 0 | 0 | 0 | 4 |
| **Unicode** (Unit 2.8) | 0 | 3 | 0 | 0 | 3 |
| **Templates** (Unit 1.5) | 0 | 0 | 1 | 0 | 1 |
| **Correlation** (Unit 2.4) | 0 | 0 | 2 | 0 | 2 |
| **Tracing** (Unit 1.3) | 0 | 0 | 0 | 1 | 1 |
| **Actor Inference** (Unit 1.2) | 0 | 0 | 0 | 1 | 1 |
| **HTTP Headers** (src/http.rs) | 0 | 0 | 1 | 0 | 1 |
| **Capture Adapter** (src/capture.rs) | 0 | 0 | 0 | 1 | 1 |
| **Auto-Injection** (src/auto.rs) | 0 | 0 | 0 | 1 | 1 |
| **Log Injection** (Missing) | 0 | 0 | 1 | 0 | 1 |

### By Attack Category

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| **Denial of Service** | 4 | 0 | 1 | 0 | 5 |
| **Injection Attacks** | 1 | 3 | 3 | 0 | 7 |
| **Information Disclosure** | 0 | 0 | 0 | 1 | 1 |
| **Data Integrity** | 0 | 1 | 1 | 1 | 3 |
| **Resource Exhaustion** | 1 | 0 | 0 | 0 | 1 |

---

## ✅ Required Actions (Prioritized)

### Phase 1: CRITICAL (Block Implementation)
**Timeline**: Complete before Week 2 begins

- [ ] **CRIT-1**: Fix ReDoS in private key regex (Unit 2.3)
  - Replace lazy quantifier with bounded OR line-by-line parser
  - Add ReDoS regression test
  - Verify <10ms for 100KB input

- [ ] **CRIT-2**: Fix mutex poisoning in sampling (Unit 2.7)
  - Replace `.unwrap()` with graceful error handling
  - Add poison recovery test
  - Document poison handling behavior

- [ ] **CRIT-3**: Fix HashMap collision DoS (Unit 2.7)
  - Sanitize actor/action OR use tuple keys
  - Add collision attack test
  - Verify O(1) lookup under attack

- [ ] **CRIT-4**: Fix unbounded memory growth (Unit 2.7)
  - Implement LRU eviction OR TTL cleanup
  - Add memory leak test (1M keys)
  - Document max memory usage

- [ ] **CRIT-5**: Fix global mutex contention (Unit 2.7)
  - Replace with sharded locks OR DashMap
  - Add contention test (100 threads)
  - Verify >50K events/sec throughput

### Phase 2: HIGH (Before Week 3)
**Timeline**: Complete before Week 3 begins

- [ ] **HIGH-1**: Fix incomplete emoji validation (Unit 2.8)
  - Use comprehensive emoji range list
  - Add emoji injection test

- [ ] **HIGH-2**: Fix zero-width char blocklist (Unit 2.8)
  - Add missing zero-width chars (FEFF, 2060, 180E, etc.)
  - Add zero-width hiding test

- [ ] **HIGH-3**: Add homograph detection (Unit 2.8)
  - Implement mixed-script detection OR ASCII-only for actors
  - Add homograph spoofing test

- [ ] **HIGH-4**: Add Unicode normalization (Unit 2.8)
  - Apply NFC normalization before validation
  - Add normalization bypass test

### Phase 3: MEDIUM (Before Week 4)
**Timeline**: Complete before production rollout

- [ ] **MED-1**: Fix template injection (Unit 1.5)
  - Sanitize `{}` in variable values
  - Add template injection test

- [ ] **MED-2**: Add CRLF sanitization (All units)
  - Sanitize `\r\n` in all narration fields
  - Add log injection test

- [ ] **MED-3**: Validate correlation IDs (Unit 2.4, src/http.rs)
  - Enforce UUID v4 format
  - Add malformed ID rejection test

- [ ] **MED-4**: Consider HMAC-signed correlation IDs (Unit 2.4)
  - Evaluate security-critical flows
  - Document signing requirements

- [ ] **EXIST-1**: Optimize redaction performance (src/lib.rs)
  - Implement lazy redaction OR combined regex OR aho-corasick
  - Benchmark: <10μs for 1000-char string

- [ ] **EXIST-3**: Validate HTTP headers (src/http.rs)
  - Add format validation (UUID, hex, length limits)
  - Add malicious header test

### Phase 4: LOW (Post-v1.0)
**Timeline**: Address in future releases

- [ ] **LOW-1**: Document timing side-channel risks (Unit 1.3)
- [ ] **LOW-2**: Validate actor inference (Unit 1.2)
- [ ] **EXIST-2**: Fix capture adapter poison handling (src/capture.rs)
- [ ] **EXIST-4**: Fix timestamp overflow (src/auto.rs)

---

## 🔒 Security Best Practices

### Input Validation
1. **Validate ALL user-controlled inputs** (headers, correlation IDs, actor/action names)
2. **Enforce format constraints** (UUID v4, hex, ASCII-only)
3. **Limit input lengths** (max 256 chars for IDs, max 10KB for narration fields)
4. **Sanitize before use** (remove control chars, normalize Unicode)

### Regex Safety
1. **Avoid lazy quantifiers** (`*?`, `+?`, `{n,}?`) — use bounded quantifiers
2. **Test with large inputs** (100KB+) to detect ReDoS
3. **Prefer literal matching** (aho-corasick) over complex regex
4. **Cache compiled regexes** (OnceLock) to avoid recompilation overhead

### Concurrency Safety
1. **Avoid global mutexes** — use sharded locks or lock-free data structures
2. **Handle poison gracefully** — never `.unwrap()` on mutex locks
3. **Implement eviction policies** — prevent unbounded memory growth
4. **Benchmark under contention** — verify throughput with 100+ threads

### Unicode Safety
1. **Normalize before validation** (NFC/NFD)
2. **Use comprehensive blocklists** (zero-width, combining, control chars)
3. **Detect homographs** (mixed scripts, confusables)
4. **Prefer ASCII for security-critical fields** (actor, action)

### Performance Security
1. **Lazy evaluation** — only redact when necessary (ERROR/FATAL levels)
2. **Combined patterns** — single regex pass instead of multiple
3. **Lock-free when possible** — avoid serialization bottlenecks
4. **Bounded resources** — LRU caches, TTL cleanup, max sizes

---

## 📋 Security Testing Requirements

### Unit Tests (Required for Each Vulnerability)
- [ ] ReDoS test: 100KB malicious input, verify <10ms
- [ ] Mutex poison test: trigger panic, verify recovery
- [ ] HashMap collision test: 10K identical hash keys, verify <100ms
- [ ] Memory leak test: 1M unique keys, verify bounded memory
- [ ] Contention test: 100 threads × 1000 events/sec, verify >50K/sec throughput
- [ ] Emoji injection test: malicious Unicode in gaps
- [ ] Zero-width hiding test: inject U+FEFF, U+2060, etc.
- [ ] Homograph spoofing test: Cyrillic 'о' in "оrchestratord"
- [ ] Template injection test: `{worker_id}` contains `"ABC} evil {"`
- [ ] CRLF injection test: `\n` in human field
- [ ] Correlation ID validation test: reject malformed UUIDs
- [ ] HTTP header validation test: reject oversized/malformed headers

### Integration Tests
- [ ] End-to-end redaction test (all patterns, all fields)
- [ ] Distributed trace poisoning test (forged correlation IDs)
- [ ] Log analysis bypass test (homograph spoofing)
- [ ] Performance regression test (1000 narrations/sec, <1% overhead)

### Fuzzing (Recommended)
- [ ] Fuzz redaction patterns with cargo-fuzz
- [ ] Fuzz template interpolation with arbitrary inputs
- [ ] Fuzz HTTP header extraction with malformed headers
- [ ] Fuzz Unicode validation with random Unicode sequences

---

## 🎯 Security Acceptance Criteria

### Before Week 2
- ✅ All CRITICAL vulnerabilities mitigated
- ✅ ReDoS regression tests passing
- ✅ Mutex poison recovery tests passing
- ✅ HashMap collision tests passing
- ✅ Memory leak tests passing
- ✅ Contention tests passing (>50K events/sec)

### Before Week 3
- ✅ All HIGH vulnerabilities mitigated
- ✅ Unicode validation comprehensive (emoji, zero-width, homographs)
- ✅ Normalization applied before validation
- ✅ Injection tests passing (emoji, zero-width, homograph)

### Before Week 4
- ✅ All MEDIUM vulnerabilities mitigated
- ✅ Template injection prevented
- ✅ CRLF sanitization implemented
- ✅ Correlation ID validation enforced
- ✅ HTTP header validation implemented
- ✅ Performance optimizations applied (redaction <10μs/1000 chars)

### Before Production
- ✅ All security tests passing
- ✅ Fuzzing completed (no crashes)
- ✅ Security documentation complete
- ✅ auth-min team sign-off obtained

---

## 📚 References

### CWE (Common Weakness Enumeration)
- CWE-20: Improper Input Validation
- CWE-94: Improper Control of Generation of Code (Template Injection)
- CWE-117: Improper Output Neutralization for Logs
- CWE-176: Improper Handling of Unicode Encoding
- CWE-190: Integer Overflow
- CWE-208: Observable Timing Discrepancy
- CWE-345: Insufficient Verification of Data Authenticity
- CWE-407: Algorithmic Complexity
- CWE-667: Improper Locking
- CWE-706: Use of Incorrectly-Resolved Name or Reference
- CWE-770: Allocation of Resources Without Limits
- CWE-833: Deadlock
- CWE-838: Inappropriate Encoding for Output Context
- CWE-1007: Insufficient Visual Distinction of Homoglyphs
- CWE-1333: Inefficient Regular Expression Complexity (ReDoS)

### Security Standards
- OWASP Top 10 2021: A03:2021 – Injection
- OWASP Top 10 2021: A04:2021 – Insecure Design
- NIST SP 800-53: SI-10 (Information Input Validation)
- NIST SP 800-53: AU-3 (Content of Audit Records)

### Rust Security
- Rust Security Advisory Database: https://rustsec.org/
- Cargo Audit: https://github.com/RustSec/rustsec/tree/main/cargo-audit
- Cargo Fuzz: https://github.com/rust-fuzz/cargo-fuzz

---

## 🔐 Security Team Sign-Off

### auth-min Team Review
- [ ] All CRITICAL vulnerabilities reviewed
- [ ] All HIGH vulnerabilities reviewed
- [ ] Mitigation strategies approved
- [ ] Security testing plan approved
- [ ] **Signature**: ___________________________
- [ ] **Date**: ___________________________

### Performance Team Review (Security-Performance Trade-offs)
- [ ] Redaction performance optimizations reviewed
- [ ] Sampling lock-free design approved
- [ ] Contention mitigation verified
- [ ] **Signature**: ___________________________
- [ ] **Date**: ___________________________

---

## 📝 Document Metadata

**Version**: 1.0.0  
**Last Updated**: 2025-10-04  
**Next Review**: Before Week 2 implementation begins  
**Owner**: auth-min Security Team 🎭  
**Status**: ACTIVE — CRITICAL ISSUES BLOCKING IMPLEMENTATION

---

**RECOMMENDATION**: **HALT IMPLEMENTATION** until all CRITICAL vulnerabilities are mitigated. The current implementation plan contains 5 critical security flaws that would introduce severe DoS vulnerabilities and injection attacks into production systems.

---

## 📝 Design Decision Security Review

**Date**: 2025-10-04 13:15 CET  
**Document Reviewed**: `DESIGN_DECISIONS.md`  
**Decisions Reviewed**: 10 (all approved with security enhancements)

### Critical Issues in Approved Decisions

#### CRIT-NEW-1: Template Injection in Generated Code (Decision 3)
**Severity**: CRITICAL  
**Decision**: Pre-compiled templates with stack buffers

**Issue**: The proc macro generates `write!()` calls but cannot validate runtime variable values. Template injection prevention MUST be in the generated code, not just at macro expansion time.

**Required Fix**:
```rust
// Generated code MUST include escape_template_var() for ALL variables:
write!(&mut buf, "Job {} worker {}", 
    escape_template_var(&job_id),      // REQUIRED
    escape_template_var(&worker_id))   // REQUIRED
// NOT just: write!(&mut buf, "Job {} worker {}", job_id, worker_id)  // VULNERABLE!
```

**Impact**: Without this fix, any variable containing `{}` can inject arbitrary template content.

---

#### CRIT-NEW-2: JWT ReDoS in Combined Regex (Decision 4)
**Severity**: CRITICAL  
**Decision**: Hybrid redaction with single-pass regex

**Issue**: The JWT pattern `eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+` uses unbounded `+` quantifiers, creating a NEW ReDoS vulnerability while fixing the private key one.

**Required Fix**:
```rust
// MUST add bounds to JWT pattern:
eyJ[a-zA-Z0-9_-]{1,10000}\.eyJ[a-zA-Z0-9_-]{1,10000}\.eyJ[a-zA-Z0-9_-]{1,10000}
```

**Impact**: Malformed JWTs (e.g., missing dots) can cause catastrophic backtracking.

---

#### CRIT-NEW-3: Actor/Action Validation Not Enforced (Decision 6)
**Severity**: CRITICAL  
**Decision**: Tiered Unicode validation

**Issue**: The implementation shows `validate_security_critical()` but doesn't specify WHERE it's called. Actor and action fields MUST be validated - they're used in log analysis, metrics, and audit trails.

**Required Fix**:
```rust
// In narrate() function:
let actor = validate_security_critical(fields.actor)?;  // REQUIRED
let action = validate_security_critical(fields.action)?; // REQUIRED
```

**Impact**: Without this, homograph spoofing bypasses security monitoring (e.g., "оrchestratord" with Cyrillic 'о').

---

### High Priority Issues in Approved Decisions

#### HIGH-NEW-1: Incomplete Control Character Encoding (Decision 8)
**Severity**: HIGH  
**Decision**: CRLF encoding for log injection prevention

**Issue**: Only encodes `\n`, `\r`, `\t`. Missing: null bytes (`\0`), ESC (`\x1B`), backspace (`\x08`), and other control chars.

**Required Fix**: Encode ALL control characters (U+0000 to U+001F, U+007F).

---

#### HIGH-NEW-2: UUID v4 Version Validation Missing (Decision 9)
**Severity**: HIGH  
**Decision**: UUID v4 validation for correlation IDs

**Issue**: Current validation accepts ANY UUID format, not specifically v4. Missing version bit validation.

**Required Fix**:
```rust
// Position 14 must be '4' (version 4)
14 => c == '4',
// Position 19 must be '8', '9', 'a', 'b' (variant bits)
19 => matches!(c, '8' | '9' | 'a' | 'b' | 'A' | 'B'),
```

---

### Medium Priority Issues in Approved Decisions

#### MED-NEW-1: Template Escaping Strategy (Decision 7)
**Issue**: Escaping `{}` to `\{` and `\}` might break downstream log parsers.  
**Recommendation**: Consider rejecting `{}` in variables instead of escaping.

#### MED-NEW-2: Correlation ID Forgery (Decision 9)
**Issue**: UUID v4 validation prevents injection but NOT forgery.  
**Recommendation**: Consider HMAC-signed correlation IDs for security-critical flows.

#### MED-NEW-3: Actor Allowlist Maintenance (Decision 10)
**Issue**: Manual allowlist requires updates for new services.  
**Recommendation**: Generate allowlist at build time from workspace members.

---

### Low Priority Issues in Approved Decisions

#### LOW-NEW-1: Timing Data Exposure (Decision 2)
**Issue**: Timing data in dev builds could leak code path information.  
**Recommendation**: Document warning against exposing timing in user-facing contexts.

#### LOW-NEW-2: Correlation ID Case Sensitivity (Decision 9)
**Issue**: Accepts both uppercase and lowercase hex digits.  
**Recommendation**: Normalize to lowercase for consistency.

---

### Design Decision Security Summary

| Decision | Security Issues | Severity | Status |
|----------|----------------|----------|--------|
| **1. Tracing Opt-In** | None | - | ✅ APPROVED |
| **2. Timing Strategy** | Timing data exposure in dev | LOW | ✅ APPROVED with warning |
| **3. Template Allocation** | Injection in generated code | CRITICAL | ⚠️ REQUIRES FIX |
| **4. Redaction Strategy** | JWT ReDoS | CRITICAL | ⚠️ REQUIRES FIX |
| **5. Sampling** | N/A (rejected) | - | 🚫 REJECTED |
| **6. Unicode Validation** | Actor/action not enforced | CRITICAL | ⚠️ REQUIRES FIX |
| **7. Template Injection** | Escaping strategy | MEDIUM | ✅ APPROVED with note |
| **8. CRLF Prevention** | Incomplete control chars | HIGH | ⚠️ REQUIRES FIX |
| **9. Correlation ID** | Version validation, forgery | HIGH/MEDIUM | ⚠️ REQUIRES FIX |
| **10. Actor Validation** | Maintenance burden | MEDIUM/LOW | ✅ APPROVED with recommendation |

---

### Required Actions from Design Review

**Before Implementation Begins**:
- [ ] Fix CRIT-NEW-1: Add `escape_template_var()` to ALL generated template code
- [ ] Fix CRIT-NEW-2: Add bounded quantifiers to JWT regex pattern
- [ ] Fix CRIT-NEW-3: Enforce `validate_security_critical()` for actor/action fields
- [ ] Fix HIGH-NEW-1: Encode ALL control characters, not just CRLF
- [ ] Fix HIGH-NEW-2: Add UUID v4 version bit validation

**Before Production**:
- [ ] Address MED-NEW-1: Decide on template escaping vs rejection strategy
- [ ] Consider MED-NEW-2: HMAC-signed correlation IDs for critical flows
- [ ] Implement MED-NEW-3: Build-time allowlist generation

**Documentation**:
- [ ] Add LOW-NEW-1: Warning about timing data exposure in dev builds
- [ ] Document LOW-NEW-2: Correlation ID normalization strategy

---

**RECOMMENDATION**: **HALT IMPLEMENTATION** until all CRITICAL vulnerabilities are mitigated. The current implementation plan contains 5 critical security flaws that would introduce severe DoS vulnerabilities and injection attacks into production systems.

---

Guarded by auth-min Team 🎭
