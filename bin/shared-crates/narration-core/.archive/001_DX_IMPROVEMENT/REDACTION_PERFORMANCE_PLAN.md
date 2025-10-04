# Redaction Performance Improvement Plan

**Created**: 2025-10-04  
**Owner**: Performance Team  
**Priority**: P0 (Critical)  
**Target**: narration-core v0.2.0

---

## Executive Summary

**Claim in README**: Redaction takes ~180ms for 200-char strings (36,000x slower than <5μs target)  
**Actual Benchmark Results**: Redaction takes ~430ns-1.4μs (already meets or exceeds target!)  
**Conclusion**: ✅ **Performance is ALREADY EXCELLENT** - README needs correction, not code!

---

## Current Performance (Measured)

### Benchmark Results

Running `cargo bench -p observability-narration-core --bench narration_benchmarks redaction`:

```
redaction/clean_1000_chars:     605 ns  (1000 chars, no secrets)
redaction/with_bearer_token:    431 ns  (32 chars, 1 secret)
redaction/with_multiple_secrets: 1.36 µs (150+ chars, 3 secrets)
```

### Analysis

| Scenario | Size | Secrets | Time | vs Target | Status |
|----------|------|---------|------|-----------|--------|
| Clean text | 1000 chars | 0 | 605 ns | 8x faster | ✅ Exceeds |
| Bearer token | 32 chars | 1 | 431 ns | 11x faster | ✅ Exceeds |
| Multiple secrets | 150 chars | 3 | 1.36 µs | 3.7x faster | ✅ Exceeds |

**Conclusion**: Current implementation already exceeds the <5μs target!

---

## Root Cause Analysis

### Why README Claims 180ms

**Hypothesis 1**: Outdated benchmark data from early implementation  
**Hypothesis 2**: Measurement error (included compilation time)  
**Hypothesis 3**: Different test case (very large strings with many secrets)

### Current Implementation Analysis

**File**: `src/redaction.rs` (lines 107-135)

```rust
pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String {
    let mut result = text.to_string();  // ← Single allocation

    if policy.mask_bearer_tokens {
        result = bearer_token_regex().replace_all(&result, &policy.replacement).to_string();
    }
    // ... 5 more patterns
    
    result
}
```

**Performance Characteristics**:

1. ✅ **Regex compiled once** (lines 41-46): Uses `OnceLock` for zero-cost reuse
2. ✅ **Bounded quantifiers** (lines 51, 59, 75, 83): No ReDoS vulnerability
3. ✅ **Early allocation** (line 108): Single `to_string()` upfront
4. ⚠️ **Multiple passes** (lines 110-132): 6 separate regex passes
5. ⚠️ **Intermediate allocations** (lines 111, 115, etc.): Each `replace_all().to_string()` allocates

### Bottleneck Identification

**Current approach** (6 passes):
```rust
result = bearer_token_regex().replace_all(&result, &policy.replacement).to_string();  // Pass 1
result = api_key_regex().replace_all(&result, &policy.replacement).to_string();       // Pass 2
result = uuid_regex().replace_all(&result, &policy.replacement).to_string();          // Pass 3
// ... 3 more passes
```

**Issues**:
- 6 separate regex scans of the entire string
- 6 intermediate String allocations
- Each pass must scan unchanged portions

**Optimization potential**: Single-pass with multi-pattern matching

---

## Improvement Strategies

### Strategy 1: Single-Pass Multi-Pattern (Recommended) ⭐

**Approach**: Use `aho-corasick` for fast multi-pattern matching

**Benefits**:
- Single pass over the string
- Single allocation
- O(n) complexity (vs current O(6n))
- 2-3x faster for typical cases

**Implementation**:

```rust
use aho_corasick::AhoCorasick;
use std::sync::OnceLock;

static PATTERN_MATCHER: OnceLock<AhoCorasick> = OnceLock::new();

fn get_pattern_matcher() -> &'static AhoCorasick {
    PATTERN_MATCHER.get_or_init(|| {
        // Patterns to detect (not the full secret, just the prefix)
        let patterns = vec![
            "Bearer ",
            "bearer ",
            "BEARER ",
            "api_key=",
            "apikey=",
            "API_KEY=",
            "key=",
            "token=",
            "password=",
            "secret=",
            "eyJ",  // JWT prefix
            "-----BEGIN",  // Private key prefix
            "://",  // URL (for password detection)
        ];
        
        AhoCorasick::new(patterns).unwrap()
    })
}

pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String {
    let matcher = get_pattern_matcher();
    
    // Fast path: no patterns found
    if !matcher.is_match(text) {
        return text.to_string();
    }
    
    // Single-pass redaction
    let mut result = String::with_capacity(text.len());
    let mut last_pos = 0;
    
    for mat in matcher.find_iter(text) {
        let start = mat.start();
        
        // Copy text before match
        result.push_str(&text[last_pos..start]);
        
        // Determine secret type and extract full secret
        let secret_end = match mat.pattern().as_usize() {
            0..=2 => extract_bearer_token(&text[start..]),  // Bearer variants
            3..=9 => extract_key_value(&text[start..]),     // key=value patterns
            10 => extract_jwt(&text[start..]),              // JWT
            11 => extract_private_key(&text[start..]),      // Private key
            12 => extract_url_password(&text[start..]),     // URL password
            _ => 0,
        };
        
        // Add redaction marker
        result.push_str(&policy.replacement);
        
        // Skip the secret
        last_pos = start + secret_end;
    }
    
    // Copy remaining text
    result.push_str(&text[last_pos..]);
    
    result
}

// Helper functions to extract full secret after pattern match
fn extract_bearer_token(text: &str) -> usize {
    // "Bearer <token>" - extract until whitespace
    text.find(|c: char| c.is_whitespace() && c != ' ')
        .unwrap_or_else(|| text.len())
}

fn extract_key_value(text: &str) -> usize {
    // "key=value" - extract until whitespace or &
    text.find(|c: char| c.is_whitespace() || c == '&')
        .unwrap_or_else(|| text.len())
}

fn extract_jwt(text: &str) -> usize {
    // JWT format: eyJ...eyJ...<signature>
    // Find end of signature (whitespace or end)
    text.find(|c: char| c.is_whitespace())
        .unwrap_or_else(|| text.len())
}

fn extract_private_key(text: &str) -> usize {
    // Find "-----END ... KEY-----"
    text.find("-----END")
        .and_then(|pos| text[pos..].find("-----").map(|end| pos + end + 5))
        .unwrap_or_else(|| text.len())
}

fn extract_url_password(text: &str) -> usize {
    // Find @ after password
    text.find('@')
        .map(|pos| pos + 1)
        .unwrap_or_else(|| text.len())
}
```

**Expected Performance**:
- Clean text (no secrets): ~100ns (fast path, single `is_match` check)
- With bearer token: ~200ns (single pass, single allocation)
- With multiple secrets: ~500ns (single pass, single allocation)

**Improvement**: 2-3x faster than current implementation

---

### Strategy 2: Lazy Evaluation (Alternative)

**Approach**: Only scan if policy enables specific patterns

**Current code** (lines 110-132):
```rust
if policy.mask_bearer_tokens {
    result = bearer_token_regex().replace_all(&result, &policy.replacement).to_string();
}
```

**Issue**: Even with `if` guard, regex is compiled and string is scanned.

**Optimization**: Early exit if no patterns enabled

```rust
pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String {
    // Fast path: no redaction enabled
    if !policy.mask_bearer_tokens 
        && !policy.mask_api_keys 
        && !policy.mask_jwt_tokens 
        && !policy.mask_private_keys 
        && !policy.mask_url_passwords 
        && !policy.mask_uuids {
        return text.to_string();
    }
    
    // ... existing logic
}
```

**Expected Performance**: Minimal improvement (~5-10%) since policies are usually enabled.

---

### Strategy 3: Cow<'_, str> for Zero-Copy (Alternative)

**Approach**: Return `Cow<'_, str>` to avoid allocation when no secrets found

**Current**:
```rust
pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String
```

**Proposed**:
```rust
pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> Cow<'_, str>
```

**Benefits**:
- Zero allocation when no secrets (common case)
- Automatic conversion to `String` when needed

**Drawbacks**:
- Breaking API change
- Caller complexity increases

**Recommendation**: Defer to v0.3.0 (breaking change)

---

## Recommended Action Plan

### Phase 1: Verify Current Performance ✅ COMPLETE

**Status**: ✅ Done (benchmarks show 431ns-1.36µs)

### Phase 2: Update Documentation 📋 IMMEDIATE

**Priority**: P0  
**Effort**: 15 minutes  
**Files**: `README.md`

**Changes**:

1. Update performance section (lines 629-633):
```markdown
### Redaction Performance
- **Target**: <5μs for strings with secrets
- **Actual**: ~430ns for single secret, ~1.4μs for multiple secrets
- **Status**: ✅ Exceeds target by 3-11x
```

2. Remove from roadmap (line 668):
```markdown
### v0.2.0 (Next)
- [ ] ~~Optimize redaction performance (36,000x improvement needed)~~  ← DELETE
- [ ] Add more property tests for edge cases
```

### Phase 3: Optimize Further (Optional) 📋 TODO

**Priority**: P2 (Nice to have, not critical)  
**Effort**: 4 hours  
**Target**: 2-3x additional improvement

**Implementation**: Strategy 1 (Single-pass multi-pattern)

**Steps**:

1. Add `aho-corasick` dependency:
```toml
[dependencies]
aho-corasick = "1.1"
```

2. Implement single-pass redaction (see Strategy 1 code above)

3. Add benchmark comparison:
```rust
#[bench]
fn bench_redaction_single_pass(b: &mut Bencher) {
    let text = "Bearer token123 and api_key=secret456";
    b.iter(|| redact_secrets_single_pass(black_box(text), RedactionPolicy::default()));
}
```

4. Verify no regression:
```bash
cargo bench -p observability-narration-core --bench narration_benchmarks
```

**Expected Results**:
- Clean text: 605ns → ~100ns (6x faster)
- Bearer token: 431ns → ~200ns (2x faster)
- Multiple secrets: 1.36µs → ~500ns (2.7x faster)

---

## Code References

### Current Implementation

**File**: `src/redaction.rs`

**Key sections**:
- **Lines 41-46**: Regex pattern caching with `OnceLock` ✅
- **Lines 48-94**: Regex pattern definitions (6 patterns)
- **Lines 107-135**: Main `redact_secrets` function (6-pass approach)
- **Lines 138-201**: Unit tests (8 tests, all passing)

**Regex patterns**:
1. **Line 51**: Bearer token - `r"(?i)bearer\s+[a-zA-Z0-9_\-\.=]+"`
2. **Line 59**: API key - `r"(?i)(api_?key|key|token|secret|password)\s*[=:]\s*[a-zA-Z0-9_\-\.]+"`
3. **Line 67**: UUID - `r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"`
4. **Line 75**: JWT - `r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+"`
5. **Line 83**: Private key - `r"-----BEGIN [A-Z ]+PRIVATE KEY-----[\s\S]{1,4096}?-----END [A-Z ]+PRIVATE KEY-----"`
6. **Line 91**: URL password - `r"://[^:]+:([^@]+)@"`

### Benchmark Code

**File**: `benches/narration_benchmarks.rs`

**Key sections**:
- **Lines 19-48**: Redaction benchmarks
- **Line 23**: Clean text test (1000 chars)
- **Line 32**: Bearer token test
- **Line 40**: Multiple secrets test

### Performance Bottlenecks

**Identified issues**:

1. **Multiple allocations** (lines 111, 115, 119, 123, 127, 131):
```rust
result = bearer_token_regex().replace_all(&result, &policy.replacement).to_string();
//                                                                       ^^^^^^^^^^^ allocation
result = api_key_regex().replace_all(&result, &policy.replacement).to_string();
//                                                                  ^^^^^^^^^^^ allocation
// ... 4 more allocations
```

**Impact**: 6 allocations per call (even if no matches)

2. **Multiple regex scans** (lines 110-132):
```rust
if policy.mask_bearer_tokens {
    result = bearer_token_regex().replace_all(&result, &policy.replacement).to_string();
}
if policy.mask_api_keys {
    result = api_key_regex().replace_all(&result, &policy.replacement).to_string();
}
// ... 4 more scans
```

**Impact**: 6 full string scans (O(6n) complexity)

3. **No fast path for clean strings**:
```rust
pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String {
    let mut result = text.to_string();  // ← Always allocates
    // ...
}
```

**Impact**: Allocates even when no secrets present

---

## Improvement Strategy

### Immediate Action: Update Documentation ⚠️

**Priority**: P0  
**Effort**: 5 minutes  
**Impact**: Removes false performance concern

**File**: `README.md` (lines 629-633)

**Current**:
```markdown
### Redaction Performance
- **Target**: <5μs for strings with secrets
- **Current**: ~180ms for 200-char strings
- **Status**: ⚠️ Optimization scheduled for v0.2.0
```

**Corrected**:
```markdown
### Redaction Performance
- **Target**: <5μs for strings with secrets
- **Actual**: ~430ns for single secret, ~1.4µs for multiple secrets
- **Status**: ✅ Exceeds target by 3-11x
- **Benchmark**: `cargo bench -p observability-narration-core redaction`
```

---

### Optional Optimization: Single-Pass Algorithm

**Priority**: P2 (Nice to have)  
**Effort**: 4 hours  
**Expected Improvement**: 2-3x faster

#### Step 1: Add Fast Path for Clean Strings

**File**: `src/redaction.rs` (after line 107)

```rust
pub fn redact_secrets(text: &str, policy: RedactionPolicy) -> String {
    // Fast path: check if any pattern might match
    if !text.contains("Bearer") 
        && !text.contains("bearer")
        && !text.contains("api_key")
        && !text.contains("key=")
        && !text.contains("token=")
        && !text.contains("eyJ")  // JWT prefix
        && !text.contains("-----BEGIN")  // Private key
        && !text.contains("://") {  // URL
        return text.to_string();  // No secrets possible
    }
    
    // ... existing logic
}
```

**Expected**: Clean strings: 605ns → ~50ns (12x faster)

#### Step 2: Reduce Intermediate Allocations

**File**: `src/redaction.rs` (lines 110-132)

**Current**:
```rust
if policy.mask_bearer_tokens {
    result = bearer_token_regex().replace_all(&result, &policy.replacement).to_string();
}
```

**Optimized**:
```rust
if policy.mask_bearer_tokens {
    let cow = bearer_token_regex().replace_all(&result, &policy.replacement);
    if matches!(cow, std::borrow::Cow::Owned(_)) {
        result = cow.into_owned();
    }
}
```

**Expected**: Avoids allocation when no match found

#### Step 3: Implement Single-Pass with aho-corasick

**File**: `src/redaction_v2.rs` (NEW)

**Full implementation** (see Strategy 1 above for complete code)

**Expected Performance**:
- Clean text: 605ns → ~100ns (6x faster)
- Bearer token: 431ns → ~200ns (2x faster)
- Multiple secrets: 1.36µs → ~500ns (2.7x faster)

**Trade-offs**:
- More complex code (~200 lines vs ~135 lines)
- Additional dependency (`aho-corasick`)
- More helper functions to maintain

---

## Testing Strategy

### Benchmark Comparison

**File**: `benches/redaction_comparison.rs` (NEW)

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use observability_narration_core::{redact_secrets, RedactionPolicy};

fn bench_current_vs_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("redaction_comparison");
    
    let test_cases = vec![
        ("clean_100", "Accepted request; queued at position 3".repeat(5)),
        ("bearer", "Authorization: Bearer abc123xyz"),
        ("multiple", "Bearer token123 and api_key=secret456 and password=pass789"),
    ];
    
    for (name, text) in test_cases {
        group.bench_function(&format!("current_{}", name), |b| {
            b.iter(|| redact_secrets(black_box(&text), RedactionPolicy::default()));
        });
        
        group.bench_function(&format!("optimized_{}", name), |b| {
            b.iter(|| redact_secrets_v2(black_box(&text), RedactionPolicy::default()));
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_current_vs_optimized);
criterion_main!(benches);
```

### Property Tests (Verify Correctness)

**File**: `tests/property_tests.rs` (add to existing)

```rust
#[test]
fn property_optimized_redaction_equivalent_to_current() {
    let test_cases = vec![
        "Bearer abc123",
        "api_key=secret",
        "Bearer token1 and api_key=secret2",
        "Clean text with no secrets",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature",
    ];
    
    for text in test_cases {
        let current = redact_secrets(text, RedactionPolicy::default());
        let optimized = redact_secrets_v2(text, RedactionPolicy::default());
        
        assert_eq!(
            current, optimized,
            "Redaction mismatch for input: {}",
            text
        );
    }
}
```

---

## Risk Assessment

### Low Risk ✅
- **Current implementation already meets target**
- **Optimization is optional, not required**
- **Existing tests provide regression protection**

### Medium Risk (If Optimizing)
- **Complexity increase**: Single-pass algorithm more complex
- **Mitigation**: Comprehensive property tests, side-by-side comparison

### No Risk
- **Documentation update**: Zero code change, zero risk

---

## Acceptance Criteria

### Phase 1: Documentation Update (REQUIRED)
- [ ] README performance section corrected
- [ ] Roadmap item removed
- [ ] Benchmark command documented
- [ ] No false performance concerns

### Phase 2: Optional Optimization (NICE TO HAVE)
- [ ] Fast path for clean strings (<100ns)
- [ ] Single-pass algorithm implemented
- [ ] All existing tests pass
- [ ] Property tests verify equivalence
- [ ] Benchmark shows 2-3x improvement
- [ ] No regression in any scenario

---

## Decision Matrix

| Approach | Effort | Improvement | Complexity | Recommendation |
|----------|--------|-------------|------------|----------------|
| **Update docs** | 5 min | N/A | None | ✅ DO NOW |
| **Fast path** | 30 min | 12x (clean) | Low | ✅ DO (easy win) |
| **Reduce allocs** | 1 hour | 1.2x | Low | ✅ DO (small win) |
| **Single-pass** | 4 hours | 2-3x | High | ⚠️ DEFER (diminishing returns) |
| **Cow<str>** | 2 hours | 1.5x | Medium | ❌ SKIP (breaking change) |

---

## Conclusion

### Critical Finding ⚠️

**The README is wrong!** Redaction performance is already excellent:
- ✅ 431ns for bearer token (11x faster than 5µs target)
- ✅ 1.36µs for multiple secrets (3.7x faster than target)
- ✅ 605ns for clean 1000-char strings

### Recommended Actions

1. **Immediate** (5 min): Update README to reflect actual performance
2. **Optional** (30 min): Add fast path for clean strings (12x improvement)
3. **Optional** (1 hour): Reduce intermediate allocations (1.2x improvement)
4. **Defer**: Single-pass algorithm (complex, diminishing returns)

### Impact

**Before**: Developers think redaction is 36,000x too slow (blocks adoption)  
**After**: Developers know redaction exceeds performance targets (confidence)

---

**The real problem is documentation, not code.** Fix the docs first! 📝
