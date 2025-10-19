# narration-core Contract Audit — Developer Experience

**Audit Date**: 2025-10-04  
**Auditor**: Developer Experience Team  
**Crate**: `bin/shared-crates/narration-core` v0.1.0  
**Audit Scope**: API usability, integration friction, documentation clarity

---

## Executive Summary

**Overall Grade**: B+ (Good, with friction points)

**Strengths**:
- ✅ Comprehensive README with examples
- ✅ Clear testing patterns with `CaptureAdapter`
- ✅ HTTP context helpers reduce boilerplate
- ✅ Auto-injection simplifies common case

**Friction Points**:
- ⚠️ Inconsistent API patterns (3 different ways to narrate)
- ⚠️ Verbose struct initialization for simple cases
- ⚠️ Unclear when to use which narration method
- ⚠️ Missing integration guide for correlation ID middleware
- ⚠️ No policy for what should/shouldn't be narrated

---

## API Usability Analysis

### Pattern 1: Basic Narration (Verbose)

```rust
use observability_narration_core::{narrate, NarrationFields};

narrate(NarrationFields {
    actor: "queen-rbee",
    action: "enqueue",
    target: job_id.to_string(),
    human: format!("Enqueued job {job_id}"),
    correlation_id: Some(req_id),
    ..Default::default()  // ← Requires Default trait
});
```

**Issues**:
- ❌ Verbose: 7 lines for one log event
- ❌ `..Default::default()` is boilerplate
- ❌ String literals not validated at compile time
- ❌ Easy to forget correlation_id

**Developer Friction**: HIGH

### Pattern 2: Auto-Injection (Better)

```rust
use observability_narration_core::auto::narrate_auto;

narrate_auto(NarrationFields {
    actor: "pool-managerd",
    action: "provision",
    target: pool_id.to_string(),
    human: format!("Provisioning pool {pool_id}"),
    ..Default::default()
});
```

**Issues**:
- ✅ Adds `emitted_by` and `emitted_at_ms` automatically
- ❌ Still verbose (6 lines)
- ❌ Still requires `..Default::default()`
- ❌ No guidance on when to use vs. basic narration

**Developer Friction**: MEDIUM

### Pattern 3: Macro (Mentioned but not shown)

```rust
use narration_core::{narrate_with_correlation, Actor, Action};

narrate_with_correlation!(
    correlation_id = req_id,
    actor = Actor::PoolManagerd,
    action = Action::Provision,
    target = pool_id,
    human = "Provisioning pool {pool_id}"
);
```

**Issues**:
- ✅ More concise (5 lines)
- ✅ Enums for actor/action (type safety)
- ❌ Macro not documented in README
- ❌ Unclear if macro exists or is aspirational
- ❌ No examples of macro usage

**Developer Friction**: UNKNOWN (not documented)

---

## Integration Friction Points

### 1. Correlation ID Extraction

**README shows**:
```rust
use observability_narration_core::http::{extract_context_from_headers, HeaderLike};

impl HeaderLike for MyRequest {
    fn get_header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(|v| v.as_str())
    }
}

let context = extract_context_from_headers(&request);
```

**Issues**:
- ❌ Requires implementing `HeaderLike` trait
- ❌ No example for Axum (most common framework)
- ❌ No middleware example (common use case)
- ❌ Validation logic not shown (`validate_correlation_id`)

**What developers actually need**:
```rust
// Axum middleware example (MISSING FROM README)
use axum::{extract::Request, middleware::Next, response::Response};
use observability_narration_core::{generate_correlation_id, validate_correlation_id};

pub async fn correlation_middleware(mut req: Request, next: Next) -> Response {
    let correlation_id = req.headers()
        .get("X-Correlation-ID")
        .and_then(|v| v.to_str().ok())
        .and_then(|id| validate_correlation_id(id).map(|_| id.to_string()))
        .unwrap_or_else(|| generate_correlation_id());
    
    req.extensions_mut().insert(correlation_id.clone());
    // ... rest of middleware
}
```

**Developer Friction**: HIGH (missing critical example)

### 2. Testing with CaptureAdapter

**README shows**:
```rust
use observability_narration_core::CaptureAdapter;
use serial_test::serial;

#[test]
#[serial(capture_adapter)]
fn test_narration() {
    let adapter = CaptureAdapter::install();
    // ... test code
}
```

**Issues**:
- ✅ Clear example
- ✅ Serial execution documented
- ❌ No guidance on when to use `#[serial]`
- ❌ No example of testing correlation ID propagation
- ❌ No example of asserting specific fields

**Developer Friction**: LOW (mostly clear)

### 3. Secret Redaction

**README shows**:
```rust
narrate!(
    actor = Actor::Orchestratord,
    action = Action::Authenticate,
    authorization = format!("Bearer {}", token), // Will be redacted
    human = "Authenticated API request"
);
```

**Issues**:
- ❌ Uses macro that's not documented
- ❌ Unclear which fields are auto-redacted
- ❌ No example of custom redaction
- ❌ No guidance on when to manually redact

**Developer Friction**: MEDIUM (unclear behavior)

---

## Documentation Gaps

### Critical Missing Sections

**1. Integration Guide for Axum** (MISSING)
- How to extract correlation ID in middleware
- How to pass correlation ID to handlers
- How to inject correlation ID into responses
- Complete working example

**2. Narration Policy** (MISSING)
- What should be narrated? (all requests? errors only?)
- What level for each event type?
- How to avoid log spam?
- Performance impact guidance

**3. Field Reference** (MISSING)
- Complete list of `NarrationFields` fields
- Which fields are required vs. optional
- Which fields are auto-redacted
- Field naming conventions

**4. Migration Guide** (MISSING)
- How to migrate from `tracing::info!` to `narrate`
- How to migrate from custom logging to narration-core
- Breaking changes between versions

**5. Troubleshooting** (MISSING)
- "Why aren't my events captured in tests?"
- "Why is my correlation ID not propagating?"
- "How do I debug redaction issues?"

---

## API Consistency Issues

### Inconsistent Naming

```rust
// Function names
narrate()           // Basic
narrate_auto()      // Auto-injection
narrate_with_correlation!()  // Macro (undocumented)

// Module names
observability_narration_core::narrate
observability_narration_core::auto::narrate_auto
narration_core::narrate_with_correlation  // Different prefix!
```

**Issue**: Three different ways to do the same thing, no clear guidance on when to use which.

### Inconsistent Types

```rust
// String literals
actor: "queen-rbee"

// Enums (in macro example)
actor = Actor::Orchestratord

// Which is preferred? README doesn't say.
```

**Issue**: Type safety unclear. Are enums available? Should we use them?

---

## Recommendations

### High Priority (P0)

**1. Add Axum Integration Example**
```rust
// Add to README under "HTTP Context Propagation"
### Complete Axum Middleware Example

\`\`\`rust
use axum::{extract::Request, middleware::Next, response::Response};
use observability_narration_core::{generate_correlation_id, validate_correlation_id};

pub async fn correlation_middleware(mut req: Request, next: Next) -> Response {
    // Extract or generate correlation ID
    let correlation_id = req.headers()
        .get("X-Correlation-ID")
        .and_then(|v| v.to_str().ok())
        .and_then(|id| validate_correlation_id(id).ok().map(|_| id.to_string()))
        .unwrap_or_else(|| generate_correlation_id());
    
    // Store in request extensions
    req.extensions_mut().insert(correlation_id.clone());
    
    // Continue request
    let mut response = next.run(req).await;
    
    // Add to response headers
    response.headers_mut().insert(
        "X-Correlation-ID",
        correlation_id.parse().unwrap()
    );
    
    response
}

// In handler
async fn my_handler(
    Extension(correlation_id): Extension<String>,
) -> Result<Json<Response>, StatusCode> {
    narrate_auto(NarrationFields {
        actor: "worker-orcd",
        action: "handle_request",
        target: "execute".to_string(),
        human: "Processing request".to_string(),
        correlation_id: Some(correlation_id),
        ..Default::default()
    });
    // ...
}
\`\`\`
```

**2. Add Narration Policy Guide**
```markdown
## When to Narrate

### Always Narrate (INFO level)
- Request received (with correlation ID)
- Request completed (with duration)
- State transitions (job enqueued, worker provisioned)
- External API calls (with correlation ID)

### Narrate on Error (ERROR level)
- Validation failures
- External service errors
- Unexpected conditions

### Don't Narrate
- Internal function calls
- Loop iterations
- Temporary variables
- Debug-only information (use TRACE instead)

### Performance Impact
- Each narration: ~1-5μs overhead
- Redaction: ~180ms for 200-char strings (optimization pending)
- Recommendation: Narrate <100 events per request
```

**3. Consolidate API Patterns**

**Recommendation**: Pick ONE primary pattern and deprecate others.

**Preferred Pattern** (if macro exists):
```rust
narrate!(
    correlation_id = req_id,
    actor = "worker-orcd",
    action = "execute",
    target = job_id,
    human = "Executing job {job_id}"
);
```

**If macro doesn't exist**: Create it or document why `narrate_auto` is preferred.

### Medium Priority (P1)

**4. Add Field Reference Table**
```markdown
## NarrationFields Reference

| Field | Type | Required | Auto-Redacted | Description |
|-------|------|----------|---------------|-------------|
| `actor` | `&str` | ✅ Yes | No | Service name (e.g., "worker-orcd") |
| `action` | `&str` | ✅ Yes | No | Action performed (e.g., "execute") |
| `target` | `String` | ✅ Yes | No | Target of action (e.g., job ID) |
| `human` | `String` | ✅ Yes | No | Human-readable description |
| `correlation_id` | `Option<String>` | No | No | Request tracking ID |
| `authorization` | `Option<String>` | No | ✅ Yes | Auth header (auto-redacted) |
| `api_key` | `Option<String>` | No | ✅ Yes | API key (auto-redacted) |
| `token` | `Option<String>` | No | ✅ Yes | Token (auto-redacted) |
| `password` | `Option<String>` | No | ✅ Yes | Password (auto-redacted) |
| `secret` | `Option<String>` | No | ✅ Yes | Secret (auto-redacted) |
| `cute` | `Option<String>` | No | No | Whimsical narration |
```

**5. Add Troubleshooting Section**
```markdown
## Troubleshooting

### Events Not Captured in Tests
**Problem**: `adapter.captured()` returns empty vec.
**Solution**: Add `#[serial(capture_adapter)]` to test.

### Correlation ID Not Propagating
**Problem**: Downstream services don't see correlation ID.
**Solution**: Ensure `inject_context_into_headers` called before HTTP request.

### Redaction Not Working
**Problem**: Secrets appearing in logs.
**Solution**: Use exact field names: `authorization`, `api_key`, `token`, `password`, `secret`.
```

### Low Priority (P2)

**6. Add Migration Guide**
**7. Add Performance Benchmarks to README**
**8. Add Examples Directory** (`examples/axum-integration/`, `examples/testing/`)

---

## Comparison to Best Practices

### ✅ What narration-core Does Well

**1. Testing Support**
- `CaptureAdapter` is excellent
- Rich assertion helpers
- Serial execution documented

**2. Security**
- Auto-redaction of secrets
- ReDoS-safe regex patterns
- Unicode safety built-in

**3. Specification**
- 42 normative requirements
- Stable IDs (NARR-xxxx)
- 100% test coverage

### ❌ What Could Be Better

**1. API Ergonomics**
- Too many ways to narrate (3 patterns)
- Verbose struct initialization
- No builder pattern

**2. Documentation**
- Missing integration guides
- No policy guidance
- Inconsistent examples

**3. Type Safety**
- String literals for actor/action
- No compile-time validation
- Enums mentioned but not documented

---

## Developer Experience Score

| Category | Score | Notes |
|----------|-------|-------|
| **API Clarity** | 6/10 | Multiple patterns, unclear which to use |
| **Documentation** | 7/10 | Comprehensive but missing key examples |
| **Integration** | 5/10 | Missing Axum middleware example |
| **Testing** | 9/10 | Excellent `CaptureAdapter` support |
| **Type Safety** | 4/10 | String literals, no enums |
| **Error Messages** | 7/10 | Good validation, could be clearer |
| **Performance** | 6/10 | Documented but redaction slow |

**Overall**: 6.3/10 (B+)

---

## Action Items for Developer Experience Team

### Immediate (This Week)

- [ ] Add Axum middleware example to README
- [ ] Add "When to Narrate" policy guide
- [ ] Clarify which narration pattern is preferred
- [ ] Add field reference table

### Short Term (Next Sprint)

- [ ] Create `examples/` directory with working code
- [ ] Add troubleshooting section
- [ ] Document or remove macro pattern
- [ ] Add migration guide from `tracing`

### Long Term (v0.2.0)

- [ ] Consolidate to single narration API
- [ ] Add builder pattern for ergonomics
- [ ] Add enums for actor/action (type safety)
- [ ] Optimize redaction performance

---

## Conclusion

**narration-core is production-ready but not developer-friendly.**

The crate works well and has excellent test coverage, but the API is verbose and inconsistent. Developers will struggle with:
1. Choosing which narration pattern to use
2. Integrating with Axum middleware
3. Understanding when to narrate vs. not narrate

**Recommendation**: Address P0 issues (Axum example, policy guide, API consolidation) before wider adoption.

**Grade**: B+ (Good, with friction points)

---
Crafted with love by Developer Experience Team 🎨
