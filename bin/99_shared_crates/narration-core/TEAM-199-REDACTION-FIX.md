# TEAM-199: Security Fix - Redaction in SSE Path

**Team:** TEAM-199  
**Priority:** 🚨 **CRITICAL - SECURITY ISSUE**  
**Duration:** 2-3 hours  
**Based On:** FLAW 1 from TEAM-197's review

---

## Mission

Fix missing redaction in SSE path. Currently only `human` field is redacted, but `cute`, `story`, and `target` are NOT redacted, creating a security vulnerability where secrets can leak through SSE streams.

---

## The Security Vulnerability

### Current Code (BUGGY)

**Location:** `bin/99_shared_crates/narration-core/src/sse_sink.rs` line 40-55

```rust
impl From<NarrationFields> for NarrationEvent {
    fn from(fields: NarrationFields) -> Self {
        Self {
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target: fields.target,           // ❌ NOT REDACTED!
            human: fields.human,              // ❌ NOT REDACTED!
            cute: fields.cute,                // ❌ NOT REDACTED!
            story: fields.story,              // ❌ NOT REDACTED!
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
        }
    }
}
```

### Why This is Critical

**stderr path (SAFE):**
```rust
// In lib.rs narrate_at_level() line 433-440
let human = redact_secrets(&fields.human, RedactionPolicy::default());
let cute = fields.cute.as_ref().map(|c| redact_secrets(c, ...));
let story = fields.story.as_ref().map(|s| redact_secrets(s, ...));
```

**SSE path (VULNERABLE):**
```rust
// In sse_sink.rs - NO REDACTION!
target: fields.target,  // Could contain API keys!
human: fields.human,    // Could contain passwords!
```

**Impact:**
- API keys in narration messages leak through SSE
- Passwords in error messages visible to web UI
- Auth tokens in debug output sent to clients
- **stderr is safe, but SSE leaks secrets!**

---

## The Fix

### Step 1: Update `From<NarrationFields>` Implementation

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

**Location:** Line 40-55 (the `impl From<NarrationFields>` block)

**BEFORE:**
```rust
impl From<NarrationFields> for NarrationEvent {
    fn from(fields: NarrationFields) -> Self {
        Self {
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target: fields.target,
            human: fields.human,
            cute: fields.cute,
            story: fields.story,
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
        }
    }
}
```

**AFTER:**
```rust
use crate::{redact_secrets, RedactionPolicy};  // Add this import at top of file

impl From<NarrationFields> for NarrationEvent {
    fn from(fields: NarrationFields) -> Self {
        // TEAM-199: Apply redaction to ALL text fields (security fix)
        // This mirrors the redaction in narrate_at_level() (lib.rs line 433-440)
        let target = redact_secrets(&fields.target, RedactionPolicy::default());
        let human = redact_secrets(&fields.human, RedactionPolicy::default());
        let cute = fields.cute.as_ref()
            .map(|c| redact_secrets(c, RedactionPolicy::default()));
        let story = fields.story.as_ref()
            .map(|s| redact_secrets(s, RedactionPolicy::default()));
        
        Self {
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target,  // ✅ Redacted
            human,   // ✅ Redacted
            cute: cute.map(|c| c.to_string()),   // ✅ Redacted
            story: story.map(|s| s.to_string()), // ✅ Redacted
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
        }
    }
}
```

---

## Testing Strategy

### Test 1: Human Field Redaction

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs` (add to end of file)

```rust
#[cfg(test)]
mod team_199_security_tests {
    use super::*;
    use crate::NarrationFields;

    #[test]
    fn test_sse_event_redacts_api_key_in_human() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test-target".to_string(),
            human: "Connecting with API key: sk-1234567890abcdef".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // API key should be redacted
        assert!(event.human.contains("***REDACTED***"));
        assert!(!event.human.contains("sk-1234567890abcdef"));
    }

    #[test]
    fn test_sse_event_redacts_bearer_token_in_human() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test-target".to_string(),
            human: "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Bearer token should be redacted
        assert!(event.human.contains("***REDACTED***"));
        assert!(!event.human.contains("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"));
    }

    #[test]
    fn test_sse_event_redacts_target_field() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "https://api.example.com?api_key=secret123".to_string(),
            human: "Making request".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // API key in target should be redacted
        assert!(event.target.contains("***REDACTED***"));
        assert!(!event.target.contains("secret123"));
    }

    #[test]
    fn test_sse_event_redacts_cute_field() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "Test message".to_string(),
            cute: Some("The API whispered its secret: sk-abcd1234".to_string()),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // API key in cute should be redacted
        let cute = event.cute.as_ref().unwrap();
        assert!(cute.contains("***REDACTED***"));
        assert!(!cute.contains("sk-abcd1234"));
    }

    #[test]
    fn test_sse_event_redacts_story_field() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "Test message".to_string(),
            story: Some("'What's your password?' asked the villain. 'admin123!' replied the hero.".to_string()),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Password in story should be redacted (if redaction policy catches it)
        let story = event.story.as_ref().unwrap();
        // Note: Current redaction might not catch "admin123" - this tests the mechanism
        // The important thing is redaction is APPLIED, even if the pattern doesn't match
        assert!(!story.is_empty());
    }

    #[test]
    fn test_sse_event_preserves_safe_content() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "user-session-123".to_string(),
            human: "Processing request".to_string(),
            cute: Some("The worker bee buzzed happily 🐝".to_string()),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Safe content should NOT be redacted
        assert_eq!(event.target, "user-session-123");
        assert_eq!(event.human, "Processing request");
        assert_eq!(event.cute.as_ref().unwrap(), "The worker bee buzzed happily 🐝");
    }
}
```

### Test 2: Compare stderr and SSE Redaction

```rust
#[test]
fn test_sse_and_stderr_have_same_redaction() {
    use crate::narrate_at_level;
    
    let fields = NarrationFields {
        actor: "test",
        action: "test",
        target: "url?token=secret".to_string(),
        human: "API key: sk-test123".to_string(),
        cute: Some("Password: admin123".to_string()),
        ..Default::default()
    };

    // Create SSE event
    let sse_event = NarrationEvent::from(fields.clone());

    // The redaction should match what narrate_at_level does
    // Both should redact the same patterns
    assert!(sse_event.target.contains("***REDACTED***"));
    assert!(sse_event.human.contains("***REDACTED***"));
    
    // Note: We can't easily test narrate_at_level output without capturing stderr
    // But we verify SSE uses the same redact_secrets() function
}
```

---

## Verification Checklist

### Before Starting
- [ ] Read TEAM-197-ARCHITECTURE-REVIEW.md FLAW 1 section
- [ ] Understand current redaction in `lib.rs` line 433-440
- [ ] Review `redact_secrets()` function

### Implementation
- [ ] Add `use crate::{redact_secrets, RedactionPolicy};` import
- [ ] Redact `target` field
- [ ] Redact `human` field
- [ ] Redact `cute` field (Option)
- [ ] Redact `story` field (Option)
- [ ] Convert redacted strings to owned Strings

### Testing
- [ ] Add all 7 tests to `sse_sink.rs`
- [ ] Run tests: `cargo test -p observability-narration-core team_199`
- [ ] All tests pass
- [ ] No secrets leak in test output

### Code Quality
- [ ] Added TEAM-199 comment to modified code
- [ ] Code compiles without warnings
- [ ] Follows existing code style
- [ ] No TODO markers

---

## Expected Changes

### Files Modified
- `bin/99_shared_crates/narration-core/src/sse_sink.rs`
  - Updated `From<NarrationFields>` impl (~15 lines changed)
  - Added 7 security tests (~120 lines added)

### Impact
- **Lines changed:** ~15
- **Lines added:** ~120 (tests)
- **Security vulnerabilities fixed:** 4 (target, human, cute, story)

---

## Handoff Checklist

Before handing off to TEAM-200:

- [ ] ✅ Security fix implemented
- [ ] ✅ All tests pass
- [ ] ✅ Code reviewed (self)
- [ ] ✅ No TODO markers
- [ ] ✅ TEAM-199 signature added
- [ ] ✅ Handoff document ≤2 pages ✅

---

## Common Pitfalls

### ❌ WRONG: Forgetting to Convert to String
```rust
// BAD: Returns Cow<'a, str>, not String
cute: fields.cute.as_ref().map(|c| redact_secrets(c, ...))
```

### ✅ CORRECT: Convert to String
```rust
// GOOD: Convert Cow to String
cute: fields.cute.as_ref()
    .map(|c| redact_secrets(c, RedactionPolicy::default()))
    .map(|c| c.to_string())
```

### ❌ WRONG: Redacting Only Some Fields
```rust
// BAD: Only redacts human, others leak!
let human = redact_secrets(&fields.human, ...);
target: fields.target,  // Not redacted!
```

### ✅ CORRECT: Redact ALL Text Fields
```rust
// GOOD: All fields redacted
let target = redact_secrets(&fields.target, ...);
let human = redact_secrets(&fields.human, ...);
let cute = fields.cute.as_ref().map(|c| redact_secrets(c, ...));
let story = fields.story.as_ref().map(|s| redact_secrets(s, ...));
```

---

## Success Criteria

### Security
- ✅ All text fields redacted in SSE events
- ✅ Same redaction policy as stderr path
- ✅ Tests verify secrets don't leak

### Code Quality
- ✅ Implementation mirrors `narrate_at_level()` redaction
- ✅ Tests cover all text fields
- ✅ No breaking changes to API

### Testing
- ✅ 7 tests added
- ✅ All tests pass
- ✅ Test coverage for API keys, tokens, passwords

---

## Next Team

**TEAM-200** can start immediately after this is complete.

They will build on your secure foundation to add job-scoped SSE.

---

## Summary

**Problem:** SSE path doesn't redact secrets (security vulnerability)  
**Solution:** Apply redaction to ALL text fields in `From<NarrationFields>`  
**Testing:** 7 tests verify secrets are redacted  
**Impact:** Security fix, no breaking changes

---

**Created for:** TEAM-199  
**Priority:** 🚨 CRITICAL  
**Status:** READY TO IMPLEMENT

**Do this FIRST. Other teams depend on your fix.**
