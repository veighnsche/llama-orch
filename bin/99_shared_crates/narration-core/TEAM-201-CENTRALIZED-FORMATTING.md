# TEAM-201: Centralized Formatting

**Team:** TEAM-201  
**Priority:** HIGH  
**Duration:** 3-4 hours  
**Based On:** TEAM-198 Phase 1 (corrected by TEAM-197)

---

## Mission

Add `formatted: String` field to `NarrationEvent` and pre-format narration text in narration-core. This eliminates manual formatting in consumers and ensures consistency everywhere.

---

## The Problem (TEAM-197 Fixed in SSE_FORMATTING_ISSUE.md)

### Current State

**narration-core formats stderr:**
```rust
// lib.rs line 449
eprintln!("[{:<10}] {:<15}: {}", actor, action, human);
```

**But SSE sends raw struct:**
```rust
// sse_sink.rs
NarrationEvent {
    actor: String,
    action: String,
    human: String,
    // No formatted field!
}
```

**Result:** Consumers must format manually:
```rust
// queen jobs.rs line 109 (TEAM-197 fixed this)
let formatted = format!("[{:<10}] {:<15}: {}", event.actor, event.action, event.human);
```

**Impact:**
- ❌ Manual duplication of formatting logic
- ❌ Format changes break all consumers
- ❌ Inconsistent formats possible

---

## The Solution: Add `formatted` Field

### Step 1: Add Field to NarrationEvent

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

**Location:** Line 19-38 (the `NarrationEvent` struct)

**BEFORE:**
```rust
/// Narration event formatted for SSE transport.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NarrationEvent {
    pub actor: String,
    pub action: String,
    pub target: String,
    pub human: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cute: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub story: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_at_ms: Option<u64>,
}
```

**AFTER:**
```rust
/// Narration event formatted for SSE transport.
/// 
/// TEAM-201: Added `formatted` field for centralized formatting.
/// Consumers should use `formatted` instead of manually formatting actor/action/human.
#[derive(Debug, Clone, serde::Serialize)]
pub struct NarrationEvent {
    /// Pre-formatted text matching stderr output
    /// Format: "[actor     ] action         : message"
    /// TEAM-201: This is the SINGLE source of truth for SSE display
    pub formatted: String,
    
    // Keep existing fields for backward compatibility and programmatic access
    pub actor: String,
    pub action: String,
    pub target: String,
    pub human: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cute: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub story: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correlation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_by: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_at_ms: Option<u64>,
}
```

---

### Step 2: Update From<NarrationFields> to Pre-Format

**File:** `bin/99_shared_crates/narration-core/src/sse_sink.rs`

**Location:** Line 40-55 (the `impl From<NarrationFields>` block)

**Note:** TEAM-199 already added redaction. Build on their work!

**AFTER TEAM-199's work:**
```rust
impl From<NarrationFields> for NarrationEvent {
    fn from(fields: NarrationFields) -> Self {
        // TEAM-199: Apply redaction to ALL text fields
        let target = redact_secrets(&fields.target, RedactionPolicy::default());
        let human = redact_secrets(&fields.human, RedactionPolicy::default());
        let cute = fields.cute.as_ref()
            .map(|c| redact_secrets(c, RedactionPolicy::default()));
        let story = fields.story.as_ref()
            .map(|s| redact_secrets(s, RedactionPolicy::default()));
        
        Self {
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target,
            human,
            cute: cute.map(|c| c.to_string()),
            story: story.map(|s| s.to_string()),
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
        }
    }
}
```

**NOW ADD (TEAM-201):**
```rust
impl From<NarrationFields> for NarrationEvent {
    fn from(fields: NarrationFields) -> Self {
        // TEAM-199: Apply redaction to ALL text fields
        let target = redact_secrets(&fields.target, RedactionPolicy::default());
        let human = redact_secrets(&fields.human, RedactionPolicy::default());
        let cute = fields.cute.as_ref()
            .map(|c| redact_secrets(c, RedactionPolicy::default()));
        let story = fields.story.as_ref()
            .map(|s| redact_secrets(s, RedactionPolicy::default()));
        
        // TEAM-201: Pre-format text (same format as stderr output)
        // Format: "[actor     ] action         : message"
        // - Actor: 10 chars (left-aligned, padded)
        // - Action: 15 chars (left-aligned, padded)
        // This matches lib.rs line 449 exactly
        let formatted = format!("[{:<10}] {:<15}: {}", fields.actor, fields.action, human);
        
        Self {
            formatted,  // ← NEW FIELD
            actor: fields.actor.to_string(),
            action: fields.action.to_string(),
            target,
            human: human.to_string(),
            cute: cute.map(|c| c.to_string()),
            story: story.map(|s| s.to_string()),
            correlation_id: fields.correlation_id,
            job_id: fields.job_id,
            emitted_by: fields.emitted_by,
            emitted_at_ms: fields.emitted_at_ms,
        }
    }
}
```

---

### Step 3: Update Queen SSE Consumer

**File:** `bin/10_queen_rbee/src/http/jobs.rs`

**Location:** Line 107-110 (inside `handle_stream_job()`)

**BEFORE (TEAM-197 fixed this):**
```rust
Ok(event) => {
    received_first_event = true;
    last_event_time = std::time::Instant::now();
    // TEAM-197: Format narration with fixed-width columns
    let formatted = format!("[{:<10}] {:<15}: {}", event.actor, event.action, event.human);
    yield Ok(Event::default().data(formatted));
}
```

**AFTER (TEAM-201):**
```rust
Ok(event) => {
    received_first_event = true;
    last_event_time = std::time::Instant::now();
    // TEAM-201: Use pre-formatted text from narration-core (no manual formatting!)
    yield Ok(Event::default().data(&event.formatted));
}
```

---

## Testing Strategy

### Test 1: Formatted Field Matches stderr

```rust
#[cfg(test)]
mod team_201_formatting_tests {
    use super::*;
    use crate::NarrationFields;

    #[test]
    fn test_formatted_field_matches_stderr_format() {
        let fields = NarrationFields {
            actor: "test-actor",
            action: "test-action",
            target: "test-target".to_string(),
            human: "Test message".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Formatted field should match: "[actor     ] action         : message"
        assert_eq!(event.formatted, "[test-actor] test-action    : Test message");
        
        // Verify padding
        assert!(event.formatted.starts_with("[test-actor]"));
        assert!(event.formatted.contains("test-action    :"));
    }

    #[test]
    fn test_formatted_with_short_actor() {
        let fields = NarrationFields {
            actor: "abc",
            action: "xyz",
            target: "test".to_string(),
            human: "Short".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Should pad to 10 chars for actor, 15 for action
        assert_eq!(event.formatted, "[abc       ] xyz            : Short");
    }

    #[test]
    fn test_formatted_with_long_actor() {
        let fields = NarrationFields {
            actor: "very-long-actor-name",
            action: "very-long-action-name",
            target: "test".to_string(),
            human: "Long".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Should truncate/handle long names (Rust format! will extend if needed)
        assert!(event.formatted.contains("very-long-actor-name"));
        assert!(event.formatted.contains("very-long-action-name"));
        assert!(event.formatted.contains("Long"));
    }

    #[test]
    fn test_formatted_uses_redacted_human() {
        let fields = NarrationFields {
            actor: "test",
            action: "test",
            target: "test".to_string(),
            human: "API key: sk-test123".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Formatted should use redacted human
        assert!(event.formatted.contains("***REDACTED***"));
        assert!(!event.formatted.contains("sk-test123"));
    }

    #[test]
    fn test_backward_compat_raw_fields_still_available() {
        let fields = NarrationFields {
            actor: "test",
            action: "action",
            target: "target".to_string(),
            human: "Message".to_string(),
            ..Default::default()
        };

        let event = NarrationEvent::from(fields);

        // Old fields still work (backward compatibility)
        assert_eq!(event.actor, "test");
        assert_eq!(event.action, "action");
        assert_eq!(event.human, "Message");
        
        // But formatted is also available (new way)
        assert!(!event.formatted.is_empty());
    }
}
```

---

## Verification Checklist

### Before Starting
- [ ] TEAM-199 has completed redaction fix
- [ ] Read TEAM-197 and TEAM-198 proposals
- [ ] Understand current formatting in lib.rs line 449

### Implementation
- [ ] Add `formatted: String` field to `NarrationEvent`
- [ ] Update `From<NarrationFields>` to pre-format
- [ ] Use TEAM-199's redacted `human` in format
- [ ] Update queen `jobs.rs` to use `event.formatted`
- [ ] Remove manual formatting from queen

### Testing
- [ ] Add 5 formatting tests
- [ ] Run: `cargo test -p observability-narration-core team_201`
- [ ] Run: `cargo test -p queen-rbee`
- [ ] All tests pass

### Integration
- [ ] Build: `cargo build -p observability-narration-core`
- [ ] Build: `cargo build -p queen-rbee`
- [ ] Test: `./rbee queen stop && ./rbee hive status`
- [ ] Verify format is correct in keeper output

---

## Expected Changes

### Files Modified
- `bin/99_shared_crates/narration-core/src/sse_sink.rs` (~20 lines changed)
- `bin/10_queen_rbee/src/http/jobs.rs` (~3 lines changed, simpler!)

### Impact
- **Lines added:** ~20 (add field + format)
- **Lines removed:** ~2 (remove manual formatting)
- **Net change:** +18 lines
- **Simplification:** Queen consumer is now trivial

---

## Common Pitfalls

### ❌ WRONG: Formatting Before Redaction
```rust
// BAD: Formats raw text, then redacts
let formatted = format!("[{:<10}] {:<15}: {}", fields.actor, fields.action, fields.human);
let human = redact_secrets(&fields.human, ...);
```

### ✅ CORRECT: Redact First, Then Format
```rust
// GOOD: Redacts, then formats redacted text
let human = redact_secrets(&fields.human, RedactionPolicy::default());
let formatted = format!("[{:<10}] {:<15}: {}", fields.actor, fields.action, human);
```

### ❌ WRONG: Using Raw human in Formatted
```rust
// BAD: Uses non-redacted human!
let formatted = format!("[{:<10}] {:<15}: {}", fields.actor, fields.action, fields.human);
```

### ✅ CORRECT: Use Redacted human
```rust
// GOOD: Uses redacted human variable
let formatted = format!("[{:<10}] {:<15}: {}", fields.actor, fields.action, human);
```

---

## Success Criteria

### Formatting
- ✅ `formatted` field matches stderr output exactly
- ✅ Same padding (actor 10 chars, action 15 chars)
- ✅ Same separator (": ")

### Security
- ✅ Formatted text uses redacted `human` (from TEAM-199)
- ✅ No secrets in formatted field
- ✅ Test verifies redaction in formatted

### Simplicity
- ✅ Queen consumer uses `event.formatted` (1 line)
- ✅ No manual formatting in consumers
- ✅ Backward compatible (old fields still work)

---

## Benefits

### For Developers
- ✅ Just use `event.formatted` (no thinking about format)
- ✅ Format changes propagate automatically
- ✅ No duplication

### For Users
- ✅ Consistent format everywhere
- ✅ stderr and SSE always match
- ✅ No surprises

### For Maintenance
- ✅ One place to change format (narration-core)
- ✅ All consumers update automatically
- ✅ No manual sync needed

---

## Next Team

**TEAM-202** depends on your work (needs formatted field for hive narration).

Make sure your tests pass before handing off!

---

## Summary

**Problem:** Manual formatting in consumers (duplication, inconsistency)  
**Solution:** Add `formatted` field, pre-format in narration-core  
**Testing:** 5 tests verify format consistency and security  
**Impact:** Simpler consumers, automatic format propagation

---

**Created for:** TEAM-201  
**Priority:** HIGH  
**Status:** READY TO IMPLEMENT (after TEAM-199)

**This makes formatting consistent. TEAM-202 will add hive narration.**
