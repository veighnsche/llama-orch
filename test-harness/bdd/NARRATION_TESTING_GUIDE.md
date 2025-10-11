# Narration Testing Guide

**TEAM-085: How to verify product code emits proper narration**

---

## The Rule

**Narration is NOT a separate feature - it's part of EVERY feature!**

Just like error handling, narration verification belongs INSIDE the feature files, not in a separate "observability" feature.

---

## How It Works

### 1. Product Code Emits Narration

The PRODUCT CODE (not tests) must emit narration using `observability-narration-core`:

```rust
use observability_narration_core::{narrate, NarrationFields};

narrate(NarrationFields {
    actor: "rbee-hive",
    action: "spawn",
    target: worker_id.clone(),
    human: format!("Spawning worker for model {}", model_ref),
    correlation_id: Some(correlation_id),
    worker_id: Some(worker_id),
    model_ref: Some(model_ref),
    ..Default::default()
});
```

### 2. BDD Tests Verify Narration

Add narration assertions to EXISTING scenarios:

```gherkin
Scenario: Complete inference workflow
  Given queen-rbee is running
  When client sends inference request
  Then tokens are streamed back
  # TEAM-085: Narration verification
  And I should see narration "Starting inference" from "queen-rbee"
  And I should see narration "Dispatching to worker" from "rbee-hive"
  And all narration events should have correlation IDs
```

---

## Available Assertions

### Basic Narration
```gherkin
And I should see narration "Starting inference" from "queen-rbee"
```
Verifies actor and human text.

### With Specific Fields
```gherkin
And I should see narration "Downloading model" with model_ref "tinyllama-q4"
And I should see narration "Model provisioned" with duration_ms
And I should see narration "Worker registered" with worker_id "worker-001"
```

### Error Scenarios
```gherkin
And narration should include error_kind "worker_crash"
```

### Editorial Standards
```gherkin
And narration human field should be under 100 characters
And all narration events should have correlation IDs
```

---

## What Gets Tested

✅ **Product code emits narration** - Not test code!  
✅ **Human-readable messages** - No cryptic codes  
✅ **Correlation ID propagation** - Track requests  
✅ **Timing information** - duration_ms field  
✅ **Contextual fields** - worker_id, model_ref, etc.  
✅ **Editorial standards** - ≤100 chars, SVO structure  

---

## Example: Adding Narration to a Feature

**Before:**
```gherkin
Scenario: Worker spawn
  Given rbee-hive is running
  When I spawn a worker
  Then worker is ready
```

**After:**
```gherkin
Scenario: Worker spawn
  Given rbee-hive is running
  When I spawn a worker
  Then worker is ready
  # TEAM-085: Narration verification
  And I should see narration "Spawning worker" from "rbee-hive"
  And I should see narration "Worker ready" with worker_id
  And narration should include duration_ms
```

---

## When Tests Fail

### ❌ "PRODUCT CODE DID NOT EMIT NARRATION!"

**Fix:** Add `narrate()` call in the product code:

```rust
use observability_narration_core::{narrate, NarrationFields};

narrate(NarrationFields {
    actor: "your-component",
    action: "your_action",
    target: "your_target".to_string(),
    human: "Your human-readable message".to_string(),
    correlation_id: Some(correlation_id),
    ..Default::default()
});
```

### ❌ "PRODUCT CODE DID NOT INCLUDE worker_id!"

**Fix:** Set the field in NarrationFields:

```rust
narrate(NarrationFields {
    // ... other fields ...
    worker_id: Some(worker_id.clone()),
    ..Default::default()
});
```

### ❌ "NARRATION TOO LONG! (Editorial Standard: ≤100 chars)"

**Fix:** The narration-core team has ultimate editorial authority.  
Rewrite to be concise while keeping it informative.

---

## Files

- **Step definitions:** `src/steps/narration_verification.rs`
- **World state:** `src/steps/world.rs` (narration_enabled, last_narration)
- **Example feature:** `tests/features/900-integration-e2e.feature`

---

## The Narration-Core Team

The `observability-narration-core` crate team has **ultimate editorial authority** over all narration.

Their standards:
- ✅ Human field under 100 characters
- ✅ Present tense, active voice
- ✅ Specific numbers and context
- ✅ Correlation ID propagation
- ✅ Automatic secret redaction

See: `bin/shared-crates/narration-core/TEAM_RESPONSIBILITIES.md`

---

**Created by:** TEAM-085  
**Date:** 2025-10-11  
**Purpose:** Make debugging delightful by forcing product code to narrate!
