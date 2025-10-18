# TEAM-089 Bug Report - Bugs That Slipped Through BDD Testing

**Date:** 2025-10-11  
**Team:** TEAM-089  
**Critical Finding:** 3 production bugs were not caught by BDD tests

---

## Bug #1: Missing `llama.vocab_size` in GGUF Metadata

### Symptom
```
Error: Missing llama.vocab_size in GGUF metadata
Worker crashes immediately on startup
```

### Root Cause
- GGUF files from some sources (e.g., TheBloke quantized models) don't include explicit `llama.vocab_size` field
- Worker code assumed this field always exists: `content.metadata.get("llama.vocab_size").unwrap()`
- Panic on `.unwrap()` when field is missing

### Fix Location
**File:** `bin/llm-worker-rbee/src/backend/models/quantized_llama.rs` (lines 104-148)

**Before:**
```rust
let vocab_size = content
    .metadata
    .get("llama.vocab_size")
    .and_then(|v| v.to_u32().ok())
    .context("Missing llama.vocab_size in GGUF metadata")? as usize;
```

**After:**
```rust
let vocab_size = content
    .metadata
    .get("llama.vocab_size")
    .and_then(|v| v.to_u32().ok())
    .or_else(|| {
        // TEAM-089: Fallback - derive from tokenizer.ggml.tokens array
        let derived_size = content.metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                candle_core::quantized::gguf_file::Value::Array(arr) => Some(arr.len() as u32),
                _ => None,
            });
        
        if let Some(size) = derived_size {
            narrate(NarrationFields {
                actor: "model-loader",
                action: "gguf_vocab_size_derived",
                target: path.display().to_string(),
                human: format!("Derived vocab_size={} from tokenizer.ggml.tokens array", size),
                cute: Some(format!("Found vocab_size by counting {} tokens! 🔢✨", size)),
                ..Default::default()
            });
        }
        
        derived_size
    })
    .with_context(|| "Cannot determine vocab_size from GGUF metadata")? as usize;
```

### Why BDD Missed This
**Missing Test Case:**
```gherkin
Scenario: GGUF file missing llama.vocab_size metadata
  Given a GGUF file without "llama.vocab_size" field
  And the GGUF file has "tokenizer.ggml.tokens" array with 32000 items
  When the worker loads the model
  Then the worker derives vocab_size from tokenizer array
  And narration event "gguf_vocab_size_derived" is emitted
  And the model loads successfully
```

**Root Cause of BDD Gap:**
- BDD tests used synthetic/complete GGUF files with all metadata fields
- No test for "real-world incomplete GGUF files"
- No test for metadata field fallback logic

---

## Bug #2: Missing Answer Narration (Policy Breach)

### Symptom
```
INFO Inference completed tokens_generated=100 duration_ms=8846 tokens_per_sec=11
INFO Inference complete job_id=test-123 tokens=86

# ❌ NO ACTUAL GENERATED TEXT IN LOGS
# User asked "Why is the sky blue?" but answer is invisible
```

### Root Cause
- Inference completes successfully
- Narration shows token count, timing, throughput
- **But never shows the actual generated text**
- Violates Narration Core Team policy: "Show what the system did, not just that it did something"

### Fix Location
**File:** `bin/llm-worker-rbee/src/backend/inference.rs` (lines 332-368)

**Before:**
```rust
narrate(NarrationFields {
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_INFERENCE_COMPLETE,
    target: format!("{}-tokens", generated_tokens.len()),
    human: format!(
        "Inference completed ({} tokens in {} ms, {} tok/s)",
        generated_tokens.len(),
        duration_ms,
        tokens_per_sec
    ),
    cute: Some(format!(
        "Generated {} tokens in {} ms! {} tok/s! 🎉",
        generated_tokens.len(),
        duration_ms,
        tokens_per_sec
    )),
    // ❌ NO ACTUAL TEXT
});
```

**After:**
```rust
// TEAM-089: Join generated text for logging
let full_text = generated_text.join("");
let text_preview = if full_text.len() > 100 {
    format!("{}...", &full_text[..100])
} else {
    full_text.clone()
};

tracing::info!(
    tokens_generated = generated_tokens.len(),
    duration_ms = duration_ms,
    tokens_per_sec = tokens_per_sec,
    text_preview = %text_preview,  // ✅ NOW VISIBLE
    "Inference completed"
);

// TEAM-089: Narrate the actual answer (CRITICAL for debugging)
narrate(NarrationFields {
    actor: ACTOR_CANDLE_BACKEND,
    action: ACTION_INFERENCE_COMPLETE,
    target: format!("{}-tokens", generated_tokens.len()),
    human: format!(
        "Generated: \"{}\" ({} tokens, {} ms, {} tok/s)",
        text_preview,  // ✅ SHOWS ACTUAL TEXT
        generated_tokens.len(),
        duration_ms,
        tokens_per_sec
    ),
    cute: Some(format!(
        "Answer: \"{}\" 🎉 ({} tok/s)",
        text_preview,  // ✅ SHOWS ACTUAL TEXT
        tokens_per_sec
    )),
});
```

### Why BDD Missed This
**Missing Test Case:**
```gherkin
Scenario: Inference narration includes generated text
  Given a worker with loaded model
  When inference completes with prompt "Why is the sky blue?"
  And generates text "The sky is blue because..."
  Then narration event "inference_complete" is emitted
  And the narration "human" field contains the generated text preview
  And the narration "cute" field contains the generated text preview
  And the generated text is visible in structured logs
```

**Root Cause of BDD Gap:**
- BDD tests checked for narration event existence
- BDD tests validated event structure (actor, action, target)
- **BDD tests did NOT validate narration content quality**
- No assertion on "does the narration actually show what happened?"

**Policy Violation:**
- Narration Core Team standard: "Logs must show WHAT happened, not just THAT it happened"
- This is a **policy breach** that should have been caught by compliance tests

---

## Bug #3: SSE Stream Hangs Forever (Cascading Shutdown Failure)

### Symptom
```bash
$ curl -N http://localhost:18888/v1/inference -d '{"job_id":"test","prompt":"Hello","max_tokens":10,...}'

data: {"type":"narration",...}
data: {"type":"started",...}
# ... hangs forever, never sends [DONE] ...
^C  # User must Ctrl+C

# Worker logs show:
INFO Inference complete job_id=test tokens=10
# But client never receives completion signal
```

### Root Cause
- Inference completes successfully
- SSE stream created with narration channel receiver
- Narration sender stored in thread-local storage
- **Sender never dropped/cleared after inference completes**
- Receiver waits forever for more messages
- Stream never closes, `[DONE]` marker never sent

### Fix Location
**File:** `bin/llm-worker-rbee/src/http/execute.rs` (lines 120-122)

**Before:**
```rust
info!(job_id = %req.job_id, tokens = result.tokens.len(), "Inference complete");

// Convert result to SSE events
let mut events = vec![...];

// TEAM-039: Merge narration events with token events
let narration_stream = UnboundedReceiverStream::new(narration_rx);
// ❌ Sender still alive in thread-local storage
// ❌ Receiver waits forever for more messages
```

**After:**
```rust
info!(job_id = %req.job_id, tokens = result.tokens.len(), "Inference complete");

// TEAM-089: Clear narration sender to close the channel
// This allows the receiver stream to complete instead of waiting forever
narration_channel::clear_sender();  // ✅ DROP SENDER

// Convert result to SSE events
let mut events = vec![...];

// TEAM-039: Merge narration events with token events
let narration_stream = UnboundedReceiverStream::new(narration_rx);
// ✅ Receiver sees closed channel, completes immediately
```

### Why BDD Missed This
**Missing Test Case:**
```gherkin
Scenario: SSE stream completes with [DONE] marker
  Given a worker with loaded model
  When a client sends inference request via SSE
  And inference completes successfully
  Then the SSE stream sends all token events
  And the SSE stream sends "End" event
  And the SSE stream sends "[DONE]" marker
  And the SSE stream closes cleanly
  And the HTTP connection terminates
  And the client receives exit code 0 (not timeout)
```

**Root Cause of BDD Gap:**
- BDD tests likely used synchronous HTTP client
- No test for "does the stream actually close?"
- No test for "does curl exit cleanly without timeout?"
- No test for cascading shutdown behavior

**Architectural Issue:**
- Thread-local storage cleanup not enforced
- No RAII guard to ensure sender is dropped
- Manual `clear_sender()` call required (error-prone)

---

## Summary: BDD Testing Gaps

### Gap #1: Real-World Data Scenarios
**Problem:** BDD tests used synthetic/complete data  
**Missing:** Tests with incomplete/malformed real-world data  
**Example:** GGUF files missing optional metadata fields

### Gap #2: Narration Content Quality
**Problem:** BDD tests validated structure, not content  
**Missing:** Assertions on "does narration show what actually happened?"  
**Example:** Checking if generated text appears in narration logs

### Gap #3: End-to-End Client Behavior
**Problem:** BDD tests used internal APIs, not real clients  
**Missing:** Tests with actual HTTP clients (curl, fetch, etc.)  
**Example:** "Does curl exit cleanly or timeout?"

### Gap #4: Cascading Shutdown Scenarios
**Problem:** BDD tests checked happy path, not cleanup  
**Missing:** Tests for resource cleanup and graceful shutdown  
**Example:** "Does SSE stream close after inference completes?"

---

## Recommended BDD Test Additions

### 1. GGUF Metadata Fallback Tests
```gherkin
Feature: GGUF Model Loading with Incomplete Metadata

  Scenario: Missing llama.vocab_size - derive from tokenizer
    Given a GGUF file without "llama.vocab_size" metadata
    And the GGUF file has "tokenizer.ggml.tokens" array with 32000 items
    When the worker loads the GGUF model
    Then narration event "gguf_vocab_size_derived" is emitted
    And the derived vocab_size is 32000
    And the model loads successfully

  Scenario: Missing both vocab_size and tokenizer array
    Given a GGUF file without "llama.vocab_size" metadata
    And the GGUF file without "tokenizer.ggml.tokens" array
    When the worker attempts to load the GGUF model
    Then narration event "gguf_metadata_missing" is emitted
    And the error_kind is "missing_metadata_field"
    And the worker fails with helpful error message
```

### 2. Narration Content Quality Tests
```gherkin
Feature: Inference Narration Content Quality

  Scenario: Inference completion shows generated text
    Given a worker with loaded model
    When inference completes with prompt "Why is the sky blue?"
    And generates text "The sky is blue because of Rayleigh scattering..."
    Then narration event "inference_complete" is emitted
    And the narration "human" field contains "The sky is blue because"
    And the narration "cute" field contains "Answer:"
    And the generated text preview is visible in logs

  Scenario: Narration text preview truncates long outputs
    Given a worker with loaded model
    When inference generates 500 characters of text
    Then the narration text preview is truncated to 100 characters
    And the preview ends with "..."
    And the full text is available in the response
```

### 3. SSE Stream Lifecycle Tests
```gherkin
Feature: SSE Stream Lifecycle and Cleanup

  Scenario: SSE stream closes cleanly after inference
    Given a worker with loaded model
    When a client connects via SSE for inference
    And inference completes successfully
    Then the SSE stream sends all token events
    And the SSE stream sends "End" event
    And the SSE stream sends "[DONE]" marker
    And the SSE connection closes within 1 second
    And the HTTP client exits with code 0

  Scenario: SSE stream closes on inference error
    Given a worker with loaded model
    When a client connects via SSE for inference
    And inference fails with error "Out of memory"
    Then the SSE stream sends error event
    And the SSE stream sends "[DONE]" marker
    And the SSE connection closes within 1 second
    And the HTTP client exits with code 0

  Scenario: Narration channel cleanup after request
    Given a worker with loaded model
    When inference request completes
    Then the narration sender is cleared from thread-local storage
    And subsequent narration calls return false
    And no memory leaks occur
```

### 4. End-to-End Client Tests
```gherkin
Feature: Real HTTP Client Behavior

  Scenario: curl client receives complete response
    Given a worker listening on port 18888
    When curl sends POST to /v1/inference with timeout 10s
    And inference generates 20 tokens
    Then curl receives all 20 token events
    And curl receives [DONE] marker
    And curl exits with code 0 (not 124 timeout)
    And curl completes in less than 10 seconds

  Scenario: Streaming client processes tokens in real-time
    Given a worker listening on port 18888
    When a streaming HTTP client connects
    And inference generates 100 tokens over 10 seconds
    Then the client receives tokens incrementally
    And the client does not wait for all tokens before processing
    And the stream closes cleanly after [DONE]
```

---

## Root Cause Analysis: Why These Bugs Existed

### 1. Incomplete GGUF Metadata Handling
**Why it happened:**
- Code written against "ideal" GGUF files with all fields
- No testing with real-world quantized models from TheBloke, etc.
- Assumption that HuggingFace models = complete metadata

**Prevention:**
- Test with actual downloaded models from multiple sources
- Add property-based tests for "any GGUF file should load or fail gracefully"
- Document required vs. optional GGUF metadata fields

### 2. Narration Content Quality Not Enforced
**Why it happened:**
- Narration system focused on structure (actor, action, target)
- No review of "does this narration actually help debugging?"
- Policy ("show what happened") not encoded in tests

**Prevention:**
- Add narration content assertions to BDD tests
- Code review checklist: "Does narration show the actual data?"
- Automated lint: "Narration must include relevant data fields"

### 3. SSE Stream Lifecycle Not Tested
**Why it happened:**
- BDD tests used internal APIs, not real HTTP clients
- No test for "does the connection actually close?"
- Thread-local storage cleanup not enforced by type system

**Prevention:**
- Add end-to-end tests with real HTTP clients (curl, fetch)
- Use RAII guard for thread-local cleanup (not manual clear)
- Test cascading shutdown scenarios explicitly

---

## Impact Assessment

### Bug #1: Missing vocab_size
- **Severity:** P0 - Worker crashes on startup
- **Affected:** All GGUF models without explicit vocab_size field
- **User Impact:** Complete service outage
- **Detection:** Immediate (worker won't start)

### Bug #2: Missing answer narration
- **Severity:** P1 - Policy breach, debugging impossible
- **Affected:** All inference requests
- **User Impact:** Cannot debug incorrect outputs
- **Detection:** Only when debugging production issues

### Bug #3: SSE stream hangs
- **Severity:** P0 - Client hangs forever
- **Affected:** All SSE inference requests
- **User Impact:** Clients timeout, poor UX
- **Detection:** Immediate (client never receives response)

---

## Lessons Learned

### 1. Test with Real-World Data
- Synthetic test data hides edge cases
- Download actual models from HuggingFace/TheBloke
- Test with incomplete/malformed data

### 2. Test Narration Content, Not Just Structure
- Validate that narration shows actual data
- Assert on specific text appearing in logs
- Review narration quality in code reviews

### 3. Test End-to-End with Real Clients
- Use curl, fetch, real HTTP clients in tests
- Check exit codes, not just response bodies
- Test connection lifecycle, not just happy path

### 4. Test Cascading Shutdown
- Verify resources are cleaned up
- Check that streams close cleanly
- Test timeout scenarios explicitly

---

**Created by:** TEAM-089  
**Date:** 2025-10-11  
**Purpose:** Document bugs that slipped through BDD testing for process improvement

---

*These bugs were fixed, but they should never have reached production. The BDD test suite needs the additions outlined above.*
