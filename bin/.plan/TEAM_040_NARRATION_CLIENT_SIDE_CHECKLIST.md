# TEAM-040: Narration Client-Side Implementation Checklist

**Team:** TEAM-040 (Next Implementation Team)  
**Date:** 2025-10-10  
**Status:** 🔴 NOT STARTED  
**Priority:** P0 - Critical User Experience Feature  
**Depends On:** TEAM-039 (Worker-side narration - ✅ COMPLETE)

---

## 🎯 Mission

Complete the narration plumbing by implementing client-side components (rbee-keeper, queen-rbee, rbee-hive) so users can see real-time progress in their shell.

**Current State:** Worker emits narration to SSE ✅  
**Required State:** User sees narration in rbee-keeper shell ❌

---

## 📋 Complete Implementation Checklist

### Part 1: rbee-keeper (CLI Client) 🔴 CRITICAL

**Location:** `bin/rbee-keeper/src/commands/infer.rs`

#### Task 1.1: Add SSE Event Handling
- [ ] Read current `infer.rs` implementation
- [ ] Identify where SSE events are consumed
- [ ] Add handler for `"narration"` event type
- [ ] Parse narration event fields (actor, action, human, cute, etc.)
- [ ] Verify event deserialization works

**Files to modify:**
- `bin/rbee-keeper/src/commands/infer.rs`

**Code pattern:**
```rust
match event.event_type.as_str() {
    "narration" => {
        // Handle narration event
        if !args.quiet {
            display_narration(&event);
        }
    }
    "token" => {
        // Existing token handling
    }
    // ... other events
}
```

---

#### Task 1.2: Display Narration to stderr
- [ ] Create `display_narration()` function
- [ ] Format narration with actor prefix: `[actor] message`
- [ ] Use `cute` field if available, fallback to `human`
- [ ] Add emoji support (already in cute messages)
- [ ] Write to **stderr** (not stdout - tokens go to stdout)
- [ ] Ensure proper line buffering

**Files to modify:**
- `bin/rbee-keeper/src/commands/infer.rs`

**Code pattern:**
```rust
fn display_narration(event: &NarrationEvent) {
    let message = event.cute.as_ref().unwrap_or(&event.human);
    eprintln!("[{}] {}", event.actor, message);
}
```

**Test command:**
```bash
# Narration goes to stderr, tokens to stdout
rbee-keeper infer --node mac --model tinyllama --prompt "hello" 2>narration.log 1>tokens.txt

# narration.log should contain: [candle-backend] 🚀 Starting inference...
# tokens.txt should contain: Hello world...
```

---

#### Task 1.3: Add --quiet Flag
- [ ] Add `quiet: bool` field to `InferArgs` struct
- [ ] Add `#[arg(long)]` attribute for `--quiet` flag
- [ ] Check `args.quiet` before displaying narration
- [ ] Ensure tokens still display when quiet
- [ ] Update help text

**Files to modify:**
- `bin/rbee-keeper/src/commands/infer.rs`
- `bin/rbee-keeper/src/cli.rs` (if args defined there)

**Code pattern:**
```rust
#[derive(Parser)]
pub struct InferArgs {
    // ... existing args ...
    
    /// Disable narration output (only show tokens)
    #[arg(long)]
    pub quiet: bool,
}
```

**Test commands:**
```bash
# With narration (default)
rbee-keeper infer --node mac --model tinyllama --prompt "hello"
# Should show: [candle-backend] 🚀 Starting inference...

# Without narration (quiet mode)
rbee-keeper infer --node mac --model tinyllama --prompt "hello" --quiet
# Should only show tokens, no narration
```

---

#### Task 1.4: Update Event Type Definitions
- [ ] Check if `NarrationEvent` struct exists
- [ ] Add struct if missing with fields: actor, action, target, human, cute, story, correlation_id, job_id
- [ ] Ensure serde derives for deserialization
- [ ] Add to SSE event enum if needed

**Files to check/modify:**
- `bin/rbee-keeper/src/types.rs` (or wherever events are defined)
- `bin/rbee-keeper/src/sse.rs` (if exists)

**Code pattern:**
```rust
#[derive(Debug, Clone, Deserialize)]
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
}
```

---

#### Task 1.5: Integration Testing
- [ ] Test with real worker (if available)
- [ ] Test with mock SSE stream
- [ ] Verify narration appears in correct order
- [ ] Verify stderr/stdout separation
- [ ] Verify --quiet flag works
- [ ] Test with long-running inference
- [ ] Test with errors (narration should show error messages)

**Test scenarios:**
```bash
# Scenario 1: Normal inference with narration
rbee-keeper infer --node mac --model tinyllama --prompt "hello"

# Scenario 2: Quiet mode
rbee-keeper infer --node mac --model tinyllama --prompt "hello" --quiet

# Scenario 3: Piping tokens to file (narration to stderr)
rbee-keeper infer ... > output.txt 2>narration.log

# Scenario 4: Error case
rbee-keeper infer --node mac --model nonexistent --prompt "hello"
# Should show error narration
```

---

### Part 2: queen-rbee (Orchestrator) 🟡 HIGH PRIORITY

**Location:** `bin/queen-rbee/src/routes/tasks.rs` (or similar)

#### Task 2.1: Identify SSE Relay Logic
- [ ] Find where queen-rbee relays worker SSE events to client
- [ ] Understand current event forwarding mechanism
- [ ] Check if event filtering is applied
- [ ] Verify correlation ID handling

**Files to read:**
- `bin/queen-rbee/src/routes/tasks.rs`
- `bin/queen-rbee/src/routes/inference.rs`
- `bin/queen-rbee/src/sse.rs` (if exists)

---

#### Task 2.2: Add Narration Event Relay
- [ ] Ensure `narration` events are not filtered out
- [ ] Pass through all narration fields unchanged
- [ ] Preserve event ordering (narration + tokens)
- [ ] Maintain correlation ID in relayed events
- [ ] Add logging for narration relay (for debugging)

**Files to modify:**
- `bin/queen-rbee/src/routes/tasks.rs`

**Code pattern:**
```rust
// Relay all events from worker to client
match event.event_type.as_str() {
    "started" | "token" | "metrics" | "narration" | "end" | "error" => {
        // Pass through to client
        client_tx.send(event).await?;
    }
    _ => {
        warn!("Unknown event type: {}", event.event_type);
    }
}
```

---

#### Task 2.3: Update Event Type Definitions
- [ ] Add `Narration` variant to queen-rbee's event enum (if exists)
- [ ] Ensure proper serialization/deserialization
- [ ] Match worker's `InferenceEvent::Narration` structure
- [ ] Add tests for narration event relay

**Files to modify:**
- `bin/queen-rbee/src/types/events.rs` (or similar)

---

#### Task 2.4: Merge Multiple Narration Sources
- [ ] Identify if queen-rbee needs to merge narration from:
  - SSH stdout (rbee-hive startup)
  - SSE from rbee-hive (worker startup via rbee-hive)
  - SSE from worker (inference)
- [ ] Implement stream merging if needed
- [ ] Preserve event ordering across sources
- [ ] Add correlation ID to all narration events

**Files to modify:**
- `bin/queen-rbee/src/routes/tasks.rs`

**Note:** This may be complex. Check TEAM_038_NARRATION_FLOW_CORRECTED.md for architecture.

---

#### Task 2.5: Integration Testing
- [ ] Test narration relay with real worker
- [ ] Verify event ordering preserved
- [ ] Test with multiple concurrent requests
- [ ] Verify correlation IDs maintained
- [ ] Test error scenarios

**Test commands:**
```bash
# Direct queen-rbee test (if possible)
curl -N http://localhost:8080/v2/tasks/job-123/events

# Should see narration events mixed with tokens
```

---

### Part 3: rbee-hive (Pool Manager) 🟡 HIGH PRIORITY

**Location:** `bin/rbee-hive/src/worker_manager.rs`

#### Task 3.1: Capture Worker Stdout
- [ ] Find where rbee-hive spawns worker processes
- [ ] Identify current stdout handling (if any)
- [ ] Add `Stdio::piped()` to worker spawn command
- [ ] Capture stdout stream
- [ ] Set up async reader for stdout

**Files to modify:**
- `bin/rbee-hive/src/worker_manager.rs`
- `bin/rbee-hive/src/worker.rs` (if exists)

**Code pattern:**
```rust
let mut child = Command::new("llm-worker-rbee")
    .args(&worker_args)
    .stdout(Stdio::piped())  // Capture stdout
    .stderr(Stdio::piped())  // Capture stderr
    .spawn()?;

let stdout = child.stdout.take().unwrap();
```

---

#### Task 3.2: Parse JSON Narration from Stdout
- [ ] Read stdout line by line
- [ ] Parse JSON log lines from tracing-subscriber
- [ ] Extract narration fields (actor, action, human, cute, etc.)
- [ ] Handle parse errors gracefully
- [ ] Filter out non-narration logs (if needed)

**Files to modify:**
- `bin/rbee-hive/src/worker_manager.rs`

**Code pattern:**
```rust
use tokio::io::{AsyncBufReadExt, BufReader};

let reader = BufReader::new(stdout);
let mut lines = reader.lines();

while let Some(line) = lines.next_line().await? {
    if let Ok(log) = serde_json::from_str::<serde_json::Value>(&line) {
        if log.get("actor").is_some() {
            // This is a narration event
            let narration = parse_narration_from_log(&log)?;
            // Send to SSE channel
        }
    }
}
```

---

#### Task 3.3: Convert Stdout to SSE Events
- [ ] Create SSE channel for worker narration
- [ ] Convert parsed narration to `InferenceEvent::Narration`
- [ ] Send to SSE stream
- [ ] Handle channel errors (closed, full, etc.)
- [ ] Add logging for debugging

**Files to modify:**
- `bin/rbee-hive/src/worker_manager.rs`

**Code pattern:**
```rust
fn parse_narration_from_log(log: &serde_json::Value) -> Result<NarrationEvent> {
    Ok(NarrationEvent {
        actor: log["actor"].as_str().unwrap_or("unknown").to_string(),
        action: log["action"].as_str().unwrap_or("unknown").to_string(),
        target: log["target"].as_str().unwrap_or("").to_string(),
        human: log["human"].as_str().unwrap_or("").to_string(),
        cute: log["cute"].as_str().map(String::from),
        story: log["story"].as_str().map(String::from),
        correlation_id: log["correlation_id"].as_str().map(String::from),
        job_id: log["job_id"].as_str().map(String::from),
    })
}
```

---

#### Task 3.4: Stream Narration to queen-rbee
- [ ] Add SSE endpoint for worker narration (if needed)
- [ ] Stream narration events to queen-rbee
- [ ] Handle worker lifecycle (startup, shutdown)
- [ ] Clean up streams when worker exits
- [ ] Add correlation ID to all events

**Files to modify:**
- `bin/rbee-hive/src/http/workers.rs` (if exists)
- `bin/rbee-hive/src/routes.rs`

**Code pattern:**
```rust
#[axum::debug_handler]
pub async fn stream_worker_narration(
    Path(worker_id): Path<String>,
) -> Result<Sse<EventStream>, StatusCode> {
    let worker = get_worker(&worker_id).ok_or(StatusCode::NOT_FOUND)?;
    
    // Stream narration events from worker
    let stream = worker.narration_rx.map(|event| {
        Ok(Event::default().json_data(&event).unwrap())
    });
    
    Ok(Sse::new(stream))
}
```

---

#### Task 3.5: Handle Worker Lifecycle Narration
- [ ] Capture narration during worker startup
- [ ] Capture narration during worker shutdown
- [ ] Handle worker crashes (stdout may close abruptly)
- [ ] Ensure all narration is sent before worker exit
- [ ] Add timeout for stdout reading

**Files to modify:**
- `bin/rbee-hive/src/worker_manager.rs`

---

#### Task 3.6: Integration Testing
- [ ] Test worker startup narration capture
- [ ] Test worker shutdown narration capture
- [ ] Test worker crash scenarios
- [ ] Test with multiple workers
- [ ] Verify no narration loss
- [ ] Test stdout parsing with malformed JSON

**Test scenarios:**
```bash
# Scenario 1: Normal worker lifecycle
# Start rbee-hive, spawn worker, run inference, shutdown
# Verify all narration captured

# Scenario 2: Worker crash
# Kill worker process mid-inference
# Verify narration up to crash point captured

# Scenario 3: Multiple workers
# Spawn 3 workers simultaneously
# Verify narration from all workers captured correctly
```

---

### Part 4: OpenAPI Specification Updates 📝 MEDIUM PRIORITY

**Location:** `contracts/openapi/worker.yaml`

#### Task 4.1: Document Narration Event Schema
- [ ] Add `NarrationEvent` schema to components
- [ ] Document all fields (actor, action, target, human, cute, story, correlation_id, job_id)
- [ ] Add field descriptions
- [ ] Add examples
- [ ] Mark optional fields correctly

**Files to modify:**
- `contracts/openapi/worker.yaml`

**Schema pattern:**
```yaml
components:
  schemas:
    NarrationEvent:
      type: object
      required:
        - type
        - actor
        - action
        - target
        - human
      properties:
        type:
          type: string
          enum: [narration]
        actor:
          type: string
          description: Component emitting the narration
          example: "candle-backend"
        action:
          type: string
          description: Action being performed
          example: "inference_start"
        target:
          type: string
          description: Target of the action
          example: "job-123"
        human:
          type: string
          description: Human-readable message
          example: "Starting inference (prompt: 15 chars, max_tokens: 50)"
        cute:
          type: string
          nullable: true
          description: Cute/friendly version of the message
          example: "Time to generate 50 tokens! Let's go! 🚀"
        story:
          type: string
          nullable: true
          description: Story-style narration
        correlation_id:
          type: string
          nullable: true
        job_id:
          type: string
          nullable: true
```

---

#### Task 4.2: Update /execute Endpoint Documentation
- [ ] Add `NarrationEvent` to SSE response oneOf
- [ ] Update event ordering documentation
- [ ] Add examples showing narration events
- [ ] Document event frequency (approximate)

**Files to modify:**
- `contracts/openapi/worker.yaml`

**Pattern:**
```yaml
/execute:
  post:
    summary: Execute inference request
    responses:
      '200':
        description: SSE stream of inference events
        content:
          text/event-stream:
            schema:
              oneOf:
                - $ref: '#/components/schemas/StartedEvent'
                - $ref: '#/components/schemas/TokenEvent'
                - $ref: '#/components/schemas/MetricsEvent'
                - $ref: '#/components/schemas/NarrationEvent'  # NEW
                - $ref: '#/components/schemas/EndEvent'
                - $ref: '#/components/schemas/ErrorEvent'
```

---

#### Task 4.3: Add Event Ordering Documentation
- [ ] Document expected event sequence
- [ ] Show narration events in sequence
- [ ] Add complete example SSE stream
- [ ] Document timing considerations

**Files to modify:**
- `contracts/openapi/worker.yaml`

**Example:**
```yaml
description: |
  SSE stream of inference events in the following order:
  1. narration (worker startup) - optional
  2. started (inference begins)
  3. narration (inference_start)
  4. narration (tokenize)
  5. token (generated tokens, 0 or more)
  6. narration (token_generate) - periodic
  7. narration (inference_complete)
  8. end (inference finished)
  
  Example stream:
  ```
  event: narration
  data: {"type":"narration","actor":"candle-backend","action":"inference_start","human":"Starting inference..."}
  
  event: started
  data: {"type":"started","job_id":"job-123","model":"llama-7b"}
  
  event: token
  data: {"type":"token","t":"Hello","i":0}
  
  event: narration
  data: {"type":"narration","actor":"candle-backend","action":"inference_complete","human":"Complete!"}
  
  event: end
  data: {"type":"end","tokens_out":50,"decode_time_ms":250}
  ```
```

---

### Part 5: Testing & Verification ✅ CRITICAL

#### Task 5.1: Unit Tests
- [ ] rbee-keeper: Test narration display function
- [ ] rbee-keeper: Test --quiet flag
- [ ] rbee-keeper: Test stderr/stdout separation
- [ ] queen-rbee: Test narration relay
- [ ] rbee-hive: Test stdout parsing
- [ ] rbee-hive: Test SSE conversion

**Test files to create:**
- `bin/rbee-keeper/tests/narration_display_test.rs`
- `bin/queen-rbee/tests/narration_relay_test.rs`
- `bin/rbee-hive/tests/stdout_capture_test.rs`

---

#### Task 5.2: Integration Tests
- [ ] End-to-end test: worker → queen-rbee → rbee-keeper
- [ ] Test with real inference request
- [ ] Verify all narration events appear
- [ ] Verify correct ordering
- [ ] Test error scenarios

**Test file to create:**
- `test-harness/bdd/tests/features/narration_e2e.feature`

**BDD scenario:**
```gherkin
Scenario: User sees real-time narration during inference
  Given rbee-hive is running on node "mac"
  And queen-rbee is running on orchestrator
  When user runs "rbee-keeper infer --node mac --model tinyllama --prompt 'hello'"
  Then user should see narration "[candle-backend] 🚀 Starting inference..."
  And user should see tokens "Hello world"
  And user should see narration "[candle-backend] 🎉 Complete!"
  And narration should appear on stderr
  And tokens should appear on stdout
```

---

#### Task 5.3: Manual Testing
- [ ] Test complete flow with real components
- [ ] Test with different models
- [ ] Test with long prompts
- [ ] Test with errors
- [ ] Test with --quiet flag
- [ ] Test piping to files
- [ ] Test with multiple concurrent requests

**Manual test script:**
```bash
#!/bin/bash
# test_narration_e2e.sh

echo "Test 1: Normal inference with narration"
rbee-keeper infer --node mac --model tinyllama --prompt "hello" 2>&1 | tee test1.log
grep "🚀" test1.log || echo "FAIL: No narration found"

echo "Test 2: Quiet mode"
rbee-keeper infer --node mac --model tinyllama --prompt "hello" --quiet 2>&1 | tee test2.log
grep "🚀" test2.log && echo "FAIL: Narration found in quiet mode" || echo "PASS"

echo "Test 3: Stderr/stdout separation"
rbee-keeper infer --node mac --model tinyllama --prompt "hello" 2>narration.log 1>tokens.txt
grep "🚀" narration.log || echo "FAIL: No narration in stderr"
grep "Hello" tokens.txt || echo "FAIL: No tokens in stdout"

echo "All tests complete"
```

---

#### Task 5.4: Performance Testing
- [ ] Measure narration overhead
- [ ] Test with high-frequency narration
- [ ] Verify no memory leaks in stdout capture
- [ ] Test with long-running workers
- [ ] Measure latency from narrate() to user display

**Performance targets:**
- Narration overhead: <1ms per event
- Memory usage: <10MB for stdout buffering
- Latency: <100ms from worker to user display

---

### Part 6: Documentation 📚 MEDIUM PRIORITY

#### Task 6.1: User Documentation
- [ ] Update rbee-keeper README with narration examples
- [ ] Document --quiet flag
- [ ] Add troubleshooting section
- [ ] Add examples of stderr/stdout redirection

**Files to create/modify:**
- `bin/rbee-keeper/README.md`
- `docs/USER_GUIDE.md`

---

#### Task 6.2: Developer Documentation
- [ ] Document narration architecture
- [ ] Add sequence diagrams
- [ ] Document stdout capture mechanism
- [ ] Add debugging guide

**Files to create:**
- `docs/NARRATION_ARCHITECTURE.md`
- `docs/DEBUGGING_NARRATION.md`

---

#### Task 6.3: API Documentation
- [ ] Update API docs with narration events
- [ ] Add client examples
- [ ] Document event ordering guarantees
- [ ] Add migration guide for existing clients

**Files to modify:**
- `docs/API.md`
- `docs/MIGRATION.md`

---

## 📊 Progress Tracking

### Overall Progress: 0/6 Parts Complete

- [ ] **Part 1: rbee-keeper** (0/5 tasks) - 🔴 CRITICAL
- [ ] **Part 2: queen-rbee** (0/5 tasks) - 🟡 HIGH
- [ ] **Part 3: rbee-hive** (0/6 tasks) - 🟡 HIGH
- [ ] **Part 4: OpenAPI** (0/3 tasks) - 📝 MEDIUM
- [ ] **Part 5: Testing** (0/4 tasks) - ✅ CRITICAL
- [ ] **Part 6: Documentation** (0/3 tasks) - 📚 MEDIUM

### Estimated Effort
- **rbee-keeper**: 1-2 days
- **queen-rbee**: 1-2 days
- **rbee-hive**: 2-3 days (most complex)
- **OpenAPI**: 0.5 day
- **Testing**: 1-2 days
- **Documentation**: 1 day

**Total: 6.5-10.5 days**

---

## 🎯 Success Criteria

### Must Have (P0)
- [x] Worker emits narration to SSE (TEAM-039 ✅)
- [ ] rbee-keeper displays narration to stderr
- [ ] rbee-keeper displays tokens to stdout
- [ ] --quiet flag works
- [ ] rbee-hive captures worker stdout
- [ ] rbee-hive converts stdout to SSE
- [ ] queen-rbee relays narration events
- [ ] End-to-end test passes

### Should Have (P1)
- [ ] OpenAPI spec updated
- [ ] Unit tests for all components
- [ ] Integration tests pass
- [ ] Performance targets met
- [ ] User documentation complete

### Nice to Have (P2)
- [ ] Developer documentation
- [ ] Debugging guide
- [ ] Performance benchmarks
- [ ] Migration guide

---

## 🚨 Critical Dependencies

### Blocked By
- ✅ TEAM-039 worker-side implementation (COMPLETE)

### Blocks
- User experience improvements
- Production readiness
- Client SDK updates

### External Dependencies
- None (all internal components)

---

## 📝 Notes for TEAM-040

### Key Insights from TEAM-039
1. **Dual-output is essential**: Narration goes to both stdout (logs) and SSE (users)
2. **Thread-local channels work well**: Use for per-request context
3. **Stream merging is straightforward**: Use `chain()` for sequential events
4. **Testing is critical**: Use `serial_test` for global state

### Common Pitfalls to Avoid
1. **Don't filter narration events**: Pass them through unchanged
2. **Preserve event ordering**: Narration must interleave with tokens correctly
3. **Handle stdout carefully**: Worker stdout contains JSON logs, parse carefully
4. **Test stderr/stdout separation**: Critical for piping workflows
5. **Don't forget correlation IDs**: Essential for debugging

### Architecture Decisions
- **Narration to stderr**: Allows piping tokens to files
- **Cute mode by default**: Use `cute` field if available, fallback to `human`
- **--quiet flag**: Suppresses narration, not tokens
- **No buffering**: Display narration immediately (line-buffered stderr)

---

## 🔗 References

### TEAM-039 Implementation
- `bin/.plan/TEAM_039_NARRATION_IMPLEMENTATION_SUMMARY.md`
- `bin/llm-worker-rbee/src/http/narration_channel.rs`
- `bin/llm-worker-rbee/src/http/sse.rs`
- `bin/llm-worker-rbee/src/narration.rs`

### Architecture Documents
- `bin/.specs/TEAM_038_NARRATION_DECISION.md`
- `bin/.specs/TEAM_038_NARRATION_FLOW_CORRECTED.md`
- `bin/.plan/TEAM_039_HANDOFF_NARRATION.md`
- `bin/llm-worker-rbee/.plan/CRITICAL_NARRATION_MISSING.md`
- `bin/llm-worker-rbee/.plan/NARRATION_WIRING_EXPLAINED.md`

### Narration Core
- `bin/shared-crates/narration-core/README.md`
- `bin/shared-crates/narration-core/src/lib.rs`

---

## ✅ Definition of Done

**This checklist is complete when:**

1. ✅ User runs `rbee-keeper infer` and sees narration in real-time
2. ✅ Narration appears on stderr (separate from tokens on stdout)
3. ✅ `--quiet` flag suppresses narration
4. ✅ Tokens can be piped to file without narration
5. ✅ All narration events from worker appear in user's shell
6. ✅ Event ordering is correct (narration interleaved with tokens)
7. ✅ rbee-hive captures worker stdout successfully
8. ✅ queen-rbee relays all events correctly
9. ✅ OpenAPI spec documents narration events
10. ✅ All tests pass (unit + integration)
11. ✅ User documentation complete
12. ✅ Performance targets met

---

**Created by:** TEAM-039 (handoff to TEAM-040)  
**Date:** 2025-10-10  
**Status:** Ready for implementation  
**Priority:** P0 - Critical for user experience

---
Verified by Testing Team 🔍
