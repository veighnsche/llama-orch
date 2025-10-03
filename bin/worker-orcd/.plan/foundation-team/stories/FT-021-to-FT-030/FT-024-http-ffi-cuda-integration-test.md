# FT-024: HTTP-FFI-CUDA Integration Test

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Integration + Gate 1  
**Size**: L (3 days)  
**Days**: 45 - 47  
**Spec Ref**: M0-W-1820

---

## Story Description

Implement comprehensive end-to-end integration test validating complete HTTP → Rust → FFI → C++ → CUDA → C++ → FFI → Rust → HTTP flow with real model inference.

---

## Acceptance Criteria

- [ ] Test loads real model (Qwen2.5-0.5B-Instruct)
- [ ] Test sends execute request via HTTP
- [ ] Test receives SSE stream with tokens
- [ ] Test validates event sequence: started → token* → end
- [ ] Test validates determinism (same seed → same tokens)
- [ ] Test validates VRAM-only operation
- [ ] Test validates health endpoint during inference
- [ ] Test validates cancellation
- [ ] Test runs in CI with CUDA feature flag

---

## Dependencies

### Upstream (Blocks This Story)
- FT-023: Integration test framework (Expected completion: Day 44)
- All previous Foundation stories (FT-001 to FT-022)

### Downstream (This Story Blocks)
- FT-025: Gate 1 validation tests
- FT-027: Gate 1 checkpoint

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/e2e_test.rs` - End-to-end test
- `bin/worker-orcd/tests/integration/determinism_test.rs` - Determinism test
- `bin/worker-orcd/tests/integration/cancellation_test.rs` - Cancellation test

### Key Test Cases
```rust
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_complete_inference_flow() {
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let req = ExecuteRequest {
        job_id: "test-001".to_string(),
        prompt: "Write a haiku about GPU computing".to_string(),
        max_tokens: 50,
        temperature: 0.7,
        seed: 42,
    };
    
    let stream = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(stream).await;
    
    // Validate event sequence
    assert!(matches!(events[0], InferenceEvent::Started { .. }));
    assert!(events.iter().any(|e| matches!(e, InferenceEvent::Token { .. })));
    assert!(matches!(events.last().unwrap(), InferenceEvent::End { .. }));
    
    // Extract tokens
    let tokens = extract_tokens(&events);
    assert!(!tokens.is_empty());
}

#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_determinism() {
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    let req = ExecuteRequest {
        job_id: "test-det-1".to_string(),
        prompt: "Count to five".to_string(),
        max_tokens: 20,
        temperature: 0.0,  // Greedy for determinism
        seed: 42,
    };
    
    // Run twice
    let stream1 = harness.execute(req.clone()).await.unwrap();
    let events1 = collect_sse_events(stream1).await;
    let tokens1 = extract_tokens(&events1);
    
    let req2 = ExecuteRequest { job_id: "test-det-2".to_string(), ..req };
    let stream2 = harness.execute(req2).await.unwrap();
    let events2 = collect_sse_events(stream2).await;
    let tokens2 = extract_tokens(&events2);
    
    // Same seed + temp=0 should produce identical tokens
    assert_eq!(tokens1, tokens2);
}
```

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Test passes with real model
- [ ] Determinism validated
- [ ] CI integration complete
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` §12.3 Integration Tests (M0-W-1820)
- Related Stories: FT-023 (framework), FT-025 (gate 1)

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
