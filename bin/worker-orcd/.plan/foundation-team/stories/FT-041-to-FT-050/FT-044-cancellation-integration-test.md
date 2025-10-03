# FT-044: Cancellation Integration Test

**Team**: Foundation-Alpha  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (2 days)  
**Days**: 83 - 84  
**Spec Ref**: M0-W-1330, M0-W-1610

---

## Story Description

Test cancellation flow: POST /cancel during inference, client disconnect detection, VRAM cleanup, and cancellation latency validation.

---

## Acceptance Criteria

- [ ] Test POST /cancel stops inference
- [ ] Test SSE error event with code CANCELLED
- [ ] Test VRAM freed after cancellation
- [ ] Test cancellation latency <100ms
- [ ] Test client disconnect detected
- [ ] Test multiple cancellations (idempotent)
- [ ] Test cancellation during different phases

---

## Dependencies

**Upstream**: FT-043 (UTF-8 edge cases, Day 82)  
**Downstream**: FT-047 (Gate 4 checkpoint)

---

## Technical Details

```rust
#[tokio::test]
async fn test_cancellation() {
    let harness = WorkerTestHarness::start("qwen2.5-0.5b-instruct-q4_k_m.gguf", 0).await.unwrap();
    
    let req = ExecuteRequest {
        job_id: "test-cancel".to_string(),
        prompt: "Write a very long story".to_string(),
        max_tokens: 1000,
        temperature: 0.7,
        seed: 42,
    };
    
    let stream = harness.execute(req).await.unwrap();
    
    // Wait for a few tokens
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    // Cancel
    let start = Instant::now();
    harness.cancel("test-cancel").await.unwrap();
    let cancel_latency = start.elapsed();
    
    // Collect remaining events
    let events = collect_sse_events(stream).await;
    
    // Should have CANCELLED error
    assert!(events.iter().any(|e| matches!(e, InferenceEvent::Error { code, .. } if code == "CANCELLED")));
    
    // Cancellation should be fast
    assert!(cancel_latency < Duration::from_millis(100));
}
```

---

## Definition of Done

- [ ] All cancellation scenarios tested
- [ ] Latency requirements met
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
