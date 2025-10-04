# FT-042: OOM Recovery Test

**Team**: Foundation-Alpha  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (2 days)  
**Days**: 79 - 80  
**Spec Ref**: M0-W-1220, Error handling

---

## Story Description

Test OOM (Out of Memory) scenarios and recovery: model too large, KV cache overflow, intentional VRAM exhaustion. Validate error reporting and graceful degradation.

---

## Acceptance Criteria

- [ ] Test model load OOM (insufficient VRAM)
- [ ] Test KV cache OOM (context too long)
- [ ] Test error messages include VRAM usage
- [ ] Test worker stays alive after OOM
- [ ] Test subsequent requests work after OOM
- [ ] Test SSE error event format
- [ ] Test health endpoint reports unhealthy

---

## Dependencies

**Upstream**: FT-041 (All models test, Day 78)  
**Downstream**: FT-047 (Gate 4 checkpoint)

---

## Technical Details

```rust
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_kv_cache_oom() {
    let harness = WorkerTestHarness::start("qwen2.5-0.5b-instruct-q4_k_m.gguf", 0).await.unwrap();
    
    let req = ExecuteRequest {
        job_id: "test-oom".to_string(),
        prompt: "x".repeat(100000), // Intentionally too long
        max_tokens: 10,
        temperature: 0.0,
        seed: 42,
    };
    
    let stream = harness.execute(req).await.unwrap();
    let events = collect_sse_events(stream).await;
    
    // Should get error event
    assert!(events.iter().any(|e| matches!(e, InferenceEvent::Error { code, .. } if code == "VRAM_OOM")));
}
```

---

## Definition of Done

- [ ] All OOM scenarios tested
- [ ] Error handling validated
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **OOM test started**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "test_start",
       target: "oom-recovery".to_string(),
       human: "Starting OOM recovery test".to_string(),
       ..Default::default()
   });
   ```

2. **OOM triggered**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_VRAM_RESIDENCY,
       action: ACTION_VRAM_ALLOCATE,
       target: "oom-trigger".to_string(),
       error_kind: Some("vram_oom".to_string()),
       human: format!("OOM triggered: requested {} MB, only {} MB available", requested / 1024 / 1024, available / 1024 / 1024),
       ..Default::default()
   });
   ```

3. **OOM recovery successful**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "test_complete",
       target: "oom-recovery".to_string(),
       duration_ms: Some(elapsed.as_millis() as u64),
       human: format!("OOM recovery test PASSED: graceful error handling verified ({} ms)", elapsed.as_millis()),
       ..Default::default()
   });
   ```

**Why this matters**: OOM recovery is critical for production stability. Narration verifies graceful handling and helps diagnose recovery failures.

---
*Narration guidance added by Narration-Core Team 🎀*
