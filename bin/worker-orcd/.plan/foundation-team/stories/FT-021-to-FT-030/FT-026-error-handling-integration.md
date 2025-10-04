# FT-026: Error Handling Integration

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Integration + Gate 1  
**Size**: M (2 days)  
**Days**: 50 - 51  
**Spec Ref**: M0-W-1500, M0-W-1510

---

## Story Description

Integrate error handling across all layers (HTTP, Rust, FFI, C++, CUDA) with proper error propagation and user-friendly messages.

---

## Acceptance Criteria

- [ ] CUDA errors propagate to HTTP responses
- [ ] Error codes stable and documented
- [ ] Error messages include context
- [ ] SSE error events formatted correctly
- [ ] OOM errors include VRAM usage info
- [ ] Unit tests for error conversion
- [ ] Integration tests for error scenarios

---

## Dependencies

### Upstream (Blocks This Story)
- FT-009: Rust error conversion (Expected completion: Day 15)
- FT-014: VRAM residency verification (Expected completion: Day 27)

### Downstream (This Story Blocks)
- FT-027: Gate 1 checkpoint

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Error handling tested
- [ ] Story marked complete in day-tracker.md

---

**Status**: 📋 Ready for execution  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **Error propagated across layers**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: ACTION_INFERENCE_ERROR,
       target: job_id.clone(),
       correlation_id: Some(correlation_id),
       error_kind: Some(error.code().to_string()),
       human: format!("Error propagated: {} -> {} -> HTTP {}", cuda_error, rust_error, http_status),
       ..Default::default()
   });
   ```

2. **Error recovery attempted**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "error_recovery",
       target: job_id.clone(),
       correlation_id: Some(correlation_id),
       human: format!("Attempting error recovery for job {}", job_id),
       ..Default::default()
   });
   ```

3. **Graceful degradation**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "graceful_degradation",
       target: "worker".to_string(),
       human: format!("Gracefully degraded: {}", degradation_reason),
       ..Default::default()
   });
   ```

**Why this matters**: Error handling integration ensures errors propagate correctly. Narration helps trace error flows across layers.

---
*Narration guidance added by Narration-Core Team 🎀*
