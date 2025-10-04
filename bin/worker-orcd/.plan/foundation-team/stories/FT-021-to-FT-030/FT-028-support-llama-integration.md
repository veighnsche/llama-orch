# FT-028: Support Llama Integration

**Team**: Foundation-Alpha  
**Sprint**: Sprint 5 - Support + Prep  
**Size**: M (2 days)  
**Days**: 53 - 54  
**Spec Ref**: Cross-team coordination

---

## Story Description

Provide integration support for Llama team as they implement Llama-specific kernels and adapters. This includes code reviews, debugging assistance, and interface clarifications.

---

## Acceptance Criteria

- [ ] Llama team has access to all Foundation APIs
- [ ] FFI interface questions answered
- [ ] Integration issues debugged
- [ ] Code reviews provided
- [ ] Documentation clarified as needed
- [ ] Llama kernels integrate with Foundation layer
- [ ] No blocking issues for Llama team

---

## Dependencies

### Upstream (Blocks This Story)
- FT-027: Gate 1 checkpoint (Expected completion: Day 52)

### Downstream (This Story Blocks)
- LT-027: Llama Gate 2 checkpoint

---

## Definition of Done

- [ ] Llama team unblocked
- [ ] Integration successful
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

1. **Llama model loaded**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "model_load",
       target: model_name.clone(),
       device: Some(format!("GPU{}", device_id)),
       model_ref: Some(model_name.clone()),
       duration_ms: Some(elapsed.as_millis() as u64),
       human: format!("Loaded Llama model {} ({} MB VRAM, {} ms)", model_name, vram_mb, elapsed.as_millis()),
       ..Default::default()
   });
   ```

2. **Llama inference completed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: ACTION_INFERENCE_COMPLETE,
       target: job_id.clone(),
       correlation_id: Some(correlation_id),
       model_ref: Some("llama".to_string()),
       tokens_out: Some(tokens_generated),
       duration_ms: Some(elapsed.as_millis() as u64),
       human: format!("Llama inference complete: {} tokens in {} ms", tokens_generated, elapsed.as_millis()),
       ..Default::default()
   });
   ```

**Why this matters**: Llama integration is a major architecture milestone. Narration helps track model loading and inference performance.

---
*Narration guidance added by Narration-Core Team 🎀*
