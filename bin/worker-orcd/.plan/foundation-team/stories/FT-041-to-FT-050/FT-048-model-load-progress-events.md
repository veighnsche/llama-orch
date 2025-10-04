# FT-048: Model Load Progress Events

**Team**: Foundation-Alpha  
**Sprint**: Sprint 7 - Final Integration  
**Size**: S (1 day)  
**Days**: 72 - 72  
**Spec Ref**: M0-W-1621

---

## Story Description

Emit progress events during model loading for observability: 0%, 25%, 50%, 75%, 100%. Enables monitoring of long-running model loads (especially GPT-OSS-20B).

---

## Acceptance Criteria

- [ ] Progress events logged at 0%, 25%, 50%, 75%, 100%
- [ ] Events include model name and VRAM usage
- [ ] Events use structured logging
- [ ] Integration with tracing
- [ ] Unit tests for progress tracking
- [ ] Integration test validates events

---

## Dependencies

**Upstream**: FT-039 (CI/CD, Day 73)  
**Downstream**: FT-047 (Gate 4)

---

## Technical Details

```rust
tracing::info!(
    event = "model_load_progress",
    model = %model_name,
    percent = 25,
    vram_bytes = vram_used,
    "Model loading 25% complete"
);
```

---

## Definition of Done

- [ ] Progress events implemented
- [ ] Tests passing
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

1. **Model load started**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "model_load",
       target: model_path.to_string(),
       model_ref: Some(model_name.clone()),
       human: format!("Loading model: {}", model_name),
       ..Default::default()
   });
   ```

2. **Model load progress**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "model_load_progress",
       target: model_name.clone(),
       model_ref: Some(model_name.clone()),
       human: format!("Model loading: {}% ({} MB / {} MB)", percentage, loaded_mb, total_mb),
       ..Default::default()
   });
   ```

3. **Model load completed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "model_load",
       target: model_name.clone(),
       model_ref: Some(model_name.clone()),
       duration_ms: Some(elapsed.as_millis() as u64),
       human: format!("Model loaded: {} ({} MB VRAM, {} ms)", model_name, vram_mb, elapsed.as_millis()),
       ..Default::default()
   });
   ```

**Why this matters**: Model loading can take seconds. Progress events provide user feedback and help diagnose slow loads.

---
*Narration guidance added by Narration-Core Team 🎀*
