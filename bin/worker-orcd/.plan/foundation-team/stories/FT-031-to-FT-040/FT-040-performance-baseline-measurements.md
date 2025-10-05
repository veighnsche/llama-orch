# FT-040: Performance Baseline Measurements
**Team**: Foundation-Alpha  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (2 days)  
**Days**: 74 - 75  
**Spec Ref**: M0-W-1600, M0-W-1601, M0-W-1602
---
## Story Description
Execute performance baseline measurements for all three M0 models: Qwen2.5-0.5B, Phi-3-Mini, GPT-OSS-20B. Document first-token latency, tokens/sec, and per-token latency.
---
## Acceptance Criteria
- [ ] Qwen2.5-0.5B baseline measured
- [ ] Phi-3-Mini baseline measured
- [ ] GPT-OSS-20B baseline measured
- [ ] Results documented in CSV/JSON
- [ ] Comparison with spec targets
- [ ] Performance report generated
- [ ]  artifacts created
---
## Dependencies
**Upstream**: FT-031 (Baseline prep, Day 61), FT-039 (CI/CD, Day 73)  
**Downstream**: FT-047 (Gate 4 checkpoint)
---
## Target Metrics
- **First token latency**: <100ms (p95)
- **Tokens/sec**: 20-100 depending on model
- **Per-token latency**: 10-50ms (p95)
---
## Definition of Done
- [ ] All baselines measured
- [ ] Report published
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
1. **Baseline measurement started**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "baseline_measure",
       target: benchmark_name.to_string(),
       model_ref: Some(model_name.clone()),
       human: format!("Starting baseline measurement: {} with {}", benchmark_name, model_name),
       ..Default::default()
   });
   ```
2. **Baseline measurement completed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "baseline_measure",
       target: benchmark_name.to_string(),
       model_ref: Some(model_name.clone()),
       tokens_out: Some(tokens_generated),
       duration_ms: Some(elapsed.as_millis() as u64),
       human: format!("Baseline: {} = {} tokens/sec ({} tokens in {} ms)", benchmark_name, tokens_per_sec, tokens_generated, elapsed.as_millis()),
       ..Default::default()
   });
   ```
3. **Baseline comparison**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "baseline_compare",
       target: benchmark_name.to_string(),
       human: format!("Baseline comparison: current={} tokens/sec, target={} tokens/sec ({}% of target)", current, target, percentage),
       ..Default::default()
   });
   ```
**Why this matters**: Performance baselines establish optimization targets. Narration tracks measurements and comparisons for performance analysis.
---
*Narration guidance added by Narration-Core Team 🎀*
