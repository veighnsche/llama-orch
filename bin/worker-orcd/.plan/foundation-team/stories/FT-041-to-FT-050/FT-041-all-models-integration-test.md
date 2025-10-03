# FT-041: All Models Integration Test

**Team**: Foundation-Alpha  
**Sprint**: Sprint 7 - Final Integration  
**Size**: L (3 days)  
**Days**: 76 - 78  
**Spec Ref**: M0-W-1230, M0-W-1820

---

## Story Description

Comprehensive integration test validating all three M0 models (Qwen2.5-0.5B, Phi-3-Mini, GPT-OSS-20B) work correctly with complete inference pipeline.

---

## Acceptance Criteria

- [ ] Test loads all three models sequentially
- [ ] Test generates tokens from each model
- [ ] Test validates determinism (temp=0)
- [ ] Test validates VRAM cleanup between models
- [ ] Test validates correct adapter selection
- [ ] Test validates tokenizer integration
- [ ] Test runs in CI
- [ ] Proof bundle artifacts generated

---

## Dependencies

**Upstream**: FT-040 (Performance baseline, Day 75)  
**Downstream**: FT-047 (Gate 4 checkpoint)

---

## Technical Details

```rust
#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_all_three_models() {
    let models = vec![
        ("qwen2.5-0.5b-instruct-q4_k_m.gguf", "llama"),
        ("phi-3-mini-4k-instruct-q4.gguf", "llama"),
        ("gpt-oss-20b-mxfp4.gguf", "gpt"),
    ];
    
    for (model_path, expected_arch) in models {
        let harness = WorkerTestHarness::start(model_path, 0).await.unwrap();
        
        let req = ExecuteRequest {
            job_id: format!("test-{}", model_path),
            prompt: "Count to three".to_string(),
            max_tokens: 20,
            temperature: 0.0,
            seed: 42,
        };
        
        let stream = harness.execute(req).await.unwrap();
        let events = collect_sse_events(stream).await;
        let tokens = extract_tokens(&events);
        
        assert!(!tokens.is_empty(), "Model {} generated no tokens", model_path);
    }
}
```

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Test passes with all models
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
