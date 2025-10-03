# FT-043: UTF-8 Streaming Edge Cases

**Team**: Foundation-Alpha  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (2 days)  
**Days**: 81 - 82  
**Spec Ref**: M0-W-1361, UTF-8 safety

---

## Story Description

Test UTF-8 streaming edge cases: multibyte characters split across tokens, emoji, special characters, various languages. Validate SSE stream never emits invalid UTF-8.

---

## Acceptance Criteria

- [ ] Test emoji in prompts and outputs
- [ ] Test multibyte characters (Chinese, Arabic, etc.)
- [ ] Test characters split across token boundaries
- [ ] Test UTF-8 buffer handles partial sequences
- [ ] Test SSE stream is always valid UTF-8
- [ ] Test with all three models
- [ ] Test edge cases documented

---

## Dependencies

**Upstream**: FT-042 (OOM recovery, Day 80)  
**Downstream**: FT-047 (Gate 4 checkpoint)

---

## Technical Details

```rust
#[tokio::test]
async fn test_emoji_streaming() {
    let harness = WorkerTestHarness::start("qwen2.5-0.5b-instruct-q4_k_m.gguf", 0).await.unwrap();
    
    let req = ExecuteRequest {
        job_id: "test-emoji".to_string(),
        prompt: "Respond with emoji: 👋 🌍 🚀".to_string(),
        max_tokens: 20,
        temperature: 0.7,
        seed: 42,
    };
    
    let stream = harness.execute(req).await.unwrap();
    let events = collect_sse_events(stream).await;
    let tokens = extract_tokens(&events);
    
    // Validate all tokens are valid UTF-8
    for token in &tokens {
        assert!(std::str::from_utf8(token.as_bytes()).is_ok());
    }
}
```

---

## Definition of Done

- [ ] All UTF-8 edge cases tested
- [ ] No invalid UTF-8 emitted
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
