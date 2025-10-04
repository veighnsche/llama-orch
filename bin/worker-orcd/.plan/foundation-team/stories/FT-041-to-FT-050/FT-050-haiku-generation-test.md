# FT-050: Haiku Generation Test (M0 Success Criteria - Anti-Cheat)

**Team**: Foundation-Alpha  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (2 days)  
**Days**: 75 - 76  
**Spec Ref**: M0-W-1800, `.docs/testing/types/e2e-haiku.md`

---

## Story Description

Implement the canonical M0 success test: anti-cheat haiku generation that proves real GPU inference by requiring the model to include the current minute (in words) within a haiku. This prevents pre-baked outputs and validates genuine token generation.

---

## Acceptance Criteria

- [ ] Test loads Qwen2.5-0.5B-Instruct on real GPU
- [ ] Prompt includes current minute in words (e.g., "twenty-nine")
- [ ] Optional 8-character nonce for additional cache-busting
- [ ] Test validates haiku contains the minute word exactly once
- [ ] Test validates SSE stream format (started → token* → end)
- [ ] Test validates VRAM-only operation
- [ ] Test validates metrics delta (tokens_out increases)
- [ ] Proof bundle artifacts generated (see below)
- [ ] Test runs in CI with `REQUIRE_REAL_LLAMA=1`
- [ ] Finishes within ≤30 seconds

---

## Dependencies

**Upstream**: FT-040 (Performance baseline, Day 75)  
**Downstream**: FT-047 (Gate 4 checkpoint)

---

## Technical Details

### Anti-Cheat Design

```rust
use chrono::Utc;
use rand::Rng;

fn minute_to_words(minute: u32) -> String {
    let ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
    let teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
                 "sixteen", "seventeen", "eighteen", "nineteen"];
    let tens = ["", "", "twenty", "thirty", "forty", "fifty"];
    
    match minute {
        0..=9 => ones[minute as usize].to_string(),
        10..=19 => teens[(minute - 10) as usize].to_string(),
        20..=59 => {
            let ten = tens[(minute / 10) as usize];
            let one = minute % 10;
            if one == 0 {
                ten.to_string()
            } else {
                format!("{}-{}", ten, ones[one as usize])
            }
        }
        _ => panic!("Invalid minute: {}", minute),
    }
}

#[tokio::test]
#[cfg(feature = "cuda")]
async fn test_haiku_generation_anti_cheat() {
    // Enforce real GPU requirement
    std::env::set_var("REQUIRE_REAL_LLAMA", "1");
    
    let harness = WorkerTestHarness::start(
        ".test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        0
    ).await.expect("Failed to start worker");
    
    // Generate dynamic minute word
    let now = Utc::now();
    let minute = now.minute();
    let minute_word = minute_to_words(minute);
    
    // Generate nonce
    let nonce: String = rand::thread_rng()
        .sample_iter(&rand::distributions::Alphanumeric)
        .take(8)
        .map(char::from)
        .collect();
    
    // Construct anti-cheat prompt
    let prompt = format!(
        "Write a haiku about GPU computing that includes the word \"{}\" (nonce: {})",
        minute_word, nonce
    );
    
    let run_id = std::env::var("LLORCH_RUN_ID")
        .unwrap_or_else(|_| uuid::Uuid::new_v4().to_string());
    
    let req = ExecuteRequest {
        job_id: format!("m0-haiku-anti-cheat-{}", run_id),
        prompt: prompt.clone(),
        max_tokens: 100,
        temperature: 0.7,
        seed: now.timestamp() as u64, // Time-based seed
    };
    
    // Capture metrics before
    let metrics_before = harness.get_metrics().await.expect("Failed to get metrics");
    let tokens_before = metrics_before.tokens_out_total;
    
    let start_time = std::time::Instant::now();
    let stream = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(stream).await;
    let elapsed = start_time.elapsed();
    
    // Capture metrics after
    let metrics_after = harness.get_metrics().await.expect("Failed to get metrics");
    let tokens_after = metrics_after.tokens_out_total;
    
    // Validate event sequence
    assert_event!(events[0], InferenceEvent::Started);
    let tokens = extract_tokens(&events);
    assert!(!tokens.is_empty(), "No tokens generated");
    assert_event!(events.last().unwrap(), InferenceEvent::End);
    
    let haiku = tokens.join("");
    
    // Anti-cheat validation
    let minute_word_count = haiku.matches(&minute_word).count();
    assert_eq!(
        minute_word_count, 1,
        "Haiku must contain minute word '{}' exactly once, found {} times",
        minute_word, minute_word_count
    );
    
    // Validate metrics delta
    let tokens_generated = tokens_after - tokens_before;
    assert!(
        tokens_generated > 0,
        "Metrics show no tokens generated (before: {}, after: {})",
        tokens_before, tokens_after
    );
    
    // Validate timing
    assert!(
        elapsed.as_secs() <= 30,
        "Test took too long: {:?}",
        elapsed
    );
    
    // Generate proof bundle
    let proof_bundle = ProofBundle::for_type(TestType::E2EHaiku, &run_id);
    
    // Save artifacts
    proof_bundle.write_json("verification.json", &serde_json::json!({
        "minute": minute,
        "minute_word": minute_word,
        "nonce": nonce,
        "prompt": prompt,
        "haiku": haiku,
        "minute_word_count": minute_word_count,
        "tokens_generated": tokens_generated,
        "elapsed_ms": elapsed.as_millis(),
        "timestamp": now.to_rfc3339(),
    })).unwrap();
    
    proof_bundle.write_ndjson("sse_transcript.ndjson", &events).unwrap();
    
    proof_bundle.write_json("metrics_snapshot.json", &serde_json::json!({
        "before": {
            "tokens_out_total": tokens_before,
        },
        "after": {
            "tokens_out_total": tokens_after,
        },
        "delta": tokens_generated,
    })).unwrap();
    
    proof_bundle.write_json("gpu_env.json", &serde_json::json!({
        "device": harness.gpu_device(),
        "model": harness.model_name(),
        "vram_bytes": harness.vram_usage(),
    })).unwrap();
    
    proof_bundle.write_markdown("test_report.md", &format!(
        "# M0 Haiku Anti-Cheat Test Report\n\n\
         **Run ID**: {}\n\
         **Timestamp**: {}\n\
         **Minute**: {} (\"{}\")\n\
         **Nonce**: {}\n\
         **Tokens Generated**: {}\n\
         **Elapsed**: {:?}\n\n\
         ## Haiku Output\n\n```\n{}\n```\n\n\
         ## Validation\n\n\
         - ✅ Minute word \"{}\" found exactly once\n\
         - ✅ Tokens generated: {}\n\
         - ✅ Completed in {:?}\n\
         - ✅ Real GPU used (VRAM-only)\n",
        run_id, now.to_rfc3339(), minute, minute_word, nonce,
        tokens_generated, elapsed, haiku, minute_word, tokens_generated, elapsed
    )).unwrap();
    
    println!("\n🎨 M0 Haiku Anti-Cheat Test PASSED");
    println!("Minute: {} (\"{}\")", minute, minute_word);
    println!("Nonce: {}", nonce);
    println!("Tokens: {}", tokens_generated);
    println!("Time: {:?}", elapsed);
    println!("\nHaiku:\n{}\n", haiku);
    println!("Proof bundle: {}", proof_bundle.path().display());
}
```

---

## Proof Bundle Artifacts

Located in: `bin/worker-orcd/.proof_bundle/e2e-haiku/<run_id>/`

- **verification.json**: Minute word, nonce, validation results
- **sse_transcript.ndjson**: Complete SSE event stream
- **metrics_snapshot.json**: Prometheus metrics before/after
- **gpu_env.json**: GPU device info, model name, VRAM usage
- **test_report.md**: Human-readable summary with haiku output

---

## Anti-Cheat Guarantees

1. **No pre-baked outputs**: Minute word changes every minute
2. **No caching**: Optional nonce adds uniqueness
3. **Real GPU required**: `REQUIRE_REAL_LLAMA=1` enforced
4. **Metrics validation**: Must observe token count increase
5. **Repository scan safe**: No hardcoded minute+nonce combinations

---

## Testing Strategy

### Unit Tests
- Test minute_to_words() for all 60 minutes
- Test nonce generation (8 chars, alphanumeric)
- Test proof bundle artifact creation

### Integration Tests
- Test with real Qwen2.5-0.5B model
- Test minute word detection in output
- Test metrics delta validation
- Test timing constraint (≤30s)

### Manual Verification
1. Run test: `cargo test --features cuda test_haiku_generation_anti_cheat`
2. Verify minute word in output
3. Check proof bundle artifacts
4. Validate metrics increased

---

## Definition of Done

- [ ] Test passes with anti-cheat validation
- [ ] Haiku contains minute word exactly once
- [ ] Metrics delta validated
- [ ] Proof bundle artifacts complete
- [ ] Test runs in CI
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04  
**Note**: 🎨 **M0 SUCCESS CRITERIA** - Anti-cheat definitive test

---

## References

- Anti-cheat spec: `.docs/testing/types/e2e-haiku.md`
- Proof bundle spec: `.specs/00_proof-bundle.md`
- Test types guide: `.docs/testing/TEST_TYPES_GUIDE.md`

---
Planned by Project Management Team 📋

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **Test started**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "test_start",
       target: "m0-haiku-anti-cheat".to_string(),
       human: format!("Starting M0 haiku anti-cheat test (minute: {})", minute_word),
       ..Default::default()
   });
   ```

2. **Test passed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "test_complete",
       target: "m0-haiku-anti-cheat".to_string(),
       tokens_out: Some(tokens_generated),
       duration_ms: Some(elapsed.as_millis() as u64),
       human: format!("M0 haiku test PASSED: {} tokens in {} ms", tokens_generated, elapsed.as_millis()),
       ..Default::default()
   });
   ```

3. **Anti-cheat validation failed**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_WORKER_ORCD,
       action: "test_complete",
       target: "m0-haiku-anti-cheat".to_string(),
       error_kind: Some("anti_cheat_failed".to_string()),
       human: format!("M0 haiku test FAILED: minute word '{}' not found in output", minute_word),
       ..Default::default()
   });
   ```

**Why this matters**: The M0 haiku test is the definitive success criteria. Narration creates an audit trail proving real GPU inference occurred.

---
*Narration guidance added by Narration-Core Team 🎀*
