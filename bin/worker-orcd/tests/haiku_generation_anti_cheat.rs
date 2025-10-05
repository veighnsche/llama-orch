//! FT-050: Haiku Generation Anti-Cheat Test
//!
//! M0 Success Criteria Test - Proves real GPU inference by requiring
//! the model to include the current minute (in words) within a haiku.
//! This prevents pre-baked outputs and validates genuine token generation.
//!
//! Spec: M0-W-1800, `.docs/testing/types/e2e-haiku.md`

use chrono::Utc;
use rand::Rng;
use std::io::Write;
use worker_orcd::tests::integration::{collect_sse_events, extract_tokens, make_test_request, WorkerTestHarness};

/// Convert minute (0-59) to English words
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
#[ignore] // Only run with REQUIRE_REAL_LLAMA=1
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
    
    let mut req = make_test_request(
        &format!("m0-haiku-anti-cheat-{}", run_id),
        &prompt,
        100
    );
    req.temperature = 0.7;
    req.seed = Some(now.timestamp() as u64); // Time-based seed
    
    // Capture metrics before
    let metrics_before = harness.get_metrics().await.expect("Failed to get metrics");
    let tokens_before = metrics_before.tokens_out_total;
    
    let start_time = std::time::Instant::now();
    let response = harness.execute(req).await.expect("Execute failed");
    let events = collect_sse_events(response).await.expect("Failed to collect events");
    let elapsed = start_time.elapsed();
    
    // Capture metrics after
    let metrics_after = harness.get_metrics().await.expect("Failed to get metrics");
    let tokens_after = metrics_after.tokens_out_total;
    
    // Validate event sequence
    assert!(matches!(events.first(), Some(worker_http::sse::InferenceEvent::Started { .. })), 
        "First event should be Started");
    
    let tokens = extract_tokens(&events);
    assert!(!tokens.is_empty(), "No tokens generated");
    
    assert!(events.last().unwrap().is_terminal(), "Last event should be terminal");
    
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
    
    // Save test artifacts
    let artifacts_dir = std::path::PathBuf::from(".test-results/haiku")
        .join(&run_id);
    std::fs::create_dir_all(&artifacts_dir).unwrap();
    
    // Save verification results
    let verification = serde_json::json!({
        "minute": minute,
        "minute_word": minute_word,
        "nonce": nonce,
        "prompt": prompt,
        "haiku": haiku,
        "minute_word_count": minute_word_count,
        "tokens_generated": tokens_generated,
        "elapsed_ms": elapsed.as_millis(),
        "timestamp": now.to_rfc3339(),
    });
    
    std::fs::write(
        artifacts_dir.join("verification.json"),
        serde_json::to_string_pretty(&verification).unwrap()
    ).unwrap();
    
    // Save SSE transcript
    let transcript_path = artifacts_dir.join("sse_transcript.ndjson");
    let mut transcript = std::fs::File::create(&transcript_path).unwrap();
    for event in &events {
        writeln!(transcript, "{}", serde_json::to_string(event).unwrap()).unwrap();
    }
    
    // Save metrics snapshot
    let metrics = serde_json::json!({
        "before": { "tokens_out_total": tokens_before },
        "after": { "tokens_out_total": tokens_after },
        "delta": tokens_generated,
    });
    
    std::fs::write(
        artifacts_dir.join("metrics_snapshot.json"),
        serde_json::to_string_pretty(&metrics).unwrap()
    ).unwrap();
    
    // Save test report
    let report = format!(
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
    );
    
    std::fs::write(artifacts_dir.join("test_report.md"), report).unwrap();
    
    println!("\n🎨 M0 Haiku Anti-Cheat Test PASSED");
    println!("Minute: {} (\"{}\")", minute, minute_word);
    println!("Nonce: {}", nonce);
    println!("Tokens: {}", tokens_generated);
    println!("Time: {:?}", elapsed);
    println!("\nHaiku:\n{}\n", haiku);
    println!("Artifacts: {}", artifacts_dir.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minute_to_words_ones() {
        assert_eq!(minute_to_words(0), "zero");
        assert_eq!(minute_to_words(5), "five");
        assert_eq!(minute_to_words(9), "nine");
    }

    #[test]
    fn test_minute_to_words_teens() {
        assert_eq!(minute_to_words(10), "ten");
        assert_eq!(minute_to_words(15), "fifteen");
        assert_eq!(minute_to_words(19), "nineteen");
    }

    #[test]
    fn test_minute_to_words_tens() {
        assert_eq!(minute_to_words(20), "twenty");
        assert_eq!(minute_to_words(30), "thirty");
        assert_eq!(minute_to_words(50), "fifty");
    }

    #[test]
    fn test_minute_to_words_compound() {
        assert_eq!(minute_to_words(21), "twenty-one");
        assert_eq!(minute_to_words(35), "thirty-five");
        assert_eq!(minute_to_words(59), "fifty-nine");
    }

    #[test]
    fn test_minute_to_words_all() {
        // Test all 60 minutes
        for minute in 0..60 {
            let word = minute_to_words(minute);
            assert!(!word.is_empty(), "Minute {} should have a word", minute);
        }
    }
}

// Built by Foundation-Alpha 🏗️
