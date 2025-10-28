// TEAM-298: Phase 1 - SSE Optional Tests
//! Comprehensive tests for opportunistic SSE delivery

use observability_narration_core::*;
use serial_test::serial;

#[tokio::test]
#[serial(sse_sink)]
async fn test_narration_without_channel_works() {
    // TEAM-298: Narration should work without SSE channel (no panic!)

    // This used to fail before Phase 1 if you forgot to create channel
    n!("test", "This works without channel");

    // Success! Narration goes to stdout, no SSE channel needed
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_narration_with_job_id_but_no_channel() {
    // TEAM-298: Even with job_id set, narration works without channel

    let job_id = "no-channel-job";
    let ctx = NarrationContext::new().with_job_id(job_id);

    with_narration_context(ctx, async {
        // job_id is set, but no channel exists
        n!("test", "This goes to stdout only");

        // Success! No panic, no error
    })
    .await;
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_narration_before_channel_creation() {
    // TEAM-298: Narration can happen BEFORE channel creation
    // This is the key improvement - order doesn't matter anymore!

    let job_id = "early-job";
    let ctx = NarrationContext::new().with_job_id(job_id);

    with_narration_context(ctx, async {
        // 1. Emit narration FIRST (channel doesn't exist yet!)
        n!("early", "This happened before channel");

        // 2. NOW create channel
        sse_sink::create_job_channel(job_id.to_string(), 100);
        let mut rx = sse_sink::take_job_receiver(job_id).unwrap();

        // 3. Future narration goes to SSE
        n!("later", "This goes to SSE");

        // 4. Only second event in channel
        let event = rx.recv().await.expect("Should receive");
        assert_eq!(event.action, "later");
        assert_eq!(event.human, "This goes to SSE");

        // First event went to stdout only (correct behavior!)
    })
    .await;
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_sse_still_works_when_available() {
    // TEAM-298: SSE still works when channel exists (backward compatible)

    let job_id = "sse-job";

    // Create channel first
    sse_sink::create_job_channel(job_id.to_string(), 100);
    let mut rx = sse_sink::take_job_receiver(job_id).unwrap();

    // Set context
    let ctx = NarrationContext::new().with_job_id(job_id);

    with_narration_context(ctx, async {
        n!("test", "This goes to SSE");

        // Should receive event
        let event = rx.recv().await.expect("Should receive");
        assert_eq!(event.action, "test");
        assert_eq!(event.human, "This goes to SSE");
    })
    .await;
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_try_send_returns_false_without_channel() {
    // TEAM-298: try_send() returns false when no channel exists

    let fields = NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        job_id: Some("no-channel".to_string()),
        ..Default::default()
    };

    let sent = sse_sink::try_send(&fields);
    assert_eq!(sent, false, "Should return false when channel doesn't exist");
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_try_send_returns_false_without_job_id() {
    // TEAM-298: try_send() returns false when no job_id

    let fields = NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        job_id: None, // No job_id
        ..Default::default()
    };

    let sent = sse_sink::try_send(&fields);
    assert_eq!(sent, false, "Should return false when no job_id");
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_try_send_returns_true_when_successful() {
    // TEAM-298: try_send() returns true when event is sent successfully

    let job_id = "success-job";
    sse_sink::create_job_channel(job_id.to_string(), 100);
    let _rx = sse_sink::take_job_receiver(job_id).unwrap();

    let fields = NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        job_id: Some(job_id.to_string()),
        ..Default::default()
    };

    let sent = sse_sink::try_send(&fields);
    assert_eq!(sent, true, "Should return true when event is sent");
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_try_send_returns_false_when_channel_full() {
    // TEAM-298: try_send() returns false when channel is full (backpressure)

    let job_id = "full-channel-job";
    sse_sink::create_job_channel(job_id.to_string(), 1); // Tiny capacity
    let _rx = sse_sink::take_job_receiver(job_id).unwrap();

    let fields = NarrationFields {
        actor: "test",
        action: "test",
        target: "test".to_string(),
        human: "Test message".to_string(),
        job_id: Some(job_id.to_string()),
        ..Default::default()
    };

    // Fill the channel
    let sent1 = sse_sink::try_send(&fields);
    assert_eq!(sent1, true, "First send should succeed");

    // Second send should fail (channel full)
    let sent2 = sse_sink::try_send(&fields);
    assert_eq!(sent2, false, "Second send should fail (channel full)");
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_multiple_narrations_before_channel() {
    // TEAM-298: Multiple narrations before channel creation all go to stdout

    let job_id = "multi-early-job";
    let ctx = NarrationContext::new().with_job_id(job_id);

    with_narration_context(ctx, async {
        // Multiple narrations before channel creation
        n!("early1", "First early narration");
        n!("early2", "Second early narration");
        n!("early3", "Third early narration");

        // All go to stdout, none fail

        // Now create channel
        sse_sink::create_job_channel(job_id.to_string(), 100);
        let mut rx = sse_sink::take_job_receiver(job_id).unwrap();

        // Future narrations go to SSE
        n!("later", "After channel");

        // Only the last one in channel
        let event = rx.recv().await.expect("Should receive");
        assert_eq!(event.action, "later");
    })
    .await;
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_narration_with_all_three_modes_no_channel() {
    // TEAM-298: All three narration modes work without channel

    n!("test",
        human: "Human message",
        cute: "🐝 Cute message",
        story: "Story message"
    );

    // Success! No panic, works without channel
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_narration_modes_with_channel() {
    // TEAM-298: All three modes work with channel too

    let job_id = "modes-job";
    sse_sink::create_job_channel(job_id.to_string(), 100);
    let mut rx = sse_sink::take_job_receiver(job_id).unwrap();

    let ctx = NarrationContext::new().with_job_id(job_id);

    with_narration_context(ctx, async {
        n!("test",
            human: "Human message",
            cute: "🐝 Cute message",
            story: "Story message"
        );

        let event = rx.recv().await.expect("Should receive");
        assert_eq!(event.human, "Human message");
        assert_eq!(event.cute, Some("🐝 Cute message".to_string()));
        assert_eq!(event.story, Some("Story message".to_string()));
    })
    .await;
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_channel_creation_after_many_narrations() {
    // TEAM-298: Channel can be created at any point

    let job_id = "late-channel-job";
    let ctx = NarrationContext::new().with_job_id(job_id);

    with_narration_context(ctx, async {
        // Lots of narration before channel
        for i in 0..10 {
            n!("early", "Early narration {}", i);
        }

        // Now create channel (late!)
        sse_sink::create_job_channel(job_id.to_string(), 100);
        let mut rx = sse_sink::take_job_receiver(job_id).unwrap();

        // New narrations go to SSE
        for i in 0..5 {
            n!("later", "Later narration {}", i);
        }

        // Should receive 5 events (only post-channel)
        let mut count = 0;
        while let Ok(Some(_)) =
            tokio::time::timeout(std::time::Duration::from_millis(100), rx.recv()).await
        {
            count += 1;
        }

        assert_eq!(count, 5, "Should receive only post-channel narrations");
    })
    .await;
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_backward_compatibility_old_pattern_still_works() {
    // TEAM-298: Old pattern (channel first) still works

    let job_id = "old-pattern-job";

    // Old pattern: create channel first
    sse_sink::create_job_channel(job_id.to_string(), 100);
    let mut rx = sse_sink::take_job_receiver(job_id).unwrap();

    let ctx = NarrationContext::new().with_job_id(job_id);

    with_narration_context(ctx, async {
        // Then narrate (old pattern)
        n!("test", "Old pattern still works");

        let event = rx.recv().await.expect("Should receive");
        assert_eq!(event.human, "Old pattern still works");
    })
    .await;
}

#[tokio::test]
#[serial(sse_sink)]
async fn test_narration_after_channel_removal() {
    // TEAM-298: Narration after channel removal goes to stdout only

    let job_id = "removed-channel-job";
    let ctx = NarrationContext::new().with_job_id(job_id);

    with_narration_context(ctx, async {
        // Create channel
        sse_sink::create_job_channel(job_id.to_string(), 100);
        let mut rx = sse_sink::take_job_receiver(job_id).unwrap();

        // Send one event
        n!("before", "Before removal");
        let event = rx.recv().await.expect("Should receive");
        assert_eq!(event.action, "before");

        // Remove channel
        sse_sink::remove_job_channel(job_id);

        // Narration after removal still works (goes to stdout)
        n!("after", "After removal");

        // Success! No panic
    })
    .await;
}
