// TEAM-243: SSE channel lifecycle tests for narration-core
// Purpose: Verify SSE channels handle concurrent operations without memory leaks
// Scale: Reasonable for NUC (5-10 concurrent, 100 channels total)
// Historical Context: TEAM-243 implemented Priority 1 critical tests for observability infrastructure

use observability_narration_core::{sse_sink, NarrationFields};

/// Test SSE channel creation
#[tokio::test]
async fn test_sse_channel_creation() {
    let job_id = "job-123";

    sse_sink::create_job_channel(job_id.to_string(), 100);

    // Verify channel was created
    assert!(sse_sink::has_job_channel(job_id));

    // Cleanup
    sse_sink::remove_job_channel(job_id);
    println!("✓ SSE channel created successfully");
}

/// Test SSE channel send and receive
#[tokio::test]
async fn test_sse_channel_send_receive() {
    let job_id = "job-456";

    sse_sink::create_job_channel(job_id.to_string(), 100);

    // Send narration event
    let fields = NarrationFields {
        actor: "test",
        action: "send",
        target: "test".to_string(),
        human: "test message".to_string(),
        job_id: Some(job_id.to_string()),
        ..Default::default()
    };
    sse_sink::send(&fields);

    // Receive message
    let mut receiver = sse_sink::take_job_receiver(job_id).unwrap();
    let event = receiver.recv().await.unwrap();

    assert_eq!(event.human, "test message");
    println!("✓ SSE channel send/receive completed successfully");
}

/// Test SSE channel cleanup after take
#[tokio::test]
async fn test_sse_channel_cleanup_after_take() {
    let job_id = "job-789";

    sse_sink::create_job_channel(job_id.to_string(), 100);

    assert!(sse_sink::has_job_channel(job_id));

    // Take receiver (removes receiver but sender remains)
    let _receiver = sse_sink::take_job_receiver(job_id);

    // Sender still exists (has_job_channel checks sender)
    assert!(sse_sink::has_job_channel(job_id));

    // Full cleanup removes both
    sse_sink::remove_job_channel(job_id);
    assert!(!sse_sink::has_job_channel(job_id));

    println!("✓ SSE channel cleaned up after take");
}

/// Test concurrent channel creation
#[tokio::test]
async fn test_concurrent_channel_creation() {
    let mut handles = vec![];

    // Create 10 channels concurrently
    for i in 0..10 {
        let handle = tokio::spawn(async move {
            let job_id = format!("job-conc-{}", i);
            sse_sink::create_job_channel(job_id.clone(), 100);
            job_id
        });
        handles.push(handle);
    }

    let mut job_ids = vec![];
    for handle in handles {
        job_ids.push(handle.await.unwrap());
    }

    // Verify all channels exist
    for job_id in &job_ids {
        assert!(sse_sink::has_job_channel(job_id));
    }

    // Cleanup
    for job_id in job_ids {
        sse_sink::remove_job_channel(&job_id);
    }

    println!("✓ 10 concurrent SSE channels created successfully");
}

/// Test memory leak prevention with 100 channels
#[tokio::test]
async fn test_memory_leak_prevention_100_channels() {
    // Create 100 channels
    for i in 0..100 {
        let job_id = format!("job-mem-{}", i);
        sse_sink::create_job_channel(job_id.clone(), 100);

        // Send a message
        let fields = NarrationFields {
            actor: "test",
            action: "mem",
            target: "test".to_string(),
            human: format!("message-{}", i),
            job_id: Some(job_id.clone()),
            ..Default::default()
        };
        sse_sink::send(&fields);
    }

    // Take all channels (cleanup)
    for i in 0..100 {
        let job_id = format!("job-mem-{}", i);
        let _receiver = sse_sink::take_job_receiver(&job_id);
        // Full cleanup
        sse_sink::remove_job_channel(&job_id);
    }

    // Verify all channels are cleaned up
    for i in 0..100 {
        let job_id = format!("job-mem-{}", i);
        assert!(!sse_sink::has_job_channel(&job_id));
    }

    println!("✓ 100 channels created and cleaned up (no memory leaks)");
}

/// Test channel isolation (job_id isolation)
#[tokio::test]
async fn test_channel_isolation() {
    // Create 2 channels
    sse_sink::create_job_channel("job-1".to_string(), 100);
    sse_sink::create_job_channel("job-2".to_string(), 100);

    // Send different messages
    let fields1 = NarrationFields {
        actor: "test",
        action: "iso",
        target: "test".to_string(),
        human: "message-1".to_string(),
        job_id: Some("job-1".to_string()),
        ..Default::default()
    };
    sse_sink::send(&fields1);

    let fields2 = NarrationFields {
        actor: "test",
        action: "iso",
        target: "test".to_string(),
        human: "message-2".to_string(),
        job_id: Some("job-2".to_string()),
        ..Default::default()
    };
    sse_sink::send(&fields2);

    // Verify isolation
    let mut receiver1 = sse_sink::take_job_receiver("job-1").unwrap();
    let event1 = receiver1.recv().await.unwrap();
    assert_eq!(event1.human, "message-1");

    let mut receiver2 = sse_sink::take_job_receiver("job-2").unwrap();
    let event2 = receiver2.recv().await.unwrap();
    assert_eq!(event2.human, "message-2");

    println!("✓ Channel isolation verified (job_id routing works)");
}

/// Test take_channel with non-existent job
#[tokio::test]
async fn test_take_channel_nonexistent() {
    let result = sse_sink::take_job_receiver("nonexistent-job");
    assert!(result.is_none());

    println!("✓ take_job_receiver returns None for non-existent job");
}

/// Test has_channel with non-existent job
#[tokio::test]
async fn test_has_channel_nonexistent() {
    assert!(!sse_sink::has_job_channel("nonexistent-job"));

    println!("✓ has_job_channel returns false for non-existent job");
}

/// Test rapid channel creation and cleanup
#[tokio::test]
async fn test_rapid_channel_creation_cleanup() {
    // Rapidly create and cleanup channels
    for i in 0..50 {
        let job_id = format!("job-rapid-{}", i);

        sse_sink::create_job_channel(job_id.clone(), 100);
        assert!(sse_sink::has_job_channel(&job_id));

        let fields = NarrationFields {
            actor: "test",
            action: "rapid",
            target: "test".to_string(),
            human: format!("message-{}", i),
            job_id: Some(job_id.clone()),
            ..Default::default()
        };
        sse_sink::send(&fields);

        let _receiver = sse_sink::take_job_receiver(&job_id);
        // Full cleanup
        sse_sink::remove_job_channel(&job_id);
        assert!(!sse_sink::has_job_channel(&job_id));
    }

    println!("✓ 50 rapid create/cleanup cycles completed successfully");
}
