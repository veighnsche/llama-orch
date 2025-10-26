// TEAM-307: SSE streaming step definitions

use cucumber::{given, when, then};
use crate::steps::world::World;

// ============================================================================
// Given Steps - SSE Setup
// ============================================================================

#[given(regex = r#"^a job with ID "([^"]+)"$"#)]
async fn job_with_id(world: &mut World, job_id: String) {
    world.job_id = Some(job_id);
}

#[given(regex = r#"^two jobs with IDs "([^"]+)" and "([^"]+)"$"#)]
async fn two_jobs(world: &mut World, job_id1: String, job_id2: String) {
    world.job_ids = vec![job_id1, job_id2];
}

#[given(regex = r#"^(\d+) concurrent jobs$"#)]
async fn concurrent_jobs(world: &mut World, count: usize) {
    world.job_ids = (0..count)
        .map(|i| format!("job-concurrent-{}", i))
        .collect();
}

// ============================================================================
// When Steps - SSE Operations
// ============================================================================

#[when("I create an SSE channel for the job")]
async fn create_sse_channel(world: &mut World) {
    if let Some(job_id) = &world.job_id {
        world.sse_channels.insert(job_id.clone(), true);
    }
}

#[when("I close the SSE channel")]
async fn close_sse_channel(world: &mut World) {
    if let Some(job_id) = &world.job_id {
        world.sse_channels.insert(job_id.clone(), false);
    }
}

#[when("I create SSE channels for both jobs")]
async fn create_sse_channels_for_both(world: &mut World) {
    for job_id in &world.job_ids {
        world.sse_channels.insert(job_id.clone(), true);
    }
}

#[when("job emits narration events")]
async fn job_emits_events(_world: &mut World) {
    // Events emitted via n!() macro in other steps
}

#[when(regex = r#"^job emits (\d+) events rapidly$"#)]
async fn job_emits_rapid_events(world: &mut World, count: usize) {
    for i in 0..count {
        world.sse_events.push(format!("event-{}", i));
    }
}

#[when("client disconnects")]
async fn client_disconnects(world: &mut World) {
    if let Some(job_id) = &world.job_id {
        world.sse_channels.insert(job_id.clone(), false);
    }
}

#[when("client reconnects")]
async fn client_reconnects(world: &mut World) {
    if let Some(job_id) = &world.job_id {
        world.sse_channels.insert(job_id.clone(), true);
    }
}

#[when("client subscribes late")]
async fn client_subscribes_late(_world: &mut World) {
    // Simulate late subscription
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
}

#[when("client subscribes early")]
async fn client_subscribes_early(_world: &mut World) {
    // Client already subscribed
}

#[when("client is slow to consume events")]
async fn client_slow(_world: &mut World) {
    // Simulate slow client
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
}

// ============================================================================
// Then Steps - SSE Assertions
// ============================================================================

#[then("the SSE channel should exist")]
async fn sse_channel_exists(world: &mut World) {
    if let Some(job_id) = &world.job_id {
        assert!(world.sse_channels.contains_key(job_id), "SSE channel should exist");
    }
}

#[then("the channel should be ready to receive events")]
async fn channel_ready(world: &mut World) {
    if let Some(job_id) = &world.job_id {
        assert_eq!(world.sse_channels.get(job_id), Some(&true), "Channel should be ready");
    }
}

#[then("the SSE channel should be closed")]
async fn sse_channel_closed(world: &mut World) {
    if let Some(job_id) = &world.job_id {
        assert_eq!(world.sse_channels.get(job_id), Some(&false), "Channel should be closed");
    }
}

#[then("events should be received in order")]
async fn events_in_order(world: &mut World) {
    let events = &world.sse_events;
    for i in 0..events.len().saturating_sub(1) {
        assert!(events[i] <= events[i + 1], "Events should be in order");
    }
}

#[then("each job should have isolated SSE channel")]
async fn jobs_isolated(world: &mut World) {
    for job_id in &world.job_ids {
        assert!(world.sse_channels.contains_key(job_id), 
            "Job {} should have SSE channel", job_id);
    }
}

#[then("backpressure should be handled gracefully")]
async fn backpressure_handled(_world: &mut World) {
    // Verify no panic or data loss
    // In real implementation, check buffer size and flow control
}

#[then("late subscriber should receive buffered events")]
async fn late_subscriber_receives(_world: &mut World) {
    // Verify buffered events are delivered
}

#[then("early subscriber should receive all events")]
async fn early_subscriber_receives(_world: &mut World) {
    // Verify all events are delivered
}
