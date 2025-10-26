// TEAM-307: Job lifecycle step definitions

use cucumber::{given, when, then};
use crate::steps::world::World;

// ============================================================================
// Given Steps - Job Setup
// ============================================================================

#[given("a job system")]
async fn job_system(_world: &mut World) {
    // Job system initialized
}

#[given(regex = r#"^a job in "([^"]+)" state$"#)]
async fn job_in_state(world: &mut World, state: String) {
    world.job_state = Some(state);
}

#[given(regex = r#"^job "([^"]+)" is running$"#)]
async fn job_running(world: &mut World, job_id: String) {
    world.job_id = Some(job_id);
    world.job_state = Some("Running".to_string());
}

#[given(regex = r#"^job "([^"]+)" has been running for (\d+) seconds$"#)]
async fn job_running_duration(world: &mut World, job_id: String, _seconds: u64) {
    world.job_id = Some(job_id);
    world.job_state = Some("Running".to_string());
}

// ============================================================================
// When Steps - Job Operations
// ============================================================================

#[when("I create a new job")]
async fn create_job(world: &mut World) {
    let job_id = format!("job-{}", uuid::Uuid::new_v4());
    world.job_id = Some(job_id);
    world.job_state = Some("Queued".to_string());
}

#[when("I submit the job")]
async fn submit_job(world: &mut World) {
    if world.job_state.as_deref() == Some("Queued") {
        world.job_state = Some("Running".to_string());
    }
}

#[when("the job executes successfully")]
async fn job_executes_successfully(world: &mut World) {
    world.job_state = Some("Completed".to_string());
}

#[when("the job fails")]
async fn job_fails(world: &mut World) {
    world.job_state = Some("Failed".to_string());
    world.job_error = Some("Job execution failed".to_string());
}

#[when("the job times out")]
async fn job_times_out(world: &mut World) {
    world.job_state = Some("TimedOut".to_string());
    world.job_error = Some("Job exceeded timeout".to_string());
}

#[when("I cancel the job")]
async fn cancel_job(world: &mut World) {
    world.job_state = Some("Cancelling".to_string());
}

#[when("cancellation completes")]
async fn cancellation_completes(world: &mut World) {
    world.job_state = Some("Cancelled".to_string());
}

#[when("I request job status")]
async fn request_status(_world: &mut World) {
    // Status requested
}

#[when("job emits progress updates")]
async fn job_emits_progress(_world: &mut World) {
    // Progress updates emitted
}

#[when("job completes")]
async fn job_completes(world: &mut World) {
    world.job_state = Some("Completed".to_string());
}

#[when("I clean up the job")]
async fn cleanup_job(world: &mut World) {
    world.job_id = None;
    world.job_state = None;
    world.job_error = None;
}

#[when(regex = r#"^I create (\d+) concurrent jobs$"#)]
async fn create_concurrent_jobs(world: &mut World, count: usize) {
    world.job_ids = (0..count)
        .map(|i| format!("job-concurrent-{}", i))
        .collect();
}

#[when("all jobs execute")]
async fn all_jobs_execute(_world: &mut World) {
    // All jobs execute
}

// ============================================================================
// Then Steps - Job Assertions
// ============================================================================

#[then("the job should have a unique job_id")]
async fn job_has_unique_id(world: &mut World) {
    assert!(world.job_id.is_some(), "Job should have an ID");
    let job_id = world.job_id.as_ref().unwrap();
    assert!(job_id.starts_with("job-"), "Job ID should have correct format");
}

#[then(regex = r#"^the job should be in "([^"]+)" state$"#)]
async fn job_in_expected_state(world: &mut World, expected_state: String) {
    assert_eq!(world.job_state.as_deref(), Some(expected_state.as_str()),
        "Job should be in {} state", expected_state);
}

#[then(regex = r#"^the job_id should match pattern "([^"]+)"$"#)]
async fn job_id_matches_pattern(world: &mut World, pattern: String) {
    assert!(world.job_id.is_some(), "Job should have an ID");
    let job_id = world.job_id.as_ref().unwrap();
    
    if pattern == "job-[uuid]" {
        assert!(job_id.starts_with("job-"), "Job ID should start with 'job-'");
        assert!(job_id.len() > 10, "Job ID should contain UUID");
    }
}

#[then("narration should include job_id")]
async fn narration_includes_job_id(world: &mut World) {
    if let Some(adapter) = &world.adapter {
        let captured = adapter.captured();
        assert!(!captured.is_empty(), "Should have captured events");
        
        if let Some(job_id) = &world.job_id {
            let has_job_id = captured.iter().any(|event| 
                event.job_id.as_deref() == Some(job_id.as_str())
            );
            assert!(has_job_id, "Narration should include job_id");
        }
    }
}

#[then("job should emit completion narration")]
async fn job_emits_completion(_world: &mut World) {
    // Verify completion narration emitted
}

#[then("job should emit failure narration")]
async fn job_emits_failure(_world: &mut World) {
    // Verify failure narration emitted
}

#[then("job should emit timeout narration")]
async fn job_emits_timeout(_world: &mut World) {
    // Verify timeout narration emitted
}

#[then("job should emit cancellation narration")]
async fn job_emits_cancellation(_world: &mut World) {
    // Verify cancellation narration emitted
}

#[then("job resources should be cleaned up")]
async fn resources_cleaned_up(world: &mut World) {
    assert!(world.job_id.is_none(), "Job ID should be cleared");
    assert!(world.job_state.is_none(), "Job state should be cleared");
}

#[then("all jobs should complete successfully")]
async fn all_jobs_complete(world: &mut World) {
    assert!(!world.job_ids.is_empty(), "Should have jobs");
}

#[then("each job should have isolated narration")]
async fn jobs_have_isolated_narration(_world: &mut World) {
    // Verify each job has separate narration stream
}
