# TEAM-307: Job Lifecycle Feature
# Tests complete job lifecycle from creation to completion

Feature: Job Lifecycle
  As a job execution system
  I want jobs to have a complete lifecycle with narration
  So that operations are tracked from start to finish

  Background:
    Given the narration capture adapter is installed
    And the capture buffer is empty

  # ============================================================================
  # Job Creation
  # ============================================================================

  Scenario: Create a new job
    When I create a new job
    Then the job should have a unique job_id
    And the job should be in "Queued" state
    And the job_id should match pattern "job-[uuid]"

  Scenario: Create job with custom ID
    When I create a job with ID "custom-job-123"
    Then the job should have job_id "custom-job-123"
    And the job should be in "Queued" state

  # ============================================================================
  # Job Execution
  # ============================================================================

  Scenario: Execute a job successfully
    Given a job with ID "job-exec-success"
    When I execute the job
    And the job emits narration during execution
    Then the job should transition to "Running" state
    And narration should be captured with job_id
    And the job should transition to "Completed" state

  Scenario: Job execution with context
    Given a job with ID "job-exec-context"
    And a narration context for the job
    When I execute the job in context
    And the job emits narration with n!("step", "Processing")
    Then all narration should have job_id "job-exec-context"
    And the job should complete successfully

  # ============================================================================
  # Job Streaming
  # ============================================================================

  Scenario: Stream job results via SSE
    Given a job with ID "job-stream-123"
    And an SSE channel for the job
    And an SSE client subscribed
    When I execute the job
    And the job emits narration events:
      | action | message    |
      | start  | Starting   |
      | work   | Working    |
      | done   | Completed  |
    Then the SSE client should receive all 3 events
    And the final event should be "[DONE]"

  # ============================================================================
  # Job Completion
  # ============================================================================

  Scenario: Job completes successfully
    Given a job with ID "job-complete-success"
    When I execute the job
    And the job finishes without errors
    Then the job state should be "Completed"
    And the SSE stream should send "[DONE]"
    And the job should be cleanable

  Scenario: Job completes with result data
    Given a job with ID "job-result-data"
    When I execute the job
    And the job produces result data
    Then the result should be accessible
    And the result should be included in completion narration

  # ============================================================================
  # Job Failure
  # ============================================================================

  Scenario: Job fails with error
    Given a job with ID "job-fail-error"
    When I execute the job
    And the job encounters an error "Database connection failed"
    Then the job state should be "Failed"
    And the error message should be "Database connection failed"
    And the SSE stream should send "[ERROR] Database connection failed"

  Scenario: Job failure is captured in narration
    Given a job with ID "job-fail-narration"
    When I execute the job
    And the job emits narration before failure
    And the job fails
    Then all narration before failure should be captured
    And the failure should be narrated
    And the SSE stream should include all events

  # ============================================================================
  # Job Timeout
  # ============================================================================

  Scenario: Job times out
    Given a job with ID "job-timeout-123"
    And a timeout of 1 second
    When I execute the job
    And the job runs for 2 seconds
    Then the job should be cancelled due to timeout
    And the job state should be "Failed"
    And the error should mention "timeout"

  # ============================================================================
  # Job Cancellation
  # ============================================================================

  Scenario: Cancel a running job
    Given a job with ID "job-cancel-running"
    And the job is in "Running" state
    When I cancel the job
    Then the job should transition to "Cancelled" state
    And the SSE stream should send "[CANCELLED]"
    And the job should stop executing

  Scenario: Cancel a queued job
    Given a job with ID "job-cancel-queued"
    And the job is in "Queued" state
    When I cancel the job
    Then the job should transition to "Cancelled" state
    And the job should never start executing

  Scenario: Cannot cancel completed job
    Given a job with ID "job-cancel-complete"
    And the job is in "Completed" state
    When I attempt to cancel the job
    Then the cancellation should be rejected
    And the job should remain in "Completed" state

  # ============================================================================
  # Job Cleanup
  # ============================================================================

  Scenario: Job resources cleaned up after completion
    Given a job with ID "job-cleanup-complete"
    When the job completes
    Then the SSE channel should be cleaned up
    And the job context should be cleared
    And resources should be released

  Scenario: Job cleanup after failure
    Given a job with ID "job-cleanup-fail"
    When the job fails
    Then the SSE channel should be cleaned up
    And error state should be preserved
    And resources should be released

  # ============================================================================
  # Multiple Jobs
  # ============================================================================

  Scenario: Multiple jobs execute concurrently
    Given 5 jobs with IDs:
      | job_id      |
      | job-multi-1 |
      | job-multi-2 |
      | job-multi-3 |
      | job-multi-4 |
      | job-multi-5 |
    When all jobs execute concurrently
    And each job emits narration
    Then all jobs should complete successfully
    And narration should be isolated per job
    And no cross-contamination should occur
