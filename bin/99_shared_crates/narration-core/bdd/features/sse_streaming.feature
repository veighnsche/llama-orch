# TEAM-307: SSE Streaming Feature
# Tests Server-Sent Events streaming for job-scoped narration

Feature: SSE Streaming
  As a distributed system
  I want narration to stream through SSE channels
  So that clients can receive real-time updates

  Background:
    Given the narration capture adapter is installed
    And the capture buffer is empty

  # ============================================================================
  # SSE Channel Lifecycle
  # ============================================================================

  Scenario: Create SSE channel for job
    Given a job with ID "job-sse-123"
    When I create an SSE channel for the job
    Then the SSE channel should exist
    And the channel should be ready to receive events

  Scenario: Emit narration to SSE channel
    Given a job with ID "job-emit-456"
    And an SSE channel exists for the job
    When I emit narration with job_id "job-emit-456"
    Then the narration should be sent to SSE channel
    And the SSE channel should have 1 event

  Scenario: Receive narration from SSE stream
    Given a job with ID "job-receive-789"
    And an SSE channel exists for the job
    And an SSE client is subscribed to the job
    When I emit narration with job_id "job-receive-789"
    Then the SSE client should receive the narration
    And the received event should match the emitted narration

  # ============================================================================
  # Signal Markers
  # ============================================================================

  Scenario: [DONE] signal sent on job completion
    Given a job with ID "job-done-123"
    And an SSE channel exists for the job
    And an SSE client is subscribed
    When the job completes successfully
    Then the SSE stream should send "[DONE]" signal
    And the stream should close

  Scenario: [ERROR] signal sent on job failure
    Given a job with ID "job-error-456"
    And an SSE channel exists for the job
    And an SSE client is subscribed
    When the job fails with error "Test error"
    Then the SSE stream should send "[ERROR] Test error" signal
    And the stream should close

  Scenario: [CANCELLED] signal sent on job cancellation
    Given a job with ID "job-cancel-789"
    And an SSE channel exists for the job
    And an SSE client is subscribed
    When the job is cancelled
    Then the SSE stream should send "[CANCELLED]" signal
    And the stream should close

  # ============================================================================
  # Event Ordering
  # ============================================================================

  Scenario: Multiple events received in order
    Given a job with ID "job-order-123"
    And an SSE channel exists for the job
    And an SSE client is subscribed
    When I emit narration events in order:
      | action | message |
      | first  | First   |
      | second | Second  |
      | third  | Third   |
    Then the SSE client should receive 3 events
    And events should be in order: "first", "second", "third"

  Scenario: High-frequency events (100 events)
    Given a job with ID "job-freq-456"
    And an SSE channel exists for the job
    And an SSE client is subscribed
    When I emit 100 narration events rapidly
    Then the SSE client should receive all 100 events
    And no events should be lost

  # ============================================================================
  # Job Isolation
  # ============================================================================

  Scenario: Concurrent jobs have isolated SSE channels
    Given two jobs with IDs:
      | job_id      |
      | job-iso-a   |
      | job-iso-b   |
    And SSE channels exist for both jobs
    And SSE clients are subscribed to both jobs
    When I emit narration to job "job-iso-a"
    And I emit narration to job "job-iso-b"
    Then job-iso-a client should receive only job-iso-a narration
    And job-iso-b client should receive only job-iso-b narration
    And no cross-contamination should occur

  # ============================================================================
  # Channel Cleanup
  # ============================================================================

  Scenario: SSE channel cleaned up when client disconnects
    Given a job with ID "job-cleanup-123"
    And an SSE channel exists for the job
    And an SSE client is subscribed
    When the SSE client disconnects
    Then the SSE channel should be cleaned up
    And resources should be released

  Scenario: New SSE channel can be created after cleanup
    Given a job with ID "job-recreate-456"
    And an SSE channel was created and cleaned up
    When I create a new SSE channel for the same job
    Then the new channel should work correctly
    And previous events should not be present

  # ============================================================================
  # Backpressure
  # ============================================================================

  Scenario: Backpressure handling when client is slow
    Given a job with ID "job-backpressure-789"
    And an SSE channel exists for the job
    And a slow SSE client is subscribed (100ms per event)
    When I emit 10 narration events rapidly
    Then all events should be buffered
    And the slow client should eventually receive all events
    And no events should be dropped

  # ============================================================================
  # Late/Early Subscribers
  # ============================================================================

  Scenario: Late subscriber (after events emitted)
    Given a job with ID "job-late-123"
    And an SSE channel exists for the job
    When I emit 3 narration events
    And an SSE client subscribes late
    Then the late client should NOT receive previous events
    And the late client should receive new events

  Scenario: Early subscriber (before events emitted)
    Given a job with ID "job-early-456"
    And an SSE channel exists for the job
    And an SSE client subscribes early
    When I emit 3 narration events
    Then the early client should receive all 3 events

  # ============================================================================
  # Error Handling
  # ============================================================================

  Scenario: Narration without job_id is not sent to SSE
    Given an SSE system is active
    When I emit narration without job_id
    Then the narration should NOT be sent to any SSE channel
    And no error should occur

  Scenario: Narration with non-existent job_id is dropped gracefully
    Given an SSE system is active
    When I emit narration with job_id "non-existent-job"
    Then the narration should be dropped gracefully
    And no error should occur
    And no SSE channel should be created
