# TEAM-155: BDD tests for SSE streaming (dual-call pattern)
# Tests the happy flow lines 21-24 from a_human_wrote_this.md

Feature: SSE Streaming (Dual-Call Pattern)
  As a rbee-keeper user
  I want to submit jobs and stream results via SSE
  So that I can see real-time inference output

  Background:
    Given queen-rbee is not running

  Scenario: Submit job and establish SSE connection
    When I run "rbee-keeper test-sse"
    Then queen-rbee should auto-start on port 8500
    And I should see "queen is awake and healthy"
    And I should see "Job created"
    And I should see "SSE URL"
    And I should see "Connecting to SSE stream"
    And I should see "Streaming events"
    And the SSE stream should complete
    And queen-rbee should shutdown cleanly

  Scenario: POST /jobs returns job_id and sse_url
    Given queen-rbee is running on port 8500
    When I POST to "/jobs" with test job data
    Then the response should have status 200
    And the response should contain "job_id"
    And the response should contain "sse_url"
    And the job_id should start with "job-"
    And the sse_url should match "/jobs/{job_id}/stream"

  Scenario: GET /jobs/{job_id}/stream establishes SSE connection
    Given queen-rbee is running on port 8500
    And a job exists with id "job-test-123"
    When I GET "/jobs/job-test-123/stream"
    Then the response should be SSE stream
    And I should receive a "started" event
    And I should receive a "[DONE]" event

  Scenario: SSE stream handles missing job gracefully
    Given queen-rbee is running on port 8500
    When I GET "/jobs/nonexistent-job/stream"
    Then the response should have status 404
    And the response should contain "Job nonexistent-job not found"

  Scenario: Full dual-call pattern flow
    Given queen-rbee is not running
    When I run "rbee-keeper test-sse"
    Then the following should happen in order:
      | Step | Expected Output |
      | 1 | queen is asleep, waking queen |
      | 2 | queen is awake and healthy |
      | 3 | Submitting test job |
      | 4 | Job created: job- |
      | 5 | SSE URL: /jobs/ |
      | 6 | Connecting to SSE stream |
      | 7 | Streaming events |
      | 8 | SSE test complete |
      | 9 | Cleanup complete |
