# TEAM-307: Context Propagation Feature
# Tests automatic context injection and propagation across async boundaries

Feature: Context Propagation
  As a distributed system
  I want context (job_id, correlation_id, actor) to propagate automatically
  So that narration is properly scoped without manual injection

  Background:
    Given the narration capture adapter is installed
    And the capture buffer is empty

  # ============================================================================
  # Basic Context Injection
  # ============================================================================

  Scenario: job_id is automatically injected from context
    Given a narration context with job_id "job-auto-123"
    When I emit narration with n!("test", "Message") in context
    Then the captured narration should have 1 event
    And event 1 should have job_id "job-auto-123"
    And event 1 should have action "test"

  Scenario: correlation_id is automatically injected from context
    Given a narration context with correlation_id "corr-xyz-789"
    When I emit narration with n!("test", "Message") in context
    Then the captured narration should have 1 event
    And event 1 should have correlation_id "corr-xyz-789"

  Scenario: actor is automatically injected from context
    Given a narration context with actor "test-actor"
    When I emit narration with n!("test", "Message") in context
    Then the captured narration should have 1 event
    And event 1 should have actor "test-actor"

  Scenario: all context fields injected together
    Given a narration context with:
      | field          | value        |
      | job_id         | job-123      |
      | correlation_id | corr-456     |
      | actor          | full-actor   |
    When I emit narration with n!("test", "Message") in context
    Then the captured narration should have 1 event
    And event 1 should have job_id "job-123"
    And event 1 should have correlation_id "corr-456"
    And event 1 should have actor "full-actor"

  # ============================================================================
  # Context Propagation
  # ============================================================================

  Scenario: context works within same task
    Given a narration context with job_id "job-same-task"
    When I emit multiple narrations in same context:
      | action  | message   |
      | first   | First     |
      | second  | Second    |
      | third   | Third     |
    Then the captured narration should have 3 events
    And all events should have job_id "job-same-task"

  Scenario: context survives await points
    Given a narration context with job_id "job-await-test"
    When I emit narration before await
    And I await for 100 milliseconds
    And I emit narration after await
    Then the captured narration should have 2 events
    And all events should have job_id "job-await-test"

  Scenario: context can be manually propagated to spawned tasks
    Given a narration context with job_id "job-spawn-manual"
    When I manually propagate context to spawned task
    And spawned task emits narration
    Then the captured narration should have 1 event
    And event 1 should have job_id "job-spawn-manual"

  Scenario: context is NOT inherited by tokio::spawn (by design)
    Given a narration context with job_id "job-no-inherit"
    When I spawn task without manual propagation
    And spawned task emits narration
    Then the captured narration should have 1 event
    And event 1 should NOT have job_id

  # ============================================================================
  # Context Isolation
  # ============================================================================

  Scenario: contexts are isolated between concurrent tasks
    Given two concurrent tasks with different contexts:
      | task | job_id    |
      | A    | job-a-123 |
      | B    | job-b-456 |
    When both tasks emit narration concurrently
    Then the captured narration should have 2 events
    And event with action "task_a" should have job_id "job-a-123"
    And event with action "task_b" should have job_id "job-b-456"

  Scenario: nested contexts (inner overrides outer)
    Given an outer context with job_id "job-outer"
    And an inner context with job_id "job-inner"
    When I emit narration in outer context
    And I emit narration in inner context
    And I emit narration in outer context again
    Then the captured narration should have 3 events
    And event 1 should have job_id "job-outer"
    And event 2 should have job_id "job-inner"
    And event 3 should have job_id "job-outer"

  # ============================================================================
  # Context with Async Primitives
  # ============================================================================

  Scenario: context works with tokio::select!
    Given a narration context with job_id "job-select"
    When I use tokio::select! with context
    And selected branch emits narration
    Then the captured narration should have 1 event
    And event 1 should have job_id "job-select"

  Scenario: context works with tokio::timeout
    Given a narration context with job_id "job-timeout"
    When I use tokio::timeout with context
    And operation emits narration before timeout
    Then the captured narration should have 1 event
    And event 1 should have job_id "job-timeout"

  Scenario: context works across channel boundaries
    Given a narration context with job_id "job-channel"
    When I emit narration before channel send
    And I send message through channel
    And I emit narration after channel receive
    Then the captured narration should have 2 events
    And all events should have job_id "job-channel"

  Scenario: context works with futures::join_all
    Given a narration context with job_id "job-join-all"
    When I use futures::join_all with 3 futures
    And each future emits narration
    Then the captured narration should have 3 events
    And all events should have job_id "job-join-all"

  # ============================================================================
  # Deep Nesting
  # ============================================================================

  Scenario: context works with deep nesting (5 levels)
    Given a narration context with job_id "job-deep-5"
    When I create 5 levels of nested async calls
    And each level emits narration
    Then the captured narration should have 5 events
    And all events should have job_id "job-deep-5"

  # ============================================================================
  # Context Without Fields
  # ============================================================================

  Scenario: narration without context works normally
    When I emit narration with n!("test", "No context") without context
    Then the captured narration should have 1 event
    And event 1 should NOT have job_id
    And event 1 should have action "test"

  Scenario: empty context (no fields set)
    Given an empty narration context
    When I emit narration with n!("test", "Empty context") in context
    Then the captured narration should have 1 event
    And event 1 should NOT have job_id
    And event 1 should NOT have correlation_id
