# Full Stack Integration Tests
# Created by: TEAM-106
# Priority: P0 - Critical for production readiness
# Purpose: Test complete queen → hive → worker flows with real services

Feature: Full Stack Integration
  As an integration tester
  I want to verify complete system flows
  So that all components work together correctly

  Background:
    Given the integration test environment is running
    And queen-rbee is healthy at 'http://localhost:8080'
    And rbee-hive is healthy at 'http://localhost:9200'
    And mock-worker is healthy at 'http://localhost:8001'

  @integration @p0 @full-stack
  Scenario: FULL-001 - Complete inference flow with all components
    Given no active inference requests
    When client sends inference request to queen-rbee
    Then queen-rbee accepts the request
    And queen-rbee routes to rbee-hive at 'http://localhost:9200'
    And rbee-hive selects available worker
    And worker processes the inference request
    And tokens stream back via SSE
    And client receives all tokens
    And worker returns to idle state
    And request completes in under 10 seconds

  @integration @p0 @full-stack
  Scenario: FULL-002 - Authentication flow end-to-end
    Given queen-rbee requires authentication
    And client has valid JWT token
    When client sends authenticated request to queen-rbee
    Then queen-rbee validates JWT
    And JWT claims are extracted
    And request proceeds to rbee-hive with auth context
    And rbee-hive validates auth context
    And request proceeds to worker
    And worker processes request
    And response includes auth correlation ID

  @integration @p0 @full-stack
  Scenario: FULL-003 - Worker registration and discovery
    Given rbee-hive is running
    And no workers are registered
    When mock-worker starts and sends ready callback
    Then rbee-hive registers the worker
    And worker appears in registry
    And queen-rbee can discover the worker
    And worker is available for inference
    And worker health check passes

  @integration @p0 @full-stack
  Scenario: FULL-004 - Cascading shutdown propagation
    Given queen-rbee is running
    And rbee-hive is running with 1 worker
    And worker is in idle state
    When queen-rbee receives SIGTERM
    Then queen-rbee signals rbee-hive to shutdown
    And rbee-hive signals worker to shutdown
    And worker completes gracefully
    And rbee-hive exits cleanly
    And queen-rbee exits cleanly
    And all processes exit within 30 seconds

  @integration @p0 @full-stack
  Scenario: FULL-005 - Failure recovery with worker crash
    Given worker is processing inference request
    And request ID is "req-test-001"
    When worker crashes unexpectedly
    Then queen-rbee detects worker failure within 5 seconds
    And queen-rbee marks worker as unavailable
    And client receives error response
    And error includes retry information
    And worker is removed from registry

  @integration @p1 @full-stack
  Scenario: FULL-006 - Concurrent request handling
    Given 3 workers are registered
    And all workers are idle
    When 10 clients send requests simultaneously
    Then all requests are queued
    And requests are distributed across workers
    And no request is lost
    And all requests complete successfully
    And no race conditions occur in registry

  @integration @p1 @full-stack
  Scenario: FULL-007 - Model provisioning flow
    Given rbee-hive is running
    And model "tinyllama-q4" is not in catalog
    When client requests inference with "tinyllama-q4"
    Then rbee-hive checks model catalog
    And model is not found
    And rbee-hive initiates model download
    And download progress is tracked
    And model is registered in catalog
    And worker is spawned with model
    And inference proceeds

  @integration @p1 @full-stack
  Scenario: FULL-008 - Health check propagation
    Given all services are running
    When queen-rbee health check is called
    Then queen-rbee returns healthy status
    And queen-rbee checks rbee-hive health
    And rbee-hive returns healthy status
    And rbee-hive checks worker health
    And worker returns healthy status
    And overall system health is reported

  @integration @p1 @full-stack
  Scenario: FULL-009 - Error propagation from worker to client
    Given worker is available
    When client sends invalid inference request
    Then worker validates request
    And worker returns validation error
    And rbee-hive propagates error
    And queen-rbee propagates error
    And client receives detailed error response
    And error includes request ID for tracing

  @integration @p2 @full-stack
  Scenario: FULL-010 - Metrics collection across components
    Given metrics collection is enabled
    When inference request is processed
    Then queen-rbee emits request metrics
    And rbee-hive emits routing metrics
    And worker emits inference metrics
    And all metrics include correlation ID
    And metrics can be aggregated by request ID
