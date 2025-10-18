# Traceability: RC-P1-DEADLINE (Release Candidate P1 Deadline Propagation)
# Created by: TEAM-099
# Components: queen-rbee, rbee-hive, llm-worker-rbee
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: Deadline Propagation - Timeout Handling Across Components
  As a system operator
  I want request deadlines to propagate through all components
  So that requests timeout gracefully and resources are not wasted

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://0.0.0.0:8080"
    And rbee-hive is running at "http://workstation.home.arpa:9200"

  @p1 @deadline @timeout
  Scenario: DEAD-001 - Propagate timeout queen → hive → worker
    Given rbee-keeper sends inference request with timeout 30s
    When request arrives at queen-rbee
    Then queen-rbee calculates deadline = now + 30s
    And queen-rbee forwards request to rbee-hive with deadline
    And rbee-hive receives deadline from queen-rbee
    And rbee-hive forwards request to worker with deadline
    And worker receives deadline from rbee-hive
    And all components use same deadline

  @p1 @deadline @cancellation
  Scenario: DEAD-002 - Cancel request when deadline exceeded
    Given rbee-keeper sends inference request with timeout 5s
    And worker processing takes 10s
    When deadline is exceeded at 5s
    Then queen-rbee cancels request
    And queen-rbee sends cancellation to rbee-hive
    And rbee-hive sends cancellation to worker
    And worker stops processing
    And response is 408 Request Timeout

  @p1 @deadline @inheritance
  Scenario: DEAD-003 - Deadline inheritance (child inherits parent)
    Given queen-rbee receives request with deadline T1
    When queen-rbee spawns worker for this request
    Then worker inherits deadline T1
    And worker does NOT get new deadline
    And worker respects parent deadline T1

  @p1 @deadline @header
  Scenario: DEAD-004 - X-Request-Deadline header propagation
    Given rbee-keeper sends request with timeout 30s
    When queen-rbee forwards to rbee-hive
    Then request includes header "X-Request-Deadline: <ISO8601 timestamp>"
    When rbee-hive forwards to worker
    Then request includes same "X-Request-Deadline" header
    And deadline timestamp is unchanged

  @p1 @deadline @response
  Scenario: DEAD-005 - 408 Request Timeout response format
    Given rbee-keeper sends inference request with timeout 2s
    And worker processing takes 5s
    When deadline is exceeded
    Then response status is 408 Request Timeout
    And response Content-Type is "application/json"
    And response body contains error_code "REQUEST_TIMEOUT"
    And response body contains message "Request exceeded deadline"
    And response body includes original deadline timestamp

  @p1 @deadline @worker-behavior
  Scenario: DEAD-006 - Worker stops processing on timeout
    Given worker is processing inference request
    And request has deadline in 5s
    When deadline is exceeded
    Then worker stops token generation
    And worker releases GPU resources
    And worker marks slot as available
    And worker logs "request timeout" event

  @p1 @deadline @security
  Scenario: DEAD-007 - Deadline cannot be extended
    Given rbee-keeper sends request with timeout 10s
    When malicious client sends "X-Request-Deadline" header with future time
    Then queen-rbee rejects extended deadline
    And queen-rbee uses original timeout 10s
    And queen-rbee logs warning "deadline extension attempt"

  @p1 @deadline @default
  Scenario: DEAD-008 - Default deadline (30s) when not specified
    Given rbee-keeper sends inference request without timeout
    When request arrives at queen-rbee
    Then queen-rbee sets default deadline = now + 30s
    And queen-rbee propagates deadline to rbee-hive
    And rbee-hive propagates deadline to worker
    And request times out after 30s if not complete
