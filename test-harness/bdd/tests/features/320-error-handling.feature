# Traceability: TEAM-098 (P0 Error Handling Tests)
# Components: rbee-hive, queen-rbee, llm-worker-rbee
# Created by: TEAM-098
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: Error Handling - No unwrap/expect, Structured Errors
  As a system operator
  I want all errors to be handled gracefully with structured responses
  So that the system never panics and errors are debuggable

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

  # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  # TEAM-098: Error Handling Tests (15 scenarios)
  # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  @p0 @error @code-quality
  Scenario: ERR-001 - No unwrap() in production code paths
    Given rbee-hive production code is analyzed
    When searching for unwrap() calls in non-test code
    Then no unwrap() calls are found in src/ directories
    And all Result types use proper error handling
    And all Option types use proper error handling

  @p0 @error @response-format
  Scenario: ERR-002 - Structured error responses (JSON format)
    Given rbee-hive is running
    When an error occurs during worker spawn
    Then response is JSON with error structure
    And response contains "error_code" field
    And response contains "message" field
    And response contains "details" object

  @p0 @error @correlation
  Scenario: ERR-003 - Error correlation IDs included
    Given rbee-hive is running
    When an error occurs during worker spawn
    Then response includes correlation_id
    And correlation_id is a valid UUID
    And correlation_id is unique per request

  @p0 @error @logging
  Scenario: ERR-004 - Correlation IDs logged for debugging
    Given rbee-hive is running
    When an error occurs during worker spawn
    Then correlation_id is logged in error message
    And correlation_id appears in all related log entries
    And correlation_id can be used to trace request flow

  @p0 @error @degradation
  Scenario: ERR-005 - Graceful degradation (DB unavailable)
    Given rbee-hive is running
    And model catalog database is unavailable
    When client requests worker spawn
    Then rbee-hive returns 503 Service Unavailable
    And response includes structured error
    And rbee-hive continues running (does NOT crash)
    And rbee-hive retries DB connection

  @p0 @error @security
  Scenario: ERR-006 - Safe error messages (no sensitive data)
    Given rbee-hive is running
    When authentication fails
    Then error message does NOT contain password
    And error message does NOT contain API key
    And error message does NOT contain token
    And error message contains safe generic message

  @p0 @error @security
  Scenario: ERR-007 - Error messages don't contain raw tokens
    Given rbee-hive is running
    When token validation fails
    Then error message does NOT contain raw token value
    And error message contains token prefix only (first 8 chars)
    And full token is logged securely (not in response)

  @p0 @error @security
  Scenario: ERR-008 - Error messages don't contain file paths
    Given rbee-hive is running
    When file operation fails
    Then error message does NOT contain absolute file paths
    And error message does NOT contain home directory paths
    And error message contains sanitized path (relative or generic)

  @p0 @error @security
  Scenario: ERR-009 - Error messages don't contain internal IPs
    Given rbee-hive is running
    When network error occurs
    Then error message does NOT contain internal IP addresses
    And error message does NOT contain hostnames
    And error message contains generic network error description

  @p0 @error @recovery
  Scenario: ERR-010 - Error recovery for non-fatal errors
    Given rbee-hive is running with 1 worker
    When worker health check fails once
    Then rbee-hive increments failed_health_checks counter
    And rbee-hive does NOT remove worker immediately
    And rbee-hive retries health check
    And rbee-hive only removes worker after 3 consecutive failures

  @p0 @error @stability
  Scenario: ERR-011 - Panic-free operation under load
    Given rbee-hive is running
    When 100 concurrent requests arrive
    And 50 requests have invalid data
    Then rbee-hive processes all requests without panic
    And invalid requests return structured errors
    And valid requests complete successfully
    And rbee-hive remains stable

  @p0 @error @response-format
  Scenario: ERR-012 - Error response includes error_code field
    Given rbee-hive is running
    When worker spawn fails due to insufficient resources
    Then response includes error_code "INSUFFICIENT_RESOURCES"
    And error_code is machine-readable string
    And error_code follows UPPER_SNAKE_CASE convention

  @p0 @error @response-format
  Scenario: ERR-013 - Error response includes details object
    Given rbee-hive is running
    When worker spawn fails due to insufficient VRAM
    Then response includes details object
    And details contains "required_vram_mb" field
    And details contains "available_vram_mb" field
    And details contains "model_ref" field
    And details object is JSON serializable

  @p0 @error @http-status
  Scenario: ERR-014 - HTTP status codes match error types
    Given rbee-hive is running
    When various errors occur
    Then authentication errors return 401 Unauthorized
    And authorization errors return 403 Forbidden
    And not found errors return 404 Not Found
    And validation errors return 400 Bad Request
    And resource exhaustion returns 503 Service Unavailable
    And internal errors return 500 Internal Server Error

  @p0 @error @audit
  Scenario: ERR-015 - Error audit logging
    Given rbee-hive is running
    When an error occurs during worker spawn
    Then error is logged with severity ERROR
    And log entry includes correlation_id
    And log entry includes error_code
    And log entry includes timestamp
    And log entry includes component name
    And log entry includes stack trace (if available)
