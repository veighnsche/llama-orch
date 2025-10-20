# Created by: TEAM-152
# Date: 2025-10-20
# Purpose: BDD tests for queen-rbee lifecycle management

Feature: Queen Lifecycle Management
  As rbee-keeper
  I need to ensure queen-rbee is running
  So I can submit jobs to it

  Scenario: Queen is already running
    Given queen-rbee is running on port 8500
    When I ensure queen is running
    Then it should return immediately without starting a new process
    And I should not see "waking queen" message

  Scenario: Queen is not running (auto-start)
    Given queen-rbee is not running
    When I ensure queen is running
    Then it should start queen-rbee process
    And I should see "queen is asleep, waking queen"
    And it should poll health until ready
    And I should see "queen is awake and healthy"
    And queen should be running on port 8500

  Scenario: Queen startup with health check
    Given queen-rbee is not running
    When I ensure queen is running
    Then queen should respond to health checks within 30 seconds
    And the health endpoint should return status "ok"

  # TEAM-153: Cleanup scenarios
  Scenario: Cleanup when we started the queen
    Given queen-rbee is not running
    When I ensure queen is running
    Then the handle should indicate we started the queen
    When I shutdown the queen handle
    Then it should send shutdown request to queen
    And queen should stop running
    And I should see "Shutting down queen" message

  Scenario: No cleanup when queen was already running
    Given queen-rbee is running on port 8500
    When I ensure queen is running
    Then the handle should indicate queen was already running
    When I shutdown the queen handle
    Then it should NOT send shutdown request
    And queen should still be running
    And I should see "Queen was already running, not shutting down" message

  Scenario: Graceful shutdown via HTTP
    Given queen-rbee is not running
    When I ensure queen is running
    And the queen handle indicates we started it
    When I shutdown the queen handle
    Then it should attempt HTTP POST to /shutdown first
    And if HTTP succeeds, it should not send SIGTERM
    And I should see "Queen shutdown via HTTP" message

  Scenario: Fallback to SIGTERM when HTTP fails
    Given queen-rbee is not running
    When I ensure queen is running
    And the queen handle indicates we started it
    And the shutdown endpoint is not available
    When I shutdown the queen handle
    Then it should attempt HTTP POST to /shutdown first
    And when HTTP fails, it should send SIGTERM to the PID
    And I should see "HTTP shutdown failed, sending SIGTERM" message
