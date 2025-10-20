# Created by: TEAM-151 (2025-10-20)
# Happy flow: "bee keeper first tests if queen is running? by calling the health."

Feature: Queen Health Check
  As rbee-keeper
  I need to check if queen-rbee is running
  So I can decide whether to start it or proceed with commands

  Background:
    Given the queen URL is "http://localhost:8500"

  Scenario: Queen is not running (connection refused)
    Given queen-rbee is not running
    When I check if queen is healthy
    Then the health check should return false
    And I should see "queen-rbee is not running"

  Scenario: Queen is running and healthy
    Given queen-rbee is running on port 8500
    When I check if queen is healthy
    Then the health check should return true
    And I should see "queen-rbee is running and healthy"

  Scenario: Queen health check with custom port
    Given queen-rbee is running on port 8501
    And the queen URL is "http://localhost:8501"
    When I check if queen is healthy
    Then the health check should return true

  Scenario: Queen health check timeout
    Given queen-rbee is not responding within 500ms
    When I check if queen is healthy
    Then the health check should timeout
    And I should see connection error
