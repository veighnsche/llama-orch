# TEAM-307: Updated for n!() macro (levels work with n!() macro)
Feature: Narration Levels
  As a developer
  I want multiple logging levels
  So that I can control narration verbosity

  Background:
    Given the narration capture adapter is installed
    And the capture buffer is empty

  Scenario: INFO level (default)
    When I narrate at INFO level with message "Normal operation"
    Then the captured narration should have 1 event
    And the narration level should be "INFO"

  Scenario: WARN level
    When I narrate at WARN level with message "Capacity low"
    Then the captured narration should have 1 event
    And the narration level should be "WARN"

  Scenario: ERROR level
    When I narrate at ERROR level with message "Operation failed"
    Then the captured narration should have 1 event
    And the narration level should be "ERROR"

  Scenario: FATAL level
    When I narrate at FATAL level with message "Critical failure"
    Then the captured narration should have 1 event
    And the narration level should be "FATAL"

  Scenario: MUTE level produces no output
    When I narrate at MUTE level with message "This should not appear"
    Then the captured narration should have 0 events

  Scenario: Multiple levels in sequence
    When I narrate at INFO level with message "Starting"
    And I narrate at WARN level with message "Warning"
    And I narrate at ERROR level with message "Error"
    Then the captured narration should have 3 events
    And event 1 level should be "INFO"
    And event 2 level should be "WARN"
    And event 3 level should be "ERROR"
