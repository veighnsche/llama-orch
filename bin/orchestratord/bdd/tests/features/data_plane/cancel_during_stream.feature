Feature: Cancel during stream
  # Ensure no tokens are emitted after a cancel request is issued mid-stream

  Scenario: Cancel prevents further tokens
    Given an OrchQueue API endpoint
    And I enqueue a completion task with valid payload
    Then I receive 202 Accepted with correlation id
    When I stream task events while canceling mid-stream
    Then no further token events are emitted
