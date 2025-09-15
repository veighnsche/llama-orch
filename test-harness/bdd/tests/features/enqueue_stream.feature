Feature: Enqueue and Stream completion
  # Traceability: ORCH-2001 (enqueue), ORCH-2002 (SSE stream)
  Scenario: Client enqueues and streams tokens
    Given an OrchQueue API endpoint
    When I enqueue a completion task with valid payload
    Then I receive 202 Accepted with correlation id
    And I stream task events
    And I receive SSE events started, token, end
