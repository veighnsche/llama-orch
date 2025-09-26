Feature: SSE started fields with backpressure/admission context
  # Traceability: ORCH-3029, ORCH-2007, OC-CTRL-2021

  Scenario: Started fields present while backpressure is occurring
    Given an OrchQueue API endpoint
    And an OrchQueue API endpoint under load
    When I stream task events
    And I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And started includes queue_position and predicted_start_ms
