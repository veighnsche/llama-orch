Feature: Backpressure 429 handling
  # Traceability: ORCH-2007 (backpressure), ORCH-2006 (ErrorEnvelope)
  Scenario: Queue saturation returns advisory 429
    Given an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms
