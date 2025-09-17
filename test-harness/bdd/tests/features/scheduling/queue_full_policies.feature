Feature: Queue full policies per engine
  # Traceability: ORCH-3005, OC-ADAPT-5001

  Scenario: llama.cpp reject policy under load
    Given a worker adapter for llama.cpp
    And queue full policy is reject
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms

  Scenario: llama.cpp drop-lru policy under load
    Given a worker adapter for llama.cpp
    And queue full policy is drop-lru
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms
