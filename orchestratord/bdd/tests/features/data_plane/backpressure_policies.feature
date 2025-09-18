Feature: Backpressure policy error codes
  # Ensure distinct error codes for reject and drop-lru policies

  Scenario: Admission reject code
    Given an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And error envelope code is ADMISSION_REJECT

  Scenario: Drop-LRU code
    Given an OrchQueue API endpoint under load
    When I enqueue a task way beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And error envelope code is QUEUE_FULL_DROP_LRU
