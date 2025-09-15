Feature: Queue full policies per engine
  # Traceability: ORCH-3005; OC-ADAPT-5001..5070

  # llama.cpp
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

  Scenario: llama.cpp shed-low-priority policy under load
    Given a worker adapter for llama.cpp
    And queue full policy is shed-low-priority
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms

  # vLLM
  Scenario: vLLM reject policy under load
    Given a worker adapter for vllm
    And queue full policy is reject
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms

  Scenario: vLLM drop-lru policy under load
    Given a worker adapter for vllm
    And queue full policy is drop-lru
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms

  Scenario: vLLM shed-low-priority policy under load
    Given a worker adapter for vllm
    And queue full policy is shed-low-priority
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms

  # TGI
  Scenario: TGI reject policy under load
    Given a worker adapter for tgi
    And queue full policy is reject
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms

  Scenario: TGI drop-lru policy under load
    Given a worker adapter for tgi
    And queue full policy is drop-lru
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms

  Scenario: TGI shed-low-priority policy under load
    Given a worker adapter for tgi
    And queue full policy is shed-low-priority
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms

  # Triton
  Scenario: Triton reject policy under load
    Given a worker adapter for triton
    And queue full policy is reject
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms

  Scenario: Triton drop-lru policy under load
    Given a worker adapter for triton
    And queue full policy is drop-lru
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms

  Scenario: Triton shed-low-priority policy under load
    Given a worker adapter for triton
    And queue full policy is shed-low-priority
    And an OrchQueue API endpoint under load
    When I enqueue a task beyond capacity
    Then I receive 429 with headers Retry-After and X-Backoff-Ms and correlation id
    And the error body includes policy_label retriable and retry_after_ms
