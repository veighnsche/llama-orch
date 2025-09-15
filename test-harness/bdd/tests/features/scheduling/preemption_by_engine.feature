Feature: Preemption semantics by engine
  # Traceability: ORCH-3085..3087; OC-ADAPT-5001..5070

  # llama.cpp
  Scenario: Soft preemption under overload (llama.cpp)
    Given a worker adapter for llama.cpp
    And soft preemption is enabled
    And under persistent overload
    Then lower priority items are preempted first
    And preemptions_total and resumptions_total metrics are exported

  Scenario: Hard preemption when supported (llama.cpp)
    Given a worker adapter for llama.cpp
    And hard preemption is enabled and adapter proves interruptible_decode
    When I stream task events
    Then preempted flag and resumable state are surfaced

  # vLLM
  Scenario: Soft preemption under overload (vLLM)
    Given a worker adapter for vllm
    And soft preemption is enabled
    And under persistent overload
    Then lower priority items are preempted first
    And preemptions_total and resumptions_total metrics are exported

  Scenario: Hard preemption when supported (vLLM)
    Given a worker adapter for vllm
    And hard preemption is enabled and adapter proves interruptible_decode
    When I stream task events
    Then preempted flag and resumable state are surfaced

  # TGI
  Scenario: Soft preemption under overload (TGI)
    Given a worker adapter for tgi
    And soft preemption is enabled
    And under persistent overload
    Then lower priority items are preempted first
    And preemptions_total and resumptions_total metrics are exported

  Scenario: Hard preemption when supported (TGI)
    Given a worker adapter for tgi
    And hard preemption is enabled and adapter proves interruptible_decode
    When I stream task events
    Then preempted flag and resumable state are surfaced

  # Triton
  Scenario: Soft preemption under overload (Triton)
    Given a worker adapter for triton
    And soft preemption is enabled
    And under persistent overload
    Then lower priority items are preempted first
    And preemptions_total and resumptions_total metrics are exported

  Scenario: Hard preemption when supported (Triton)
    Given a worker adapter for triton
    And hard preemption is enabled and adapter proves interruptible_decode
    When I stream task events
    Then preempted flag and resumable state are surfaced
