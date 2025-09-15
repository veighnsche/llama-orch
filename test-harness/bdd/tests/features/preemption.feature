Feature: Preemption semantics
  # Traceability: ORCH-3085..3087
  Scenario: Soft preemption under overload
    Given soft preemption is enabled
    And under persistent overload
    Then lower priority items are preempted first
    And preemptions_total and resumptions_total metrics are exported

  Scenario: Hard preemption when supported
    Given hard preemption is enabled and adapter proves interruptible_decode
    When I stream task events
    Then preempted flag and resumable state are surfaced
