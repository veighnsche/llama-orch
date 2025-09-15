Feature: WFQ with deadlines and preemption
  # Traceability: ORCH-3075, ORCH-3076, ORCH-3079, ORCH-3085, ORCH-3087

  Scenario: Weighted fairness with deadlines and preemption metrics
    Given WFQ weights are configured for tenants and priorities
    And a task with infeasible deadline
    And soft preemption is enabled
    When load arrives across tenants and priorities
    Then observed share approximates configured weights
    And preemptions_total and resumptions_total metrics are exported
    And SSE metrics include on_time_probability
