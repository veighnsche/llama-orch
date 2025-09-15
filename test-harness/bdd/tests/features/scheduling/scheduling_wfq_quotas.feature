Feature: WFQ fairness and quotas
  # Traceability: ORCH-3075..3077
  Scenario: WFQ approximates weights
    Given WFQ weights are configured for tenants and priorities
    When load arrives across tenants and priorities
    Then observed share approximates configured weights

  Scenario: Quotas reject overflow
    Given quotas are configured per tenant
    When load arrives across tenants and priorities
    Then requests beyond quota are rejected
