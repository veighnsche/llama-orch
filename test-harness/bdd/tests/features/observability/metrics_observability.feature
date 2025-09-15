Feature: Metrics and observability
  # Traceability: ORCH-3027, ORCH-3028, OC-CTRL-2050..2051
  Scenario: Metrics conform to linter
    Given an OrchQueue API endpoint
    Then metrics conform to linter names and labels
    And label cardinality budgets are enforced

  Scenario: Logs and started event fields
    Given started event and admission logs
    Then include queue_position and predicted_start_ms
