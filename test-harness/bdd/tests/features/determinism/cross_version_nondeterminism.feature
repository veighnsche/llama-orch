Feature: Determinism across engine or model versions
  # Traceability: ORCH-3047

  Scenario: Cross-version runs are not assumed deterministic
    Given replicas across engine or model versions are used
    And same prompt parameters and seed are used
    Then determinism is not assumed across engine or model updates
