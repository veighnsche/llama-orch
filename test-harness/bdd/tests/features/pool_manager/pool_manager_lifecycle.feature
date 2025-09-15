Feature: Pool Manager lifecycle and placement
  # Traceability: OC-POOL-3001..3012, ORCH-3010..3011
  Scenario: Preload failure keeps pool Unready
    Given pool is Unready due to preload failure
    Then pool readiness is false and last error cause is present

  Scenario: Driver error triggers backoff restart
    Given driver error occurs
    Then pool transitions to Unready and restarts with backoff
    And restart storms are bounded by circuit breaker

  Scenario: Placement respects device masks
    Given device masks are configured
    Then placement respects device masks; no cross-mask spillover occurs

  Scenario: Heterogeneous split ratios cap per-GPU KV
    Given heterogeneous split ratios are configured
    Then per-GPU resident KV is capped for smallest GPU
