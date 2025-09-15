Feature: Catalog strict trust policy rejects unsigned artifacts
  # Traceability: ORCH-3060..3065, ORCH-3093

  Scenario: Strict policy rejects unsigned artifact at ingest
    Given a Control Plane API endpoint
    And strict trust policy is enabled
    And an unsigned catalog artifact
    When I create a catalog model
    Then catalog ingestion fails with UNTRUSTED_ARTIFACT
