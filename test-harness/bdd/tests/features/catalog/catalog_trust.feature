Feature: Catalog & Trust policy
  # Traceability: ORCH-3060..3066
  Scenario: Create model catalog entry
    Given a Control Plane API endpoint
    And a catalog model payload
    When I create a catalog model
    Then the model is created

  Scenario: Get model
    Given a Control Plane API endpoint
    When I get the catalog model
    Then the manifest signatures and sbom are present

  Scenario: Verify model
    Given a Control Plane API endpoint
    And a catalog model payload
    When I verify the catalog model
    Then verification starts
