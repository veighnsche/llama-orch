Feature: Catalog Management
  # Traceability: B-CAT-001 through B-CAT-043
  # Model CRUD operations, lifecycle management, verification

  Scenario: Create model with digest
    Given a catalog endpoint
    When I create a model with id llama-3-8b and digest sha256:abc123
    Then I receive 201 Created
    And the response includes id llama-3-8b
    And the response includes digest sha256:abc123

  Scenario: Create model without id fails
    Given a catalog endpoint
    When I create a model without an id
    Then I receive 400 Bad Request
    And the error message is "id required"

  Scenario: Get existing model
    Given a model "llama-3-8b" exists in catalog
    When I get model "llama-3-8b"
    Then I receive 200 OK
    And the response includes id and digest

  Scenario: Get non-existent model
    Given a model "unknown" does not exist
    When I get model "unknown"
    Then I receive 404 Not Found

  Scenario: Verify model updates timestamp
    Given a model "llama-3-8b" exists in catalog
    When I verify model "llama-3-8b"
    Then I receive 202 Accepted
    And last_verified_ms is updated

  Scenario: Set model state to Retired
    Given a model "llama-3-8b" exists with state Active
    When I set model state to Retired
    Then I receive 202 Accepted
    And the model state is Retired

  Scenario: Delete existing model
    Given a model "llama-3-8b" exists in catalog
    When I delete model "llama-3-8b"
    Then I receive 204 No Content
    And the model is removed from catalog
