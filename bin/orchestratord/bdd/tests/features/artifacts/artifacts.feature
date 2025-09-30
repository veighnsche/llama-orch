Feature: Artifact Storage
  # Traceability: B-ART-001 through B-ART-023
  # Content-addressed artifact storage with SHA-256 IDs

  Scenario: Create artifact returns SHA-256 ID
    Given an artifacts endpoint
    When I create an artifact with document {"key": "value"}
    Then I receive 201 Created
    And the response includes id
    And the artifact id is a SHA-256 hash

  Scenario: Get existing artifact
    Given an artifact with id "test-artifact" exists
    When I get artifact "test-artifact"
    Then I receive 200 OK
    And the response is the artifact document

  Scenario: Get non-existent artifact
    Given an artifact with id "unknown" does not exist
    When I get artifact "unknown"
    Then I receive 404 Not Found

  Scenario: Artifact storage is idempotent
    Given an artifacts endpoint
    When I create the same artifact twice
    Then both requests return the same id
