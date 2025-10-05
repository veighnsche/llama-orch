Feature: Error Handling
  As a worker implementation
  I want to classify errors correctly
  So that the orchestrator can retry appropriately

  Scenario: Timeout error is retriable
    Given a worker error "Timeout"
    When I check if the error is retriable
    Then the error should be retriable
    And the HTTP status code should be 504

  Scenario: Invalid request is not retriable
    Given a worker error "InvalidRequest"
    When I check if the error is retriable
    Then the error should be non-retriable
    And the HTTP status code should be 400

  Scenario: Internal error is retriable
    Given a worker error "Internal"
    When I check if the error is retriable
    Then the error should be retriable
    And the HTTP status code should be 500
