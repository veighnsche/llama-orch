Feature: Error Handling
  As a worker implementation
  I want to classify errors correctly
  So that the orchestrator can retry appropriately

  Scenario: Timeout error is retriable
    Given a worker error "Timeout"
    When I check if the error is retriable
    Then the error should be retriable
    And the HTTP status code should be 408

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
  
  Scenario: CUDA error is retriable
    Given a worker error "Cuda"
    When I check if the error is retriable
    Then the error should be retriable
    And the HTTP status code should be 500
  
  Scenario: Unhealthy worker is not retriable
    Given a worker error "Unhealthy"
    When I check if the error is retriable
    Then the error should be non-retriable
    And the HTTP status code should be 503
