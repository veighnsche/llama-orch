Feature: Request Validation
  As a worker implementation
  I want to validate HTTP requests
  So that I reject invalid requests early

  Scenario: Valid request with all required headers
    Given a request with "Content-Type: application/json" header
    And a request with "X-Correlation-Id: abc123" header
    When I validate the request
    Then the validation should pass

  Scenario: Missing required header
    Given a request with "Content-Type: application/json" header
    When I validate the request
    Then the validation should fail
    And the error should be "MissingHeader"
