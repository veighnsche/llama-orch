Feature: Request Validation
  As a worker implementation
  I want to validate HTTP requests
  So that I reject invalid requests early

  Scenario: Valid request with all parameters
    Given a request with job_id "test-123"
    And a request with prompt "Hello world"
    And a request with max_tokens 100
    And a request with temperature 0.7
    When I validate the request
    Then the validation should pass

  Scenario: Empty job_id
    Given a request with job_id ""
    And a request with prompt "Hello world"
    And a request with max_tokens 100
    When I validate the request
    Then the validation should fail
    And the error field should be "job_id"

  Scenario: Empty prompt
    Given a request with job_id "test-123"
    And a request with prompt ""
    And a request with max_tokens 100
    When I validate the request
    Then the validation should fail
    And the error field should be "prompt"

  Scenario: Invalid max_tokens (too small)
    Given a request with job_id "test-123"
    And a request with prompt "Hello"
    And a request with max_tokens 0
    When I validate the request
    Then the validation should fail
    And the error field should be "max_tokens"

  Scenario: Invalid temperature (too high)
    Given a request with job_id "test-123"
    And a request with prompt "Hello"
    And a request with max_tokens 100
    And a request with temperature 2.5
    When I validate the request
    Then the validation should fail
    And the error field should be "temperature"
