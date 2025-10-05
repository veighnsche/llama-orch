Feature: Inference Execution
  As a worker implementation
  I want to run inference
  So that I can generate text

  Scenario: Run inference with valid parameters
    Given an initialized compute backend
    And a loaded model
    And a prompt "Hello world"
    And max_tokens is 100
    And temperature is 0.7
    When I start inference
    Then inference should start successfully
    And I should receive 2 tokens

  Scenario: Run inference with empty prompt
    Given an initialized compute backend
    And a loaded model
    And a prompt ""
    And max_tokens is 100
    And temperature is 0.7
    When I start inference
    Then the inference should fail
    And the error should be "InvalidParameter"

  Scenario: Run inference with invalid temperature
    Given an initialized compute backend
    And a loaded model
    And a prompt "Test"
    And max_tokens is 100
    And temperature is 2.5
    When I start inference
    Then the inference should fail
    And the error should be "InvalidParameter"

  Scenario: Run inference with zero max_tokens
    Given an initialized compute backend
    And a loaded model
    And a prompt "Test"
    And max_tokens is 0
    And temperature is 0.7
    When I start inference
    Then the inference should fail
    And the error should be "InvalidParameter"
