Feature: Prompt Validation
  As a security-conscious system
  I want to validate user prompts
  So that I prevent resource exhaustion and null byte attacks

  Scenario: Accept valid short prompt
    Given a prompt "Hello, world!"
    And a max length of 100000
    When I validate the prompt
    Then the validation should succeed

  Scenario: Accept valid long prompt
    Given a prompt "Write a story about..."
    And a max length of 100000
    When I validate the prompt
    Then the validation should succeed

  Scenario: Accept empty prompt
    Given a prompt ""
    And a max length of 100000
    When I validate the prompt
    Then the validation should succeed

  Scenario: Accept prompt with Unicode
    Given a prompt "Unicode: café ☕"
    And a max length of 100000
    When I validate the prompt
    Then the validation should succeed

  Scenario: Accept prompt with newlines
    Given a prompt "Line 1\nLine 2"
    And a max length of 100000
    When I validate the prompt
    Then the validation should succeed

  Scenario: Accept prompt with tabs
    Given a prompt "Text\twith\ttabs"
    And a max length of 100000
    When I validate the prompt
    Then the validation should succeed

  Scenario: Reject prompt exceeding max length
    Given a prompt with 100001 characters
    And a max length of 100000
    When I validate the prompt
    Then the validation should fail
    And the error should be "TooLong"

  Scenario: Accept prompt at exact max length (boundary)
    Given a prompt with 100000 characters
    And a max length of 100000
    When I validate the prompt
    Then the validation should succeed

  Scenario: Accept prompt one character under max length
    Given a prompt with 99999 characters
    And a max length of 100000
    When I validate the prompt
    Then the validation should succeed

  Scenario: Reject prompt with null byte
    Given a prompt "prompt\0null"
    And a max length of 100000
    When I validate the prompt
    Then the validation should fail
    And the error should be "NullByte"

  Scenario: Reject prompt with null byte at start
    Given a prompt "\0prompt"
    And a max length of 100000
    When I validate the prompt
    Then the validation should fail
    And the error should be "NullByte"

  Scenario: Reject prompt with null byte at end
    Given a prompt "prompt\0"
    And a max length of 100000
    When I validate the prompt
    Then the validation should fail
    And the error should be "NullByte"

  Scenario: Accept prompt with custom max length
    Given a prompt with 50 characters
    And a max length of 50
    When I validate the prompt
    Then the validation should succeed
