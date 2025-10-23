Feature: Range Validation
  As a security-conscious system
  I want to validate integer ranges
  So that I prevent overflow and invalid values

  Scenario: Accept value within range
    Given a value 2
    And a range from 0 to 4
    When I validate the range
    Then the validation should succeed

  Scenario: Accept value at minimum (inclusive)
    Given a value 0
    And a range from 0 to 4
    When I validate the range
    Then the validation should succeed

  Scenario: Accept value one below maximum
    Given a value 3
    And a range from 0 to 4
    When I validate the range
    Then the validation should succeed

  Scenario: Reject value at maximum (exclusive)
    Given a value 4
    And a range from 0 to 4
    When I validate the range
    Then the validation should fail
    And the error should be "OutOfRange"

  Scenario: Reject value below minimum
    Given a value -1
    And a range from 0 to 4
    When I validate the range
    Then the validation should fail
    And the error should be "OutOfRange"

  Scenario: Reject value above maximum
    Given a value 5
    And a range from 0 to 4
    When I validate the range
    Then the validation should fail
    And the error should be "OutOfRange"

  Scenario: Support negative ranges
    Given a value -5
    And a range from -10 to 0
    When I validate the range
    Then the validation should succeed

  Scenario: Support ranges crossing zero
    Given a value 0
    And a range from -10 to 10
    When I validate the range
    Then the validation should succeed

  Scenario: Reject value below negative minimum
    Given a value -11
    And a range from -10 to 0
    When I validate the range
    Then the validation should fail
    And the error should be "OutOfRange"

  Scenario: Accept value at exact minimum (boundary)
    Given a value 0
    And a range from 0 to 10
    When I validate the range
    Then the validation should succeed

  Scenario: Reject value one below minimum (boundary)
    Given a value -1
    And a range from 0 to 10
    When I validate the range
    Then the validation should fail
    And the error should be "OutOfRange"

  Scenario: Accept value one below maximum (boundary)
    Given a value 9
    And a range from 0 to 10
    When I validate the range
    Then the validation should succeed

  Scenario: Reject value at exact maximum (boundary)
    Given a value 10
    And a range from 0 to 10
    When I validate the range
    Then the validation should fail
    And the error should be "OutOfRange"
