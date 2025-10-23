Feature: Hex String Validation
  As a security-conscious system
  I want to validate hexadecimal strings (digests, hashes, signatures)
  So that I ensure cryptographic integrity

  Scenario: Accept valid lowercase hex string
    Given a hex string "abcdef0123456789"
    And an expected length of 16
    When I validate the hex string
    Then the validation should succeed

  Scenario: Accept valid uppercase hex string
    Given a hex string "ABCDEF0123456789"
    And an expected length of 16
    When I validate the hex string
    Then the validation should succeed

  Scenario: Accept mixed case hex string (case-insensitive)
    Given a hex string "AbCdEf0123456789"
    And an expected length of 16
    When I validate the hex string
    Then the validation should succeed

  Scenario: Accept SHA-256 digest (64 hex chars)
    Given a hex string with 64 valid hex characters
    And an expected length of 64
    When I validate the hex string
    Then the validation should succeed

  Scenario: Accept SHA-1 digest (40 hex chars)
    Given a hex string with 40 valid hex characters
    And an expected length of 40
    When I validate the hex string
    Then the validation should succeed

  Scenario: Accept MD5 digest (32 hex chars)
    Given a hex string with 32 valid hex characters
    And an expected length of 32
    When I validate the hex string
    Then the validation should succeed

  Scenario: Reject hex string shorter than expected
    Given a hex string "abc"
    And an expected length of 64
    When I validate the hex string
    Then the validation should fail
    And the error should be "WrongLength"

  Scenario: Reject hex string longer than expected
    Given a hex string with 65 valid hex characters
    And an expected length of 64
    When I validate the hex string
    Then the validation should fail
    And the error should be "WrongLength"

  Scenario: Reject hex string with null byte
    Given a hex string "abc\0def"
    And an expected length of 7
    When I validate the hex string
    Then the validation should fail
    And the error should be "NullByte"

  Scenario: Reject hex string with 'g' character
    Given a hex string "abcg"
    And an expected length of 4
    When I validate the hex string
    Then the validation should fail
    And the error should be "InvalidHex"

  Scenario: Reject hex string with space
    Given a hex string "abc 123"
    And an expected length of 7
    When I validate the hex string
    Then the validation should fail
    And the error should be "InvalidHex"

  Scenario: Reject hex string with hyphen
    Given a hex string "abc-def"
    And an expected length of 7
    When I validate the hex string
    Then the validation should fail
    And the error should be "InvalidHex"

  Scenario: Reject empty hex string when length expected
    Given a hex string ""
    And an expected length of 64
    When I validate the hex string
    Then the validation should fail
    And the error should be "WrongLength"

  Scenario: Accept hex string at exact expected length (boundary)
    Given a hex string with 64 valid hex characters
    And an expected length of 64
    When I validate the hex string
    Then the validation should succeed
