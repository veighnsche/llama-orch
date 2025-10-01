Feature: String Sanitization
  As a security-conscious system
  I want to sanitize strings for safe logging
  So that I prevent log injection and terminal attacks

  Scenario: Accept and return normal text unchanged
    Given a string "normal text"
    When I sanitize the string
    Then the validation should succeed

  Scenario: Accept text with tab
    Given a string "text with\ttab"
    When I sanitize the string
    Then the validation should succeed

  Scenario: Accept text with newline
    Given a string "text with\nnewline"
    When I sanitize the string
    Then the validation should succeed

  Scenario: Accept text with CRLF
    Given a string "text with\r\nCRLF"
    When I sanitize the string
    Then the validation should succeed

  Scenario: Accept empty string
    Given a string ""
    When I sanitize the string
    Then the validation should succeed

  Scenario: Accept Unicode text
    Given a string "café ☕"
    When I sanitize the string
    Then the validation should succeed

  Scenario: Reject string with null byte
    Given a string "text\0null"
    When I sanitize the string
    Then the validation should fail
    And the error should be "NullByte"

  Scenario: Reject string with null byte at start
    Given a string "\0text"
    When I sanitize the string
    Then the validation should fail
    And the error should be "NullByte"

  Scenario: Reject string with null byte at end
    Given a string "text\0"
    When I sanitize the string
    Then the validation should fail
    And the error should be "NullByte"

  Scenario: Reject string with ANSI color escape
    Given a string "text\x1b[31mred"
    When I sanitize the string
    Then the validation should fail
    And the error should be "AnsiEscape"

  Scenario: Reject string with ANSI reset escape
    Given a string "text\x1b[0m"
    When I sanitize the string
    Then the validation should fail
    And the error should be "AnsiEscape"

  Scenario: Reject string with ANSI cursor movement
    Given a string "text\x1b[2J"
    When I sanitize the string
    Then the validation should fail
    And the error should be "AnsiEscape"

  Scenario: Reject string with ANSI escape at start
    Given a string "\x1b[31mred text"
    When I sanitize the string
    Then the validation should fail
    And the error should be "AnsiEscape"

  Scenario: Reject string with control character (0x01)
    Given a string "text\x01control"
    When I sanitize the string
    Then the validation should fail
    And the error should be "ControlCharacter"

  Scenario: Reject string with control character (0x1f)
    Given a string "text\x1fcontrol"
    When I sanitize the string
    Then the validation should fail
    And the error should be "ControlCharacter"

  Scenario: Reject string with bell character (0x07)
    Given a string "text\x07bell"
    When I sanitize the string
    Then the validation should fail
    And the error should be "ControlCharacter"

  Scenario: Allow newlines (multi-line logs are valid)
    Given a string "text\nmore text"
    When I sanitize the string
    Then the validation should succeed

  Scenario: Block ANSI escapes even with newlines
    Given a string "text\x1b[31m[ERROR] Fake"
    When I sanitize the string
    Then the validation should fail
    And the error should be "AnsiEscape"
