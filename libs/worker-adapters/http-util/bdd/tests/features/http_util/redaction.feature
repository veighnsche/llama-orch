Feature: http-util redaction (HTU-1004)
  As a developer using http-util
  I want sensitive headers and tokens to be redacted
  So that logs never leak secrets

  Scenario: Redact Authorization Bearer token
    Given no special http-util configuration
    And a log line with Authorization Bearer token "s3cr3t-token-xyz"
    When I apply http-util redaction
    Then the output masks the token and includes its fp6
    And the output does not contain the raw token
