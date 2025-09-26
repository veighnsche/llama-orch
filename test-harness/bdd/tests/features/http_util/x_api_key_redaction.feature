Feature: http-util X-API-Key redaction (HTU-1004)
  As a developer using http-util
  I want API keys to be redacted in logs
  So that secrets are not leaked

  Scenario: Redact X-API-Key
    Given a log line with X-API-Key "key-1234-abcdef"
    When I apply http-util redaction
    Then the output masks X-API-Key and includes its fp6
