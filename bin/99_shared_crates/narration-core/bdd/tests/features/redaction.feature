Feature: Secret Redaction Behaviors
  As a security-conscious developer
  I want automatic secret redaction
  So that sensitive data never appears in logs

  Scenario: B-RED-001 - Default policy masks bearer tokens
    When I create a default redaction policy
    Then mask_bearer_tokens is true

  Scenario: B-RED-002 - Default policy masks API keys
    When I create a default redaction policy
    Then mask_api_keys is true

  Scenario: B-RED-003 - Default policy does not mask UUIDs
    When I create a default redaction policy
    Then mask_uuids is false

  Scenario: B-RED-004 - Default replacement is [REDACTED]
    When I create a default redaction policy
    Then replacement is "[REDACTED]"

  Scenario: B-RED-010 - Bearer token is redacted (lowercase bearer)
    When I redact "Authorization: Bearer abc123xyz"
    Then the output is "Authorization: [REDACTED]"

  Scenario: B-RED-011 - Bearer token is redacted (lowercase bearer)
    When I redact "Authorization: bearer abc123xyz"
    Then the output is "Authorization: [REDACTED]"

  Scenario: B-RED-012 - Bearer token is redacted (uppercase BEARER)
    When I redact "Authorization: BEARER abc123xyz"
    Then the output is "Authorization: [REDACTED]"

  Scenario: B-RED-013 - Multiple bearer tokens are redacted
    When I redact "Bearer token1 and Bearer token2"
    Then the output does not contain "token1"
    And the output does not contain "token2"
    And the output contains "[REDACTED]"

  Scenario: B-RED-015 - No bearer tokens means no changes
    When I redact "Accepted request; queued at position 3"
    Then the output is "Accepted request; queued at position 3"

  Scenario: B-RED-020 - API key with equals is redacted
    When I redact "api_key=secret123"
    Then the output is "[REDACTED]"

  Scenario: B-RED-021 - apikey (no underscore) is redacted
    When I redact "apikey=secret123"
    Then the output is "[REDACTED]"

  Scenario: B-RED-022 - key= is redacted
    When I redact "key=secret123"
    Then the output is "[REDACTED]"

  Scenario: B-RED-023 - token= is redacted
    When I redact "token=secret123"
    Then the output is "[REDACTED]"

  Scenario: B-RED-024 - secret= is redacted
    When I redact "secret=value123"
    Then the output is "[REDACTED]"

  Scenario: B-RED-025 - password= is redacted
    When I redact "password=value123"
    Then the output is "[REDACTED]"

  Scenario: B-RED-026 - API key with colon is redacted
    When I redact "api_key: secret123"
    Then the output is "[REDACTED]"

  Scenario: B-RED-030 - UUID is redacted when enabled
    When I redact "session_id: 550e8400-e29b-41d4-a716-446655440000" with mask_uuids enabled
    Then the output is "session_id: [REDACTED]"

  Scenario: B-RED-031 - UUID is not redacted by default
    When I redact "session_id: 550e8400-e29b-41d4-a716-446655440000"
    Then the output is "session_id: 550e8400-e29b-41d4-a716-446655440000"

  Scenario: B-RED-005 - Custom replacement string
    When I redact "Bearer abc123" with replacement "***"
    Then the output is "***"
