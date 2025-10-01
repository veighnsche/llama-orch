Feature: Security Properties
  As a security-conscious system
  I want secrets to never be logged or exposed
  So that I prevent credential leakage

  Scenario: Secrets are not logged via Debug trait
    Given a secret file containing "secret-token"
    When I load the secret from file
    Then the operation should succeed
    And the secret should not be logged

  Scenario: Secrets are zeroized on drop
    Given a secret file containing "secret-token"
    When I load the secret from file
    Then the operation should succeed
    And the secret should be zeroized on drop

  Scenario: Error messages do not contain secret values
    Given a secret file containing "secret-token-abc123"
    When I verify the secret with "wrong-token"
    Then the verification should fail
    And the secret should not be logged

  Scenario: File paths are validated before reading
    Given a secret file at "/etc/llorch/secrets/api-token"
    And a secret file with permissions 384
    When I load the secret from file
    Then the operation should succeed

  Scenario: Symlinks are resolved and validated
    Given a secret file at "/etc/llorch/secrets/symlink-to-token"
    When I load the secret from file
    Then the operation should succeed
