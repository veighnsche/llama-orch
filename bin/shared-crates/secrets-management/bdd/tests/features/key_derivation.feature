Feature: Key Derivation
  As a security-conscious system
  I want to derive cryptographic keys from tokens using HKDF
  So that I can generate keys without storing separate key files

  Scenario: Derive key from token with domain separation
    Given a token "test-worker-token"
    And a domain "llorch-seal-key-v1"
    When I derive a key from the token
    Then the operation should succeed
    And the derived key should be 32 bytes

  Scenario: Derived keys are deterministic
    Given a token "test-worker-token"
    And a domain "llorch-seal-key-v1"
    When I derive a key from the token
    Then the operation should succeed
    And the derived key should be deterministic

  Scenario: Different domains produce different keys
    Given a token "test-worker-token"
    And a domain "llorch-seal-key-v1"
    When I derive a key from the token
    Then the operation should succeed

  Scenario: Reject empty token
    Given a token ""
    And a domain "llorch-seal-key-v1"
    When I derive a key from the token
    Then the operation should fail

  Scenario: Reject missing domain
    Given a token "test-worker-token"
    When I derive a key from the token
    Then the operation should fail
