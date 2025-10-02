Feature: Cryptographic Signature Robustness
  As a security engineer
  I want cryptographic signatures to be robust against edge cases and attacks
  So that VRAM integrity is guaranteed under all conditions

  Background:
    Given a test shard with valid properties
    And a 32-byte seal key

  # ========== Input Validation ==========

  Scenario: Empty seal key is rejected
    When I attempt to compute signature with empty seal key
    Then the operation should fail with ConfigError
    And the error message should mention "seal key cannot be empty"

  Scenario: Short seal key is rejected
    When I attempt to compute signature with 16-byte seal key
    Then the operation should fail with ConfigError
    And the error message should mention "must be at least 32 bytes"

  Scenario: Oversized seal key is rejected
    When I attempt to compute signature with 2048-byte seal key
    Then the operation should fail with ConfigError
    And the error message should mention "too large"

  Scenario: Empty shard ID is rejected
    Given the shard has empty shard_id
    When I attempt to compute signature
    Then the operation should fail with InvalidInput
    And the error message should mention "shard_id cannot be empty"

  Scenario: Invalid digest length is rejected
    Given the shard has digest with length 32 instead of 64
    When I attempt to compute signature
    Then the operation should fail with InvalidInput
    And the error message should mention "digest must be 64 hex chars"

  Scenario: Zero VRAM bytes is rejected
    Given the shard has vram_bytes set to 0
    When I attempt to compute signature
    Then the operation should fail with InvalidInput
    And the error message should mention "vram_bytes cannot be zero"

  # ========== Signature Verification ==========

  Scenario: Empty signature is rejected
    When I attempt to verify with empty signature
    Then the operation should fail with SealVerificationFailed

  Scenario: Wrong length signature is rejected
    When I attempt to verify with 64-byte signature instead of 32
    Then the operation should fail with SealVerificationFailed
    And the error should be logged

  Scenario: Single bit flip is detected
    Given a valid signature
    When I flip bit 0 of byte 0 in the signature
    And I attempt to verify the tampered signature
    Then the operation should fail with SealVerificationFailed

  Scenario: All bits flips are detected
    Given a valid signature
    When I flip each bit in the first 8 bytes
    Then all verification attempts should fail

  # ========== Edge Cases ==========

  Scenario: All-zeros key works
    Given a seal key with all bytes set to 0x00
    When I compute and verify signature
    Then the signature should be valid

  Scenario: All-ones key works
    Given a seal key with all bytes set to 0xFF
    When I compute and verify signature
    Then the signature should be valid

  Scenario: Maximum key size works
    Given a 1024-byte seal key
    When I compute and verify signature
    Then the signature should be valid

  Scenario: Large VRAM size works
    Given the shard has vram_bytes set to 128GB
    When I compute and verify signature
    Then the signature should be valid

  Scenario: Unicode shard ID works
    Given the shard has shard_id "shard-🦀-rust-αβγ-测试"
    When I compute and verify signature
    Then the signature should be valid

  Scenario: Maximum length shard ID works
    Given the shard has shard_id with 256 characters
    When I compute and verify signature
    Then the signature should be valid

  # ========== Tampering Detection ==========

  Scenario: Tampered shard ID is detected
    Given a valid signature
    When I change the shard_id
    And I attempt to verify the signature
    Then the operation should fail with SealVerificationFailed

  Scenario: Tampered digest is detected
    Given a valid signature
    When I change the digest
    And I attempt to verify the signature
    Then the operation should fail with SealVerificationFailed

  Scenario: Wrong key is detected
    Given a signature computed with key1
    When I attempt to verify with key2
    Then the operation should fail with SealVerificationFailed

  # ========== Determinism ==========

  Scenario: Signature computation is deterministic
    When I compute signature twice with same inputs
    Then both signatures should be identical

  Scenario: Different shards produce different signatures
    Given two shards with different shard_ids
    When I compute signatures for both
    Then the signatures should be different

  Scenario: Different keys produce different signatures
    Given two different seal keys
    When I compute signatures with both keys
    Then the signatures should be different
