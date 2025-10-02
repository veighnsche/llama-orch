Feature: Shared Crate Integration
  As a security-conscious system
  I want to leverage shared validation and audit crates
  So that security policies are consistent across all binaries

  Background:
    Given a VramManager with 10MB capacity

  Scenario: Validate shard ID via input-validation crate
    Given a shard ID with invalid characters "@#$%"
    When I attempt to seal a model with that shard ID
    Then validation should fail via input-validation crate
    And the error should be InvalidInput

  Scenario: Validate shard ID with path traversal
    Given a shard ID "../etc/passwd"
    When I attempt to seal a model with that shard ID
    Then validation should fail via input-validation crate
    And the error should mention "path traversal"

  Scenario: Audit logging integration on seal
    Given a model with 1MB of data
    When I seal the model with shard_id "test-shard" on GPU 0
    Then an audit event "VramSealed" should be emitted
    And the event should contain shard_id "test-shard"
    And the event should contain gpu_device 0
    And the event should contain vram_bytes 1048576

  Scenario: Audit logging integration on verification failure
    Given a sealed shard "test" with 1MB of data
    And the shard digest is modified
    When I verify the sealed shard
    Then an audit event "SealVerificationFailed" should be emitted
    And the event should have severity "CRITICAL"
    And the event should contain reason "digest_mismatch"

  Scenario: Secrets management integration for seal keys
    Given a worker token "test-token-42"
    When I create a VramManager with that token
    Then the seal key should be derived via HKDF-SHA256
    And the seal key should be auto-zeroizing
    And the token should be logged as fingerprint only

  Scenario: Defense-in-depth validation
    Given a shard ID that passes shared validation
    But contains VRAM-specific invalid patterns
    When I attempt to seal a model with that shard ID
    Then local validation should reject it
    And the error should be InvalidInput
