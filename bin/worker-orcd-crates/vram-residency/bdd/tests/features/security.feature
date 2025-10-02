Feature: Security Properties
  As a security-conscious system
  I want to enforce cryptographic integrity and prevent attacks
  So that sealed models cannot be tampered with

  Background:
    Given a VramManager with 10MB capacity

  Scenario: Signature verification detects tampering
    Given a sealed shard "test" with 1MB of data
    And the shard signature is forged
    When I verify the sealed shard
    Then the verification should fail with "SealVerificationFailed"
    And an audit event "SealVerificationFailed" should be emitted

  Scenario: Digest verification detects VRAM corruption
    Given a sealed shard "test" with 1MB of data
    And the VRAM contents are corrupted
    When I verify the sealed shard
    Then the verification should fail with "SealVerificationFailed"


  Scenario: Seal keys are never logged
    Given a model with 1MB of data
    When I seal the model with shard_id "test" on GPU 0
    Then the seal should succeed
    And the logs should not contain seal key material

  Scenario: VRAM pointers are never exposed
    Given a sealed shard "test" with 1MB of data
    When I serialize the shard to JSON
    Then the JSON should not contain VRAM pointer
    And the JSON should not contain memory addresses
