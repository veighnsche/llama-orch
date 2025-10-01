Feature: Seal Model in VRAM
  As a worker-orcd service
  I want to seal models in VRAM with cryptographic integrity
  So that I can detect tampering and ensure VRAM residency

  Background:
    Given a VramManager with 10MB capacity

  Scenario: Successfully seal model
    Given a model with 1MB of data
    When I seal the model with shard_id "shard-123" on GPU 0
    Then the seal should succeed
    And the sealed shard should have:
      | field       | value       |
      | shard_id    | shard-123   |
      | gpu_device  | 0           |
      | vram_bytes  | 1048576     |
      | digest      | 64 hex chars|
    And an audit event "VramSealed" should be emitted

  Scenario: Reject invalid shard ID with path traversal
    Given a model with 1MB of data
    When I seal the model with shard_id "../etc/passwd" on GPU 0
    Then the seal should fail with "InvalidInput"
    And no audit event should be emitted

  Scenario: Reject invalid shard ID with null byte
    Given a model with 1MB of data
    When I seal the model with shard_id "shard\0null" on GPU 0
    Then the seal should fail with "InvalidInput"
    And no audit event should be emitted

  Scenario: Fail on insufficient VRAM
    Given a model with 20MB of data
    When I seal the model with shard_id "large-model" on GPU 0
    Then the seal should fail with "InsufficientVram"
    And the error should indicate needed=20MB available=10MB

  Scenario: Accept model at exact capacity
    Given a model with 10MB of data
    When I seal the model with shard_id "exact-fit" on GPU 0
    Then the seal should succeed
    And an audit event "VramSealed" should be emitted

  Scenario: Reject oversized model
    Given a model with 101GB of data
    When I seal the model with shard_id "huge-model" on GPU 0
    Then the seal should fail with "InvalidInput"
