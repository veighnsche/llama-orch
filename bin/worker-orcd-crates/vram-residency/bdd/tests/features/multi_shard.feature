Feature: Multi-Shard Operations
  As a worker-orcd service
  I want to manage multiple sealed shards concurrently
  So that I can handle tensor-parallel and multi-model workloads

  Background:
    Given a VramManager with 50MB capacity

  Scenario: Seal multiple shards concurrently
    Given a model with 5MB of data
    When I seal the model with shard_id "shard-1" on GPU 0
    And I seal the model with shard_id "shard-2" on GPU 0
    And I seal the model with shard_id "shard-3" on GPU 0
    Then all seals should succeed
    And 3 shards should be tracked
    And total VRAM used should be 15MB

  Scenario: Verify multiple shards independently
    Given 3 sealed shards with 5MB each
    When I verify shard "shard-1"
    And I verify shard "shard-2"
    And I verify shard "shard-3"
    Then all verifications should succeed

  Scenario: Detect tampering in one of multiple shards
    Given 3 sealed shards with 5MB each
    And shard "shard-2" digest is tampered
    When I verify shard "shard-1"
    Then the verification should succeed
    When I verify shard "shard-2"
    Then the verification should fail with "SealVerificationFailed"
    When I verify shard "shard-3"
    Then the verification should succeed

  Scenario: Seal different sized shards
    Given a model with 1MB of data
    When I seal the model with shard_id "small" on GPU 0
    Given a model with 10MB of data
    When I seal the model with shard_id "medium" on GPU 0
    Given a model with 20MB of data
    When I seal the model with shard_id "large" on GPU 0
    Then all seals should succeed
    And shard "small" should have 1MB
    And shard "medium" should have 10MB
    And shard "large" should have 20MB

  Scenario: Capacity exhaustion with multiple shards
    Given a model with 15MB of data
    When I seal the model with shard_id "shard-1" on GPU 0
    And I seal the model with shard_id "shard-2" on GPU 0
    And I seal the model with shard_id "shard-3" on GPU 0
    Then the first 2 seals should succeed
    And the 3rd seal should fail with "InsufficientVram"
