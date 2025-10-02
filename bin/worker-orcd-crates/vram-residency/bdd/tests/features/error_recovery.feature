Feature: Error Recovery
  As a resilient system
  I want to recover gracefully from errors
  So that temporary failures don't crash the worker

  Background:
    Given a VramManager with 10MB capacity

  Scenario: Recover from failed seal attempt
    Given a model with 20MB of data
    When I seal the model with shard_id "too-large" on GPU 0
    Then the seal should fail with "InsufficientVram"
    Given a model with 5MB of data
    When I seal the model with shard_id "normal" on GPU 0
    Then the seal should succeed

  Scenario: Recover from verification failure
    Given a sealed shard "test" with 1MB of data
    And the shard digest is modified to "0000000000000000000000000000000000000000000000000000000000000000"
    When I verify the sealed shard
    Then the verification should fail with "SealVerificationFailed"
    Given a sealed shard "test2" with 1MB of data
    When I verify the sealed shard
    Then the verification should succeed

  Scenario: Continue after invalid input
    Given a model with 1MB of data
    When I seal the model with shard_id "../invalid" on GPU 0
    Then the seal should fail with "InvalidInput"
    When I seal the model with shard_id "valid-shard" on GPU 0
    Then the seal should succeed

  Scenario: Handle zero-size model gracefully
    Given a model with 0 bytes of data
    When I seal the model with shard_id "empty" on GPU 0
    Then the seal should fail with "InvalidInput"
    Given a model with 1MB of data
    When I seal the model with shard_id "normal" on GPU 0
    Then the seal should succeed
