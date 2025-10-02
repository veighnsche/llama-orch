Feature: Stress Testing
  As a production system under heavy load
  I want to handle extreme conditions gracefully
  So that the system remains stable and doesn't leak resources

  Background:
    Given a VramManager with 100MB capacity

  Scenario: Seal models until VRAM exhausted
    When I seal 1MB models until capacity reached
    Then the last seal should fail with InsufficientVram
    And all previous seals should remain valid
    And no memory leaks should occur

  Scenario: Rapid seal cycles
    When I perform 100 rapid seal operations
    Then most seals should succeed
    And failures should only be InsufficientVram
    And no panics should occur

  Scenario: Large model seal
    Given a 10MB model
    When I seal the large model
    Then the seal should succeed or fail with InsufficientVram
    And if successful, verification should pass
    And the shard should report correct size

  Scenario: Many small allocations
    When I seal 1000 tiny 1KB models
    Then at least some models should seal successfully
    And random verification samples should pass
    And allocation tracking should remain consistent

  Scenario: Repeated verification stress
    Given a sealed shard "test" with 1MB of data
    When I verify the shard 1000 times
    Then all verifications should succeed
    And performance should remain consistent

  Scenario: Alternating seal and verify under load
    When I perform 100 alternating seal/verify operations
    Then no deadlocks should occur
    And no data corruption should occur
    And all operations should complete

  Scenario: Capacity queries under load
    When I seal 50 models while querying capacity
    Then all capacity queries should succeed
    And total >= available should always hold
    And no race conditions should occur

  Scenario: Varying model sizes
    When I seal models of sizes 1B, 1KB, 10KB, 100KB, 1MB
    Then each seal should report correct size
    And verification should pass for all sizes
    And no size-related bugs should occur

  Scenario: Signature computation stress
    When I compute signatures for 1000 different shards
    Then all computations should succeed
    And all signatures should be 32 bytes
    And no performance degradation should occur

  Scenario: Verification stress
    When I verify 1000 different signatures
    Then all verifications should succeed
    And timing should remain constant (no timing attacks)

  Scenario: Edge case - single byte model
    Given a 1-byte model
    When I seal the model
    Then the seal should succeed
    And verification should pass
    And the shard should report 1 byte

  Scenario: Edge case - all zeros model
    Given a 1KB model of all zeros
    When I seal the model
    Then the seal should succeed
    And the digest should be deterministic
    And verification should pass

  Scenario: Edge case - all ones model
    Given a 1KB model of all 0xFF bytes
    When I seal the model
    Then the seal should succeed
    And the digest should differ from all-zeros
    And verification should pass
