Feature: Verify Sealed Shard
  As a worker-orcd service
  I want to verify sealed shards before execution
  So that I can detect VRAM corruption or tampering

  Background:
    Given a sealed shard "shard-123" with 1MB of data

  Scenario: Verify valid seal
    When I verify the sealed shard
    Then the verification should succeed
    And an audit event "SealVerified" should be emitted

  Scenario: Reject tampered digest
    Given the shard digest is modified to "0000000000000000000000000000000000000000000000000000000000000000"
    When I verify the sealed shard
    Then the verification should fail with "SealVerificationFailed"
    And an audit event "SealVerificationFailed" should be emitted
    And the event should have severity "critical"

  Scenario: Reject forged signature
    Given the shard signature is replaced with zeros
    When I verify the sealed shard
    Then the verification should fail with "SealVerificationFailed"
    And an audit event "SealVerificationFailed" should be emitted
    And the event should have severity "critical"
