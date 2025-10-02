Feature: Extended Seal Verification
  As a worker-orcd service
  I want comprehensive seal verification
  So that all tampering scenarios are detected

  Background:
    Given a VramManager with 10MB capacity

  Scenario: Verify freshly sealed shard
    Given a model with 1MB of data
    When I seal the model with shard_id "fresh" on GPU 0
    And I immediately verify the sealed shard
    Then the verification should succeed

  Scenario: Verify shard after time delay
    Given a sealed shard "old" with 1MB of data
    And 5 seconds have passed
    When I verify the sealed shard
    Then the verification should succeed

  Scenario: Reject shard with missing signature
    Given a sealed shard "test" with 1MB of data
    And the shard signature is removed
    When I verify the sealed shard
    Then the verification should fail with "NotSealed"

  Scenario: Reject unsealed shard
    Given an unsealed shard "test" with 1MB of data
    When I verify the sealed shard
    Then the verification should fail with "NotSealed"

