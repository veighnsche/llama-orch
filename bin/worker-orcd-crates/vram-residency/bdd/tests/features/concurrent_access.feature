Feature: Concurrent Access Robustness
  As a worker-orcd service under load
  I want to handle concurrent seal and verify operations safely
  So that race conditions do not corrupt VRAM state

  Background:
    Given a VramManager with 100MB capacity

  Scenario: Concurrent seal operations succeed
    Given 10 concurrent threads
    When each thread seals a 1MB model
    Then all seals should succeed
    And all shard IDs should be unique
    And no race conditions should occur

  Scenario: Concurrent verification of same shard
    Given a sealed shard "test-shard" with 1MB of data
    And 10 concurrent threads
    When each thread verifies the same shard
    Then all verifications should succeed
    And no data corruption should occur

  Scenario: Interleaved seal and verify operations
    Given 20 concurrent threads
    When threads alternate between seal and verify operations
    Then no deadlocks should occur
    And no panics should occur
    And allocation tracking should remain consistent

  Scenario: Concurrent capacity queries
    Given 20 concurrent threads
    When each thread queries available VRAM
    Then all queries should succeed
    And total VRAM should always be >= available VRAM

  Scenario: Concurrent seal until capacity exhausted
    Given 20 concurrent threads
    When threads seal 1MB models until VRAM exhausted
    Then some seals should succeed
    And some seals should fail with InsufficientVram
    And all successful seals should remain valid
