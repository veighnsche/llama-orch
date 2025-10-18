# Integration Scenario Tests
# Created by: TEAM-106
# Priority: P1 - High priority integration scenarios
# Purpose: Test complex multi-component scenarios

Feature: Integration Scenarios
  As an integration tester
  I want to test complex multi-component scenarios
  So that edge cases and failure modes are validated

  Background:
    Given the integration test environment is running

  @integration @p1 @multi-hive
  Scenario: INT-001 - Multi-hive deployment
    Given queen-rbee is running
    And rbee-hive-1 is running on port 9200
    And rbee-hive-2 is running on port 9201
    And rbee-hive-1 has 2 workers
    And rbee-hive-2 has 2 workers
    When client sends 10 inference requests
    Then requests are distributed across both hives
    And each hive processes requests
    And all requests complete successfully
    And load is balanced across hives

  @integration @p1 @worker-churn
  Scenario: INT-002 - Worker churn (rapid spawn/shutdown)
    Given rbee-hive is running
    When 5 workers are spawned simultaneously
    And 3 workers are shutdown immediately
    And 2 new workers are spawned
    Then registry state remains consistent
    And no orphaned workers exist
    And active workers are tracked correctly
    And shutdown workers are removed

  @integration @p1 @worker-churn
  Scenario: INT-003 - Worker restart during inference
    Given worker is processing long-running inference
    And inference is 50% complete
    When worker is restarted
    Then in-flight request is handled gracefully
    And client receives appropriate error
    And worker restarts successfully
    And worker is available for new requests
    And no data corruption occurs

  @integration @p2 @network-partition
  Scenario: INT-004 - Network partition between queen and hive
    Given queen-rbee is running
    And rbee-hive is running
    And network connection is stable
    When network partition occurs
    Then queen-rbee detects connection loss
    And queen-rbee marks hive as unavailable
    And new requests are rejected with error
    When network is restored
    Then queen-rbee reconnects to hive
    And hive is marked as available
    And requests resume normally

  @integration @p2 @network-partition
  Scenario: INT-005 - Network partition between hive and worker
    Given rbee-hive is running
    And worker is registered
    When network partition occurs between hive and worker
    Then rbee-hive detects worker timeout
    And worker is marked as unhealthy
    And new requests are not routed to worker
    When network is restored
    Then worker sends heartbeat
    And worker is marked as healthy
    And requests resume to worker

  @integration @p2 @database-failure
  Scenario: INT-006 - Model catalog database corruption
    Given rbee-hive is running
    And model catalog has 5 models
    When database file is corrupted
    Then rbee-hive detects corruption on next query
    And rbee-hive attempts recovery
    And error is logged with details
    And rbee-hive continues with in-memory fallback
    And new models can still be provisioned

  @integration @p2 @database-failure
  Scenario: INT-007 - Worker registry database failure
    Given queen-rbee is running
    And 3 workers are registered
    When registry database becomes unavailable
    Then queen-rbee uses in-memory cache
    And existing workers remain accessible
    And new registrations are queued
    When database is restored
    Then queued registrations are persisted
    And registry state is synchronized

  @integration @p2 @oom
  Scenario: INT-008 - Worker OOM during model loading
    Given rbee-hive attempts to spawn worker
    And model requires 8GB VRAM
    And only 4GB VRAM is available
    When worker attempts to load model
    Then worker OOM kills during loading
    And rbee-hive detects worker crash
    And error is reported to client
    And worker is not registered
    And resources are cleaned up

  @integration @p2 @oom
  Scenario: INT-009 - Worker OOM during inference
    Given worker is processing inference
    And inference generates large context
    When worker exceeds memory limit
    Then worker is OOM killed
    And rbee-hive detects crash
    And client receives error response
    And worker is removed from registry
    And resources are freed

  @integration @p1 @concurrency
  Scenario: INT-010 - Concurrent worker registration
    Given rbee-hive is running
    When 10 workers register simultaneously
    Then all registrations are processed
    And no race conditions occur
    And all workers have unique IDs
    And registry state is consistent
    And all workers are queryable

  @integration @p1 @concurrency
  Scenario: INT-011 - Concurrent model downloads
    Given rbee-hive is running
    When 3 clients request same model simultaneously
    Then only one download is initiated
    And other requests wait for download
    And download completes successfully
    And all clients receive model
    And no duplicate downloads occur

  @integration @p1 @failure-recovery
  Scenario: INT-012 - Queen-rbee restart with active workers
    Given queen-rbee is running
    And 3 workers are registered
    When queen-rbee is restarted
    Then queen-rbee loses in-memory registry
    And queen-rbee queries rbee-hive for workers
    And workers are re-discovered
    And registry is rebuilt
    And requests resume normally

  @integration @p1 @failure-recovery
  Scenario: INT-013 - Rbee-hive restart with active workers
    Given rbee-hive is running
    And 3 workers are registered
    When rbee-hive is restarted
    Then rbee-hive loads registry from database
    And workers send heartbeats
    And workers are marked as healthy
    And registry is restored
    And requests resume normally

  @integration @p2 @performance
  Scenario: INT-014 - High throughput stress test
    Given system is running
    When 1000 requests are sent over 60 seconds
    Then all requests are processed
    And average latency is under 100ms
    And p99 latency is under 500ms
    And no requests timeout
    And no memory leaks occur

  @integration @p2 @performance
  Scenario: INT-015 - Long-running inference stability
    Given worker is available
    When inference runs for 10 minutes
    Then worker remains stable
    And memory usage is constant
    And no resource leaks occur
    And worker can process subsequent requests
    And system remains responsive
