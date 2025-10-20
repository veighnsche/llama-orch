# Created by: TEAM-159
# REAL integration test - spawns actual daemons and tests first heartbeat flow
#
# This tests the ACTUAL flow:
# 1. Queen-rbee starts
# 2. Queen adds localhost to hive catalog
# 3. Queen spawns rbee-hive daemon (or we spawn it manually)
# 4. rbee-hive sends first heartbeat to queen
# 5. Queen triggers device detection
# 6. Queen stores device capabilities
# 7. Queen updates hive status to Online

Feature: Integration Test - First Heartbeat from Real rbee-hive
  As a system integrator
  I want to verify the complete first heartbeat flow
  So that I know queen and hive work together correctly

  Background:
    Given a temporary directory for test databases
    And queen-rbee is configured to use the test database

  Scenario: Queen receives first heartbeat from spawned rbee-hive
    Given queen-rbee HTTP server is running on port 18500
    And the hive catalog contains a hive "localhost" with status "Unknown"
    And the hive entry points to "localhost:18600"
    
    When rbee-hive daemon starts on port 18600
    And rbee-hive is configured to send heartbeats to "http://localhost:18500"
    And we wait 2 seconds for rbee-hive to initialize
    
    Then rbee-hive should send its first heartbeat to queen
    And queen should receive the heartbeat
    And queen should trigger device detection to "http://localhost:18600/v1/devices"
    And rbee-hive should respond with real device information
    And queen should store the device capabilities in the catalog
    And queen should update hive status to "Online"
    And queen should emit narration "First heartbeat from localhost"
    And queen should emit narration "Checking capabilities"
    
    When we query the hive catalog for "localhost"
    Then the hive should have status "Online"
    And the hive should have device capabilities stored
    And the hive should have a recent last_heartbeat timestamp

  Scenario: Real rbee-hive sends periodic heartbeats
    Given queen-rbee is running on port 18500
    And rbee-hive is running on port 18600 with status "Online"
    
    When we wait 16 seconds for the next heartbeat cycle
    
    Then rbee-hive should send another heartbeat
    And queen should NOT trigger device detection again
    And the hive last_heartbeat timestamp should be updated
    And narration should NOT contain "First heartbeat"

  Scenario: Queen detects when rbee-hive goes offline
    Given queen-rbee is running on port 18500
    And rbee-hive is running on port 18600 with status "Online"
    And rbee-hive has sent at least one heartbeat
    
    When we kill the rbee-hive daemon
    And we wait 60 seconds for heartbeat timeout
    
    Then queen should detect the missed heartbeats
    And queen should update hive status to "Offline"
    And queen should emit narration about hive going offline
