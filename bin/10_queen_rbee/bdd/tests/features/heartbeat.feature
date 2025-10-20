# Created by: TEAM-158
# Tests heartbeat receiving and device detection on first heartbeat

Feature: Hive Heartbeat Management
  As queen-rbee
  I want to receive heartbeats from hives
  So that I can track hive health and trigger device detection

  Scenario: First heartbeat triggers device detection
    Given the hive catalog contains a hive "localhost" with status "Unknown"
    When the hive "localhost" sends its first heartbeat
    Then the heartbeat should be acknowledged
    And the hive status should be updated to "Online"
    And device detection should be triggered
    And narration should contain "First heartbeat from localhost"
    And narration should contain "Checking capabilities"

  Scenario: Subsequent heartbeats do not trigger device detection
    Given the hive catalog contains a hive "localhost" with status "Online"
    When the hive "localhost" sends a heartbeat
    Then the heartbeat should be acknowledged
    And the hive status should remain "Online"
    And device detection should NOT be triggered
    And narration should NOT contain "First heartbeat"

  Scenario: Heartbeat from unknown hive is rejected
    Given the hive catalog is empty
    When an unknown hive "unknown-hive" sends a heartbeat
    Then the heartbeat should be rejected with 404
    And the error message should contain "Hive not found"

  Scenario: Heartbeat updates last_heartbeat timestamp
    Given the hive catalog contains a hive "localhost" with status "Online"
    And the hive "localhost" has no previous heartbeat
    When the hive "localhost" sends a heartbeat
    Then the hive "localhost" should have a last_heartbeat timestamp
    And the timestamp should be recent
