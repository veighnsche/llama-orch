# Created by: TEAM-159
# Tests device capability storage (CRUD operations)
# Device DETECTION happens on hive, queen only STORES the results

Feature: Device Capability Storage
  As queen-rbee
  I want to store device capabilities from hives
  So that I can use them for scheduling decisions

  Scenario: Store device capabilities from hive response
    Given the hive catalog contains a hive "localhost" with status "Unknown"
    And a mock hive server is running on port 8600
    And the mock hive responds with device capabilities
    When queen requests devices from the mock hive
    And queen receives the device response
    Then queen should store the device capabilities in the catalog
    And the hive "localhost" should have CPU with 8 cores and 32 GB RAM
    And the hive "localhost" should have 2 GPUs stored

  Scenario: Store CPU-only capabilities
    Given the hive catalog contains a hive "localhost" with status "Unknown"
    And a mock hive server returns CPU-only response
    When queen requests and stores the device capabilities
    Then the hive "localhost" should have CPU capabilities
    And the hive "localhost" should have 0 GPUs stored

  Scenario: Update existing device capabilities
    Given the hive catalog contains a hive "localhost" with status "Unknown"
    And the hive "localhost" already has device capabilities stored
    And a mock hive server returns different capabilities
    When queen requests and stores the new device capabilities
    Then the old capabilities should be replaced
    And the hive "localhost" should have the new capabilities
