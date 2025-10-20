# Created by: TEAM-156
# Tests hive catalog checking and "no hives found" narration

Feature: Hive Catalog Management
  As queen-rbee
  I want to check the hive catalog for available hives
  So that I can route jobs appropriately

  Scenario: No hives found on clean install
    Given the hive catalog is empty
    When I submit a job to queen-rbee
    Then the SSE stream should contain "No hives found."
    And the job should complete with [DONE]

  Scenario: Hive catalog is initialized
    Given queen-rbee starts with a clean database
    Then the hive catalog should be created
    And the hive catalog should be empty
