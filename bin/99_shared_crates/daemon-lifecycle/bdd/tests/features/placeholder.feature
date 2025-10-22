# TEAM-TESTING: Daemon Lifecycle BDD Tests
# Purpose: Verify daemon spawning, lifecycle, and critical Stdio::null() behavior

Feature: Daemon Lifecycle Management
  As a developer
  I want to spawn and manage daemon processes
  So that rbee-keeper, queen-rbee, and rbee-hive can manage their child processes

  # CRITICAL: Stdio::null() tests (TEAM-164 fix)
  # These tests verify that spawned daemons don't hold parent's pipes open
  # Without this fix, E2E tests hang indefinitely

  Scenario: Daemon doesn't hold parent's stdout pipe
    Given I have a daemon binary to spawn
    When I spawn the daemon using DaemonManager
    Then the daemon should not hold the parent's stdout pipe
    And the parent should be able to exit immediately

  Scenario: Daemon doesn't hold parent's stderr pipe
    Given I have a daemon binary to spawn
    When I spawn the daemon using DaemonManager
    Then the daemon should not hold the parent's stderr pipe
    And the parent should be able to exit immediately

  Scenario: Command::output() doesn't hang with spawned daemon
    Given I have a daemon binary to spawn
    When I spawn the daemon and capture output
    Then Command::output() should complete within 5 seconds
    And the parent should not hang waiting for pipes

  Scenario: SSH_AUTH_SOCK is propagated to daemon
    Given SSH_AUTH_SOCK environment variable is set
    When I spawn the daemon using DaemonManager
    Then the daemon should receive the SSH_AUTH_SOCK environment variable
    And the daemon can use SSH agent for authentication

  Scenario: Daemon spawn with missing binary fails gracefully
    Given I specify a non-existent binary path
    When I attempt to spawn the daemon
    Then the spawn should fail with an error
    And the error should indicate the binary was not found

  Scenario: Daemon spawn returns valid PID
    Given I have a daemon binary to spawn
    When I spawn the daemon using DaemonManager
    Then the daemon should have a valid PID
    And the PID should be greater than 0

  # Binary resolution tests
  Scenario: Find binary in target/debug directory
    Given the queen-rbee binary exists in target/debug
    When I call find_in_target("queen-rbee")
    Then the function should return the debug binary path
    And the path should exist

  Scenario: Find binary in target/release directory
    Given the queen-rbee binary exists in target/release
    When I call find_in_target("queen-rbee")
    Then the function should return the release binary path
    And the path should exist

  Scenario: Find binary error for missing binary
    Given the binary "nonexistent-xyz" does not exist
    When I call find_in_target("nonexistent-xyz")
    Then the function should return an error
    And the error should indicate the binary was not found

  # Concurrent spawn tests (reasonable scale: 5 concurrent)
  Scenario: Spawn 5 daemons concurrently
    Given I have 5 daemon binaries to spawn
    When I spawn all 5 daemons concurrently
    Then all 5 daemons should spawn successfully
    And each daemon should have a unique PID
    And no resource conflicts should occur

  # Binary path edge cases
  Scenario: Daemon spawn with absolute path
    Given I have an absolute path to a daemon binary
    When I spawn the daemon using the absolute path
    Then the daemon should spawn successfully
    And the binary should be executed from the specified path

  Scenario: Daemon spawn with relative path
    Given I have a relative path to a daemon binary
    When I spawn the daemon using the relative path
    Then the daemon should spawn successfully
    And the binary should be resolved correctly

  Scenario: Daemon spawn with symlink path
    Given I have a symlink to a daemon binary
    When I spawn the daemon using the symlink path
    Then the daemon should spawn successfully
    And the symlink should be followed correctly
