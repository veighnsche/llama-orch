# Traceability: TEST-001 (split by TEAM-077)
# Architecture: TEAM-037 (queen-rbee orchestration)
# Components: queen-rbee (orchestrator daemon), rbee-keeper (CLI)
# Refactored by: TEAM-077 (reorganized to correct BDD architecture)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: queen-rbee Daemon Lifecycle
  As a system managing the queen-rbee orchestrator daemon
  I want to control deployment modes and lifecycle
  So that queen-rbee can run in ephemeral or persistent mode

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"

  @lifecycle @critical
  Scenario: rbee-keeper exits after inference (CLI dies, daemons live)
    Given queen-rbee is running at "http://localhost:8080"
    And rbee-hive is running as persistent daemon
    And worker is running and idle
    When rbee-keeper completes inference request
    Then rbee-keeper exits with code 0
    And queen-rbee continues running as daemon
    And rbee-hive continues running as daemon
    And worker continues running as daemon
    And worker remains in rbee-hive's in-memory registry

  @lifecycle @critical
  Scenario: Ephemeral mode - rbee-keeper spawns queen-rbee
    Given rbee-keeper is configured to spawn queen-rbee
    When rbee-keeper runs inference command
    Then rbee-keeper spawns queen-rbee as child process
    And queen-rbee starts HTTP daemon
    And queen-rbee spawns rbee-hive via SSH
    And rbee-hive spawns worker
    And inference completes
    And rbee-keeper sends SIGTERM to queen-rbee
    And queen-rbee cascades shutdown to all rbee-hive instances via SSH
    And rbee-hive cascades shutdown to worker
    And worker exits
    And rbee-hive exits
    And queen-rbee exits
    And rbee-keeper exits with code 0

  @lifecycle @critical
  Scenario: Persistent mode - queen-rbee pre-started
    Given queen-rbee is already running as daemon at "http://localhost:8080"
    And queen-rbee was started manually by operator
    And rbee-hive is already running as daemon
    And rbee-hive was started by queen-rbee via SSH
    When rbee-keeper runs inference command
    Then rbee-keeper connects to existing queen-rbee HTTP API
    And rbee-keeper does NOT spawn queen-rbee
    And inference completes
    And rbee-keeper exits
    And queen-rbee continues running (was not spawned by rbee-keeper)
    And rbee-hive continues running (was not spawned by rbee-keeper)
    And worker continues running (idle timeout not reached)
