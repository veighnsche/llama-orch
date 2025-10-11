# Traceability: TEST-001 Phase 2a (SSH preflight checks)
# Architecture: TEAM-037 (queen-rbee orchestration)
# Components: queen-rbee (SSH connection validation)
# Created by: TEAM-078
# Stakeholder: DevOps / SSH operations
# Timing: Phase 2a (before starting rbee-hive)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ DO NOT use mock servers - wire up actual queen-rbee SSH preflight checker

Feature: SSH Preflight Validation
  As a DevOps engineer
  I want to validate SSH connectivity before starting rbee-hive
  So that I can detect connection issues early

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

  Scenario: SSH connection validation succeeds
    Given SSH credentials are configured for "workstation.home.arpa"
    When queen-rbee validates SSH connection
    Then SSH connection to "workstation.home.arpa" succeeds
    And queen-rbee logs "SSH preflight: connection OK"
    And preflight check passes

  @error-handling
  Scenario: EH-001a - SSH connection timeout
    Given SSH host "unreachable.example.com" is unreachable
    When queen-rbee validates SSH connection
    And connection times out after 10 seconds
    Then queen-rbee detects timeout
    And rbee-keeper displays:
      """
      [queen-rbee] ❌ Error: SSH connection timeout
        Host: unreachable.example.com
        Timeout: 10s
        
      Check network connectivity:
        ping unreachable.example.com
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-001b - SSH authentication failure
    Given SSH credentials are invalid for "workstation.home.arpa"
    When queen-rbee validates SSH connection
    Then SSH authentication fails
    And rbee-keeper displays:
      """
      [queen-rbee] ❌ Error: SSH authentication failed
        Host: workstation.home.arpa
        
      Check SSH key:
        ssh-add -l
        ssh workstation.home.arpa
      """
    And the exit code is 1

  Scenario: SSH command execution test
    Given SSH connection to "workstation.home.arpa" is established
    When queen-rbee executes test command "echo test"
    Then the command succeeds
    And stdout is "test"
    And queen-rbee logs "SSH preflight: command execution OK"

  Scenario: Network latency check
    Given SSH connection to "workstation.home.arpa" is established
    When queen-rbee measures SSH round-trip time
    Then the latency is less than 100ms
    And queen-rbee logs "SSH preflight: latency 15ms (OK)"

  Scenario: rbee-hive binary exists on remote node
    Given SSH connection to "workstation.home.arpa" is established
    When queen-rbee checks for rbee-hive binary
    Then the command "which rbee-hive" succeeds
    And stdout is "/usr/local/bin/rbee-hive"
    And queen-rbee logs "SSH preflight: rbee-hive binary found"
