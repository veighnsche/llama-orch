# Traceability: TEST-001 Phase 3a (rbee-hive preflight checks)
# Architecture: TEAM-037 (queen-rbee orchestration)
# Components: rbee-hive (HTTP API health checks)
# Created by: TEAM-078
# Stakeholder: Platform readiness team
# Timing: Phase 3a (before spawning workers)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ DO NOT use mock servers - wire up actual rbee-hive preflight checker

Feature: rbee-hive Preflight Validation
  As a platform engineer
  I want to validate rbee-hive readiness before spawning workers
  So that I can ensure the platform is ready for workloads

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"
    And rbee-hive is running at "http://workstation.home.arpa:9200"

  Scenario: rbee-hive HTTP API health check succeeds
    Given rbee-hive is running
    When queen-rbee checks rbee-hive health endpoint
    Then health endpoint returns 200 OK
    And response body is:
      """
      {
        "status": "healthy",
        "version": "0.1.0"
      }
      """
    And queen-rbee logs "rbee-hive preflight: health check OK"

  Scenario: rbee-hive version compatibility check
    Given rbee-hive is running with version "0.1.0"
    And queen-rbee requires version ">=0.1.0"
    When queen-rbee validates version compatibility
    Then version check passes
    And queen-rbee logs "rbee-hive preflight: version 0.1.0 compatible"

  Scenario: Backend catalog populated
    Given rbee-hive is running
    When queen-rbee queries available backends
    Then the response contains detected backends:
      | backend | available |
      | cuda    | true      |
      | cpu     | true      |
    And queen-rbee logs "rbee-hive preflight: backends detected (cuda, cpu)"

  Scenario: Sufficient resources available
    Given rbee-hive is running
    When queen-rbee queries available resources
    Then the response contains:
      """
      {
        "ram_total_gb": 32,
        "ram_available_gb": 24,
        "disk_total_gb": 500,
        "disk_available_gb": 400
      }
      """
    And ram_available_gb >= 8
    And disk_available_gb >= 50
    And queen-rbee logs "rbee-hive preflight: resources sufficient"
