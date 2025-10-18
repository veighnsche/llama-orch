# Traceability: TEST-001 Phase 3b (worker binary provisioning)
# Architecture: TEAM-037 (queen-rbee orchestration)
# Components: rbee-hive (worker binary catalog and builder)
# Created by: TEAM-078
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ DO NOT use mock servers - wire up actual rbee-hive WorkerProvisioner

Feature: Worker Provisioning
  As a system managing worker binaries
  I want to build and catalog worker binaries from git
  So that rbee-hive can spawn workers with the correct features

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "workstation"
    And rbee-hive is running at "http://localhost:9200"
    And the worker catalog is SQLite at "~/.rbee/workers.db"

  Scenario: Build worker from git succeeds
    Given worker binary "llm-worker-rbee" not in catalog
    When rbee-hive builds worker from git with features ["cuda"]
    Then cargo build command is:
      """
      cargo build --release --bin llm-worker-rbee --features cuda
      """
    And the build succeeds
    And the binary is registered in catalog at "/target/release/llm-worker-rbee"
    And the catalog entry includes features ["cuda"]

  Scenario: Worker binary registered in catalog
    Given worker binary "llm-worker-rbee" built successfully
    When rbee-hive registers the worker in catalog
    Then the SQLite INSERT statement is:
      """
      INSERT INTO worker_binaries (id, worker_type, binary_path, features, built_at_unix)
      VALUES ('llm-worker-rbee-cuda', 'llm-worker-rbee', 
              '/target/release/llm-worker-rbee', '["cuda"]', 1728508603)
      """
    And the catalog query now returns the worker binary

  Scenario: Query available worker types
    Given the worker catalog contains:
      | worker_type           | features      | binary_path                       |
      | llm-worker-rbee       | ["cuda"]      | /target/release/llm-worker-rbee   |
      | llm-worker-rbee       | ["cpu"]       | /target/release/llm-worker-rbee   |
    When rbee-hive queries workers with features ["cuda"]
    Then the query returns 1 worker
    And the returned worker has features ["cuda"]

  Scenario: Build triggered when binary not found
    Given worker binary "llm-worker-rbee" not in catalog
    When rbee-hive attempts to spawn worker
    Then rbee-hive checks the worker catalog
    And the query returns no results
    And rbee-hive triggers worker build with features ["cuda"]
    And after build completes, rbee-hive spawns the worker

  @error-handling
  Scenario: EH-020a - Cargo build failure
    Given worker binary "llm-worker-rbee" not in catalog
    When rbee-hive builds worker from git
    And cargo build fails with compilation error
    Then rbee-hive captures stderr output
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: Worker build failed
        Worker: llm-worker-rbee
        Features: ["cuda"]
        
      Cargo error:
        error[E0425]: cannot find function `nonexistent_fn` in this scope
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-020b - Missing CUDA toolkit
    Given worker binary "llm-worker-rbee" not in catalog
    When rbee-hive builds worker with features ["cuda"]
    And CUDA toolkit is not installed
    Then cargo build fails with linker error
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: CUDA toolkit not found
        Worker: llm-worker-rbee
        Features: ["cuda"]
        
      Install CUDA toolkit:
        https://developer.nvidia.com/cuda-downloads
      """
    And the exit code is 1

  @error-handling
  Scenario: EH-020c - Binary verification failed
    Given worker binary built successfully
    When rbee-hive verifies the binary
    And the binary is not executable
    Then rbee-hive displays:
      """
      [rbee-hive] ❌ Error: Binary verification failed
        Binary: /target/release/llm-worker-rbee
        
      Binary is not executable. Check file permissions.
      """
    And the exit code is 1
