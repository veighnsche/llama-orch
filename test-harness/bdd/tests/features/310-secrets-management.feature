# Traceability: RC-P0-SECRETS (Release Candidate P0 Secrets Management)
# Created by: TEAM-097
# Components: queen-rbee, rbee-hive, llm-worker-rbee
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: Secrets Management
  As a security-conscious system
  I want to load secrets from files with proper permissions
  So that credentials are never exposed in logs or memory dumps

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"

  @p0 @secrets @security
  Scenario: SEC-001 - Load API token from file with 0600 permissions
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600" (owner read/write only)
    And file contains "test-token-12345"
    When queen-rbee starts with config:
      """
      api_token_file: /tmp/llorch-test/api-token
      bind: 0.0.0.0:8080
      """
    Then queen-rbee starts successfully
    And API token is loaded from file
    And token is stored in memory with zeroization
    And log contains "API token loaded from file"
    And log does not contain "test-token-12345"

  @p0 @secrets @security
  Scenario: SEC-002 - Reject world-readable secret file (0644)
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0644" (world-readable)
    And file contains "test-token-12345"
    When queen-rbee starts with config:
      """
      api_token_file: /tmp/llorch-test/api-token
      bind: 0.0.0.0:8080
      """
    Then queen-rbee fails to start
    And displays error: "Secret file has insecure permissions: 0644"
    And displays error: "Expected: 0600 (owner read/write only)"
    And exit code is 1

  @p0 @secrets @security
  Scenario: SEC-003 - Reject group-readable secret file (0640)
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0640" (group-readable)
    And file contains "test-token-12345"
    When queen-rbee starts with config:
      """
      api_token_file: /tmp/llorch-test/api-token
      bind: 0.0.0.0:8080
      """
    Then queen-rbee fails to start
    And displays error: "Secret file has insecure permissions: 0640"
    And displays error: "Expected: 0600 (owner read/write only)"
    And exit code is 1

  @p0 @secrets @systemd
  Scenario: SEC-004 - Load from systemd credentials (/run/credentials/)
    Given systemd credential exists at "/run/credentials/queen-rbee/api_token"
    And file permissions are "0400" (owner read-only)
    And file contains "systemd-token-12345"
    When queen-rbee starts with systemd credential "api_token"
    Then queen-rbee starts successfully
    And API token is loaded from systemd credential
    And log contains "API token loaded from systemd credential"
    And log does not contain "systemd-token-12345"

  @p0 @secrets @memory-safety
  Scenario: SEC-005 - Memory zeroization on drop (not in memory dump)
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600"
    And file contains "zeroize-test-token-12345"
    When queen-rbee starts and loads API token
    And I trigger garbage collection
    And I capture memory dump
    Then memory dump does not contain "zeroize-test-token-12345"
    And secret memory is zeroed after use

  @p0 @secrets @crypto
  Scenario: SEC-006 - Derive encryption key from token (HKDF-SHA256)
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600"
    And file contains "master-token-12345"
    When queen-rbee derives encryption key from API token
    Then encryption key is derived using HKDF-SHA256
    And key derivation uses salt "llama-orch-encryption-v1"
    And derived key is 32 bytes (256 bits)
    And derived key is different from API token
    And log contains "Encryption key derived"
    And log does not contain derived key

  @p0 @secrets @validation
  Scenario: SEC-007 - Secret file must exist or fail to start
    Given API token file does not exist at "/tmp/llorch-test/nonexistent"
    When queen-rbee starts with config:
      """
      api_token_file: /tmp/llorch-test/nonexistent
      bind: 0.0.0.0:8080
      """
    Then queen-rbee fails to start
    And displays error: "API token file not found: /tmp/llorch-test/nonexistent"
    And exit code is 1

  @p0 @secrets @validation
  Scenario: SEC-008 - Secret file must be readable or fail to start
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0000" (no permissions)
    When queen-rbee starts with config:
      """
      api_token_file: /tmp/llorch-test/api-token
      bind: 0.0.0.0:8080
      """
    Then queen-rbee fails to start
    And displays error: "Cannot read API token file: Permission denied"
    And exit code is 1

  @p0 @secrets @logging
  Scenario: SEC-009 - Secrets never appear in logs
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600"
    And file contains "secret-never-log-this-12345"
    When queen-rbee starts and processes 100 requests
    And I search all log files for "secret-never-log-this-12345"
    Then no log file contains the raw secret
    And all log entries use token fingerprints only

  @p0 @secrets @error-handling
  Scenario: SEC-010 - Secrets never appear in error messages
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600"
    And file contains "secret-error-test-12345"
    When queen-rbee encounters error loading secret
    Then error message does not contain "secret-error-test-12345"
    And error message contains "Failed to load API token"
    And error message contains file path only

  @p0 @secrets @timing-attack
  Scenario: SEC-011 - Timing-safe secret verification
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600"
    And file contains "timing-safe-token-12345"
    When I send 100 requests with correct token
    And I send 100 requests with incorrect token (same length)
    Then verification time variance is < 10%
    And no timing side-channel is detectable

  @p0 @secrets @hot-reload
  Scenario: SEC-012 - Secret rotation without restart (SIGHUP)
    Given queen-rbee is running with API token "old-token-12345"
    And API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600"
    When I update file to contain "new-token-67890"
    And I send SIGHUP to queen-rbee process
    Then queen-rbee reloads API token from file
    And requests with "old-token-12345" are rejected with 401
    And requests with "new-token-67890" are accepted with 200
    And log contains "API token reloaded"
    And log does not contain "old-token-12345" or "new-token-67890"

  @p0 @secrets @multi-component
  Scenario: SEC-013 - Each component has separate secret file
    Given queen-rbee token file exists at "/tmp/llorch-test/queen-token"
    And rbee-hive token file exists at "/tmp/llorch-test/hive-token"
    And llm-worker-rbee token file exists at "/tmp/llorch-test/worker-token"
    And all files have permissions "0600"
    When all components start
    Then queen-rbee loads from "/tmp/llorch-test/queen-token"
    And rbee-hive loads from "/tmp/llorch-test/hive-token"
    And llm-worker-rbee loads from "/tmp/llorch-test/worker-token"
    And each component has different token
    And no token is shared between components

  @p0 @secrets @file-format
  Scenario: SEC-014 - Secret file format validation
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600"
    And file contains "valid-token-12345\n"
    When queen-rbee loads API token
    Then trailing newline is stripped
    And token is "valid-token-12345" (no newline)
    And token is valid for authentication

  @p0 @secrets @file-format
  Scenario: SEC-015 - Reject secret file with multiple lines
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600"
    And file contains:
      """
      line1-token
      line2-token
      """
    When queen-rbee starts
    Then queen-rbee fails to start
    And displays error: "API token file must contain exactly one line"
    And exit code is 1

  @p0 @secrets @validation
  Scenario: SEC-016 - Reject empty secret file
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600"
    And file is empty
    When queen-rbee starts
    Then queen-rbee fails to start
    And displays error: "API token file is empty"
    And exit code is 1

  @p0 @secrets @validation
  Scenario: SEC-017 - Reject secret file with only whitespace
    Given API token file exists at "/tmp/llorch-test/api-token"
    And file permissions are "0600"
    And file contains "   \n\t  \n"
    When queen-rbee starts
    Then queen-rbee fails to start
    And displays error: "API token file contains only whitespace"
    And exit code is 1
