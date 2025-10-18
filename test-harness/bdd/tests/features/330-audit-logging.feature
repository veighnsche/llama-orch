# Traceability: RC-P1-AUDIT (Release Candidate P1 Audit Logging)
# Created by: TEAM-099
# Components: queen-rbee, rbee-hive, llm-worker-rbee
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: Audit Logging - Tamper-Evident Security Logs
  As a security-conscious system operator
  I want all security-relevant events logged with tamper-evident hash chains
  So that I can detect unauthorized modifications and maintain compliance

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And audit logging is enabled

  @p1 @audit @security
  Scenario: AUDIT-001 - Log worker spawn events with full context
    Given queen-rbee is running at "http://0.0.0.0:8080"
    When I spawn a worker with model "hf:test/model" on node "workstation"
    Then audit log contains event "worker.spawn"
    And audit entry includes timestamp
    And audit entry includes actor identity
    And audit entry includes worker_id
    And audit entry includes model_ref
    And audit entry includes node name
    And audit entry includes request correlation_id

  @p1 @audit @security
  Scenario: AUDIT-002 - Log authentication events (success and failure)
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "valid-token-123"
    When I send request with valid token "valid-token-123"
    Then audit log contains event "auth.success"
    And audit entry includes token fingerprint (not raw token)
    When I send request with invalid token "wrong-token-456"
    Then audit log contains event "auth.failure"
    And audit entry includes token fingerprint (not raw token)
    And audit entry includes failure reason

  @p1 @audit @tamper-evident
  Scenario: AUDIT-003 - Tamper-evident hash chain maintained
    Given queen-rbee is running with audit logging
    When 10 audit events are logged
    Then each audit entry includes previous_hash field
    And hash chain is valid (each hash matches previous entry)
    And first entry has previous_hash = "0000000000000000"
    And hash algorithm is SHA-256

  @p1 @audit @tamper-evident
  Scenario: AUDIT-004 - Detect log tampering
    Given queen-rbee has logged 5 audit events
    And audit log file exists at "/var/log/rbee/audit.log"
    When I modify audit entry #3 in the log file
    Then hash chain validation fails
    And tampered entry is identified as entry #3
    And all entries after #3 are flagged as potentially invalid

  @p1 @audit @format
  Scenario: AUDIT-005 - Log format is JSON structured
    Given queen-rbee is running with audit logging
    When an audit event is logged
    Then audit log entry is valid JSON
    And entry contains "timestamp" field (ISO 8601)
    And entry contains "event_type" field
    And entry contains "actor" field
    And entry contains "details" object
    And entry contains "previous_hash" field
    And entry contains "entry_hash" field

  @p1 @audit @rotation
  Scenario: AUDIT-006 - Log rotation preserves hash chain
    Given queen-rbee is running with audit logging
    And audit log rotation is configured at 10MB
    When audit log reaches 10MB
    Then new audit log file is created
    And first entry in new file includes previous_hash from last entry of old file
    And hash chain continues across files
    And old log file is archived with timestamp suffix

  @p1 @audit @disk-space
  Scenario: AUDIT-007 - Disk space monitoring for audit logs
    Given queen-rbee is running with audit logging
    And audit log directory has 100MB free space
    When audit logs consume 95MB
    Then queen-rbee logs warning "audit log disk space low"
    And queen-rbee continues logging (does not stop)
    When audit logs consume 99MB
    Then queen-rbee logs error "audit log disk space critical"
    And queen-rbee triggers log rotation

  @p1 @audit @correlation
  Scenario: AUDIT-008 - Log correlation IDs for request tracing
    Given queen-rbee is running with audit logging
    When I send inference request with correlation_id "req-12345"
    Then audit log contains event "inference.start" with correlation_id "req-12345"
    And audit log contains event "worker.assigned" with correlation_id "req-12345"
    And audit log contains event "inference.complete" with correlation_id "req-12345"
    And all related events share same correlation_id

  @p1 @audit @security @secrets
  Scenario: AUDIT-009 - Safe logging (no secrets in audit logs)
    Given queen-rbee is running with audit logging
    When authentication event is logged
    Then audit log does NOT contain raw API token
    And audit log does NOT contain passwords
    And audit log does NOT contain SSH keys
    And audit log contains token fingerprint only
    And audit log contains sanitized actor identity

  @p1 @audit @persistence
  Scenario: AUDIT-010 - Audit log persistence across restarts
    Given queen-rbee is running with audit logging
    When 5 audit events are logged
    And I restart queen-rbee
    Then previous audit log file is preserved
    And new audit log continues hash chain from previous file
    And all 5 previous events are still readable
    And hash chain validation passes across restart
