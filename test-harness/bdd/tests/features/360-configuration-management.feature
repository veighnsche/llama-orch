# Traceability: RC-P2-CONFIG (Release Candidate P2 Configuration Management)
# Created by: TEAM-100 (THE CENTENNIAL TEAM! 💯🎉)
# Components: pool-managerd, queen-rbee, config-schema
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# 🎀 SPECIAL: TEAM-100 integrates narration-core for human-readable debugging!

Feature: Configuration Management - TOML Config with Hot-Reload
  As a system operator
  I want robust configuration management with validation and hot-reload
  So that I can update settings without downtime and catch errors early

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And narration capture is enabled

  @p2 @config @startup
  Scenario: CFG-001 - Load config from TOML file with narration
    Given a valid config file at "/etc/rbee/pool-managerd.toml"
    When pool-managerd starts
    Then config is loaded successfully
    And narration event is emitted with action "config_load"
    And narration human field contains "Loaded configuration from"
    And narration includes source_location field
    And narration correlation_id is present

  @p2 @config @validation
  Scenario: CFG-002 - Validate config on startup
    Given a config file with the following content:
      """
      [server]
      host = "0.0.0.0"
      port = 9090
      
      [pool]
      max_workers = 10
      heartbeat_interval_ms = 5000
      """
    When pool-managerd validates the config
    Then validation succeeds
    And narration event confirms "Configuration validated successfully"
    And narration includes validation duration_ms
    And no secrets are leaked in narration

  @p2 @config @hot-reload
  Scenario: CFG-003 - Hot-reload config (SIGHUP) with narration
    Given pool-managerd is running with config file "/etc/rbee/pool-managerd.toml"
    And current config has max_workers = 5
    When I update config file to set max_workers = 10
    And I send SIGHUP signal to pool-managerd
    Then config is reloaded without restart
    And new config has max_workers = 10
    And narration event is emitted with action "config_reload"
    And narration human field contains "Configuration reloaded"
    And narration correlation_id links reload events

  @p2 @config @env-override
  Scenario: CFG-004 - Environment variables override file config
    Given a config file with port = 9090
    And environment variable RBEE_POOL_PORT = "9091"
    When pool-managerd starts
    Then effective port is 9091
    And narration event explains "Config overridden by environment variable"
    And narration includes both file value and env value
    And narration redacts sensitive env vars

  @p2 @config @schema
  Scenario: CFG-005 - Config schema validation with detailed errors
    Given a config file with invalid schema:
      """
      [server]
      host = "0.0.0.0"
      port = "not-a-number"
      """
    When pool-managerd validates the config
    Then validation fails with error "port must be integer"
    And narration event includes error_kind "config_validation_failed"
    And narration human field explains validation error clearly
    And narration includes field name and expected type

  @p2 @config @startup-failure
  Scenario: CFG-006 - Invalid config fails startup with narration
    Given a config file with missing required field "server.host"
    When pool-managerd attempts to start
    Then startup fails with exit code 1
    And narration event is emitted with action "startup_failed"
    And narration error_kind is "config_invalid"
    And narration human field explains "Missing required field: server.host"
    And narration includes config file path

  @p2 @config @examples
  Scenario: CFG-007 - Config examples provided and validated
    Given example config files exist in "examples/config/"
    When I validate "examples/config/pool-managerd.example.toml"
    Then validation succeeds
    And example config includes all required fields
    And example config includes helpful comments
    And narration confirms "Example config validated successfully"

  @p2 @config @narration
  Scenario: CFG-008 - Narration events for config load/reload
    Given pool-managerd is running
    When config is loaded
    Then narration event is emitted with actor "pool-managerd"
    And narration event is emitted with action "config_load"
    And narration human field contains config file path
    And narration emitted_by field contains "pool-managerd@"
    When config is reloaded
    Then narration event is emitted with action "config_reload"
    And narration correlation_id links load and reload events

  @p2 @config @secrets
  Scenario: CFG-009 - Secret redaction in config logs
    Given a config file with sensitive fields:
      """
      [auth]
      api_key = "secret-key-123"
      bearer_token = "Bearer abc123xyz"
      """
    When pool-managerd loads the config
    Then narration events do not contain "secret-key-123"
    And narration events do not contain "abc123xyz"
    And narration events contain "[REDACTED]" for sensitive fields
    And narration human field says "Loaded auth config (secrets redacted)"
    And Bearer tokens are redacted
    And API keys are redacted

  @p2 @config @cute-mode
  Scenario: CFG-010 - Cute mode for config validation errors
    Given pool-managerd is running with cute mode enabled
    And a config file with invalid port value
    When config validation fails
    Then narration cute field is present
    And narration cute field describes error whimsically
    And example cute message: "Oh no! The port number doesn't look right! 😟"
    And narration cute field is under 150 characters
    And narration cute field includes emoji
