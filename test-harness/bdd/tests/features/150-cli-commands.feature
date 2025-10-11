# Traceability: TEST-001 (split by TEAM-077)
# Architecture: TEAM-037 (queen-rbee orchestration)
# Components: rbee-keeper
# Updated by: TEAM-036 (installation system)
# Refactored by: TEAM-077 (reorganized to correct BDD architecture)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: CLI Commands
  As a user of the rbee-keeper CLI
  I want to execute commands for installation, configuration, and management
  So that I can control the system from the command line

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

  @install @team-036
  Scenario: CLI command - install to user paths
    When I run "rbee-keeper install"
    Then binaries are installed to "~/.local/bin/"
    And config directory is created at "~/.config/rbee/"
    And data directory is created at "~/.local/share/rbee/models/"
    And default config file is generated at "~/.config/rbee/config.toml"
    And the following binaries are copied:
      | binary           | source                          | destination                  |
      | rbee-keeper      | target/release/rbee-keeper      | ~/.local/bin/rbee-keeper     |
      | rbee-hive        | target/release/rbee-hive        | ~/.local/bin/rbee-hive       |
      | llm-worker-rbee  | target/release/llm-worker-rbee  | ~/.local/bin/llm-worker-rbee |
    And installation instructions are displayed
    And the exit code is 0

  @install @team-036
  Scenario: CLI command - install to system paths
    When I run "rbee-keeper install --system"
    Then binaries are installed to "/usr/local/bin/"
    And config directory is created at "/etc/rbee/"
    And data directory is created at "/var/lib/rbee/models/"
    And default config file is generated at "/etc/rbee/config.toml"
    And sudo permissions are required
    And the exit code is 0

  @install @team-036
  Scenario: Config file loading with XDG priority
    Given the following config files exist:
      | path                           | priority |
      | /tmp/custom.toml               | 1 (RBEE_CONFIG env var) |
      | ~/.config/rbee/config.toml     | 2 (user config) |
      | /etc/rbee/config.toml          | 3 (system config) |
    When RBEE_CONFIG="/tmp/custom.toml" is set
    Then rbee-keeper loads config from "/tmp/custom.toml"
    When RBEE_CONFIG is not set
    And "~/.config/rbee/config.toml" exists
    Then rbee-keeper loads config from "~/.config/rbee/config.toml"
    When neither RBEE_CONFIG nor user config exist
    Then rbee-keeper loads config from "/etc/rbee/config.toml"

  @install @team-036
  Scenario: Remote binary path configuration
    Given config file contains:
      """
      [remote]
      binary_path = "/opt/rbee/bin/rbee-hive"
      git_repo_dir = "/opt/rbee/repo"
      """
    When rbee-keeper executes remote command on "workstation"
    Then the command uses "/opt/rbee/bin/rbee-hive" instead of "rbee-hive"
    And git commands use "/opt/rbee/repo" instead of "~/llama-orch"

  Scenario: CLI command - basic inference
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --prompt "write a short story" \
        --max-tokens 20 \
        --temperature 0.7 \
        --backend cuda \
        --device 1
      """
    Then the command executes the full inference flow
    And tokens are streamed to stdout
    And the exit code is 0

  Scenario: CLI command - list workers
    Given workers are registered on multiple nodes
    When I run "rbee-keeper workers list"
    Then the output shows all registered workers with their state
    And the exit code is 0

  Scenario: CLI command - check worker health
    When I run "rbee-keeper workers health --node workstation"
    Then the output shows health status of workers on workstation
    And the exit code is 0

  Scenario: CLI command - manually shutdown worker
    Given a worker with id "worker-abc123" is running
    When I run "rbee-keeper workers shutdown --id worker-abc123"
    Then the worker receives shutdown command
    And the worker unloads model and exits
    And the exit code is 0

  Scenario: CLI command - view logs
    When I run "rbee-keeper logs --node workstation --follow"
    Then logs from workstation are streamed to stdout
    And the stream continues until Ctrl+C
    And the exit code is 0 or 130
