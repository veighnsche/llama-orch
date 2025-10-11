# Traceability: TEST-001 (split by TEAM-077)
# Architecture: TEAM-037 (queen-rbee orchestration, in-memory worker registry, SQLite model catalog)
# Components: rbee-keeper (config + testing tool), queen-rbee (orchestrator), rbee-hive (pool manager)
# Updated by: TEAM-038 (aligned with queen-rbee orchestration and GGUF support)
# Updated by: TEAM-041 (added rbee-hive Registry module, SSH setup flow, rbee-keeper configuration mode)
# Refactored by: TEAM-077 (split from test-001.feature into focused feature files)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ DO NOT use mock servers - wire up actual rbee-hive and llm-worker-rbee libraries
# ⚠️ See TEAM_063_REAL_HANDOFF.md for implementation requirements

Feature: SSH Registry Management
  As a user setting up distributed inference
  I want to register remote nodes with SSH details
  So that queen-rbee can manage remote rbee-hive instances

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    # NOTE: This test suite uses workstation node with cuda backend on device 1
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"
    And the model catalog is SQLite at "~/.rbee/models.db"
    And the worker registry is in-memory ephemeral per node
    And the rbee-hive registry is SQLite at "~/.rbee/beehives.db"

  # ============================================================================
  # PREREQUISITES: rbee-hive Registry Setup (TEAM-041)
  # ============================================================================
  # CRITICAL: Before any inference can happen, remote rbee-hive nodes must be
  # configured through rbee-keeper. The queen-rbee maintains a persistent
  # rbee-hive Registry with SSH connection details.
  # ============================================================================

  @setup @critical
  Scenario: Add remote rbee-hive node to registry
    Given queen-rbee is running
    And the rbee-hive registry is empty
    When I run:
      """
      rbee-keeper setup add-node \
        --name workstation \
        --ssh-host workstation.home.arpa \
        --ssh-user vince \
        --ssh-key ~/.ssh/id_ed25519 \
        --git-repo https://github.com/user/llama-orch.git \
        --git-branch main \
        --install-path ~/rbee
      """
    Then rbee-keeper sends request to queen-rbee at "http://localhost:8080/v2/registry/beehives/add"
    And queen-rbee validates SSH connection with:
      """
      ssh -i /home/vince/.ssh/id_ed25519 vince@workstation.home.arpa "echo 'connection test'"
      """
    And the SSH connection succeeds
    And queen-rbee saves node to rbee-hive registry:
      | field              | value                                      |
      | node_name          | workstation                                |
      | ssh_host           | workstation.home.arpa                      |
      | ssh_port           | 22                                         |
      | ssh_user           | vince                                      |
      | ssh_key_path       | /home/vince/.ssh/id_ed25519                |
      | git_repo_url       | https://github.com/user/llama-orch.git     |
      | git_branch         | main                                       |
      | install_path       | /home/vince/rbee                           |
      | last_connected_unix| 1728508603                                 |
      | status             | reachable                                  |
    And rbee-keeper displays:
      """
      [queen-rbee] 🔌 Testing SSH connection to workstation.home.arpa
      [queen-rbee] ✅ SSH connection successful! Node 'workstation' saved to registry
      """
    And the exit code is 0

  @setup @error-handling
  Scenario: EH-001a - SSH connection timeout
    When I run:
      """
      rbee-keeper setup add-node \
        --name unreachable \
        --ssh-host unreachable.home.arpa \
        --ssh-user vince \
        --ssh-key ~/.ssh/id_ed25519 \
        --git-repo https://github.com/user/llama-orch.git \
        --install-path ~/rbee
      """
    Then queen-rbee attempts SSH connection with 10s timeout
    And the SSH connection fails with timeout
    And queen-rbee retries 3 times with exponential backoff
    And queen-rbee does NOT save node to registry
    And rbee-keeper displays:
      """
      [queen-rbee] 🔌 Testing SSH connection to unreachable.home.arpa
      [queen-rbee] ⏳ Attempt 1/3 failed: Connection timeout
      [queen-rbee] ⏳ Attempt 2/3 failed: Connection timeout (delay 200ms)
      [queen-rbee] ⏳ Attempt 3/3 failed: Connection timeout (delay 400ms)
      [queen-rbee] ❌ SSH connection failed after 3 attempts
      """
    And the exit code is 1

  @setup @error-handling
  Scenario: EH-001b - SSH authentication failure
    Given SSH key at "~/.ssh/id_ed25519" has wrong permissions
    When I run:
      """
      rbee-keeper setup add-node \
        --name workstation \
        --ssh-host workstation.home.arpa \
        --ssh-user vince \
        --ssh-key ~/.ssh/id_ed25519 \
        --git-repo https://github.com/user/llama-orch.git \
        --install-path ~/rbee
      """
    Then queen-rbee attempts SSH connection
    And the SSH connection fails with "Permission denied (publickey)"
    And queen-rbee does NOT save node to registry
    And rbee-keeper displays:
      """
      [queen-rbee] 🔌 Testing SSH connection to workstation.home.arpa
      [queen-rbee] ❌ SSH authentication failed: Permission denied
      
      Suggestion: Check SSH key permissions:
        chmod 600 ~/.ssh/id_ed25519
      """
    And the exit code is 1

  @setup @error-handling
  Scenario: EH-001c - SSH command execution failure
    Given SSH connection succeeds
    But rbee-hive binary does not exist on remote node
    When queen-rbee attempts to start rbee-hive via SSH
    Then the SSH command fails with "command not found"
    And rbee-keeper displays:
      """
      [queen-rbee] ❌ Failed to start rbee-hive: command not found
      
      Suggestion: Install rbee-hive on workstation:
        rbee-keeper setup install --node workstation
      """
    And the exit code is 1

  @setup
  Scenario: Install rbee-hive on remote node
    Given node "workstation" is registered in rbee-hive registry
    When I run:
      """
      rbee-keeper setup install --node workstation
      """
    Then queen-rbee loads SSH details from registry
    And queen-rbee executes installation via SSH:
      """
      ssh -i /home/vince/.ssh/id_ed25519 vince@workstation.home.arpa << 'EOF'
        cd /home/vince/rbee
        git clone https://github.com/user/llama-orch.git .
        git checkout main
        cargo build --release --bin rbee-hive
        cargo build --release --bin llm-worker-rbee
      EOF
      """
    And rbee-keeper displays:
      """
      [queen-rbee] 📦 Cloning repository on workstation
      [queen-rbee] 🔨 Building rbee-hive and llm-worker-rbee
      [queen-rbee] ✅ Installation complete on workstation!
      """
    And the exit code is 0

  @setup
  Scenario: List registered rbee-hive nodes
    Given multiple nodes are registered in rbee-hive registry
    When I run "rbee-keeper setup list-nodes"
    Then rbee-keeper displays:
      """
      Registered rbee-hive Nodes:
      
      workstation (workstation.home.arpa)
        Status: reachable
        Last connected: 2024-10-09 14:22:15
        Install path: /home/vince/rbee
      
      blep (blep.home.arpa)
        Status: reachable
        Last connected: 2024-10-09 14:22:15
        Install path: /home/vince/rbee
      """
    And the exit code is 0

  @setup
  Scenario: Remove node from rbee-hive registry
    Given node "workstation" is registered in rbee-hive registry
    When I run "rbee-keeper setup remove-node --name workstation"
    Then queen-rbee removes node from registry
    And rbee-keeper displays:
      """
      [queen-rbee] ✅ Node 'workstation' removed from registry
      """
    And the exit code is 0

  @setup @error-handling
  Scenario: EH-011a - Invalid SSH key path
    When I run:
      """
      rbee-keeper setup add-node \
        --name workstation \
        --ssh-host workstation.home.arpa \
        --ssh-user vince \
        --ssh-key /nonexistent/key \
        --git-repo https://github.com/user/llama-orch.git \
        --install-path ~/rbee
      """
    Then rbee-keeper validates SSH key path before sending to queen-rbee
    And validation fails with "SSH key not found: /nonexistent/key"
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: SSH key not found
        Path: /nonexistent/key
        
      Check the key path and try again.
      """
    And the exit code is 1

  @setup @error-handling
  Scenario: EH-011b - Duplicate node name
    Given node "workstation" already exists in registry
    When I run:
      """
      rbee-keeper setup add-node \
        --name workstation \
        --ssh-host workstation2.home.arpa \
        --ssh-user vince \
        --ssh-key ~/.ssh/id_ed25519 \
        --git-repo https://github.com/user/llama-orch.git \
        --install-path ~/rbee
      """
    Then queen-rbee detects duplicate node name
    And rbee-keeper displays:
      """
      [queen-rbee] ❌ Error: Node 'workstation' already exists in registry
      
      To update this node, run:
        rbee-keeper setup update-node --name workstation ...
      
      To remove and re-add:
        rbee-keeper setup remove-node --name workstation
      """
    And the exit code is 1

  @setup @critical
  Scenario: Inference fails when node not in registry
    Given the rbee-hive registry does not contain node "workstation"
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --prompt "test"
      """
    Then queen-rbee queries rbee-hive registry for node "workstation"
    And the query returns no results
    And rbee-keeper displays:
      """
      [queen-rbee] ❌ ERROR: Node 'workstation' not found in rbee-hive registry
      
      To add this node, run:
        rbee-keeper setup add-node --name workstation --ssh-host workstation.home.arpa ...
      """
    And the exit code is 1
