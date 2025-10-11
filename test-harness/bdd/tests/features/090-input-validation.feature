# Traceability: TEST-001 (split by TEAM-077)
# Architecture: TEAM-037 (queen-rbee orchestration)
# Components: rbee-keeper, rbee-hive
# Refactored by: TEAM-077 (reorganized to correct BDD architecture)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: Input Validation
  As a system validating user inputs
  I want to detect invalid parameters and authentication issues
  So that I can fail fast with helpful error messages

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"

  @error-handling @validation
  Scenario: EH-015a - Invalid model reference format
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model invalid-format \
        --prompt "test"
      """
    Then rbee-keeper validates model reference format
    And validation fails
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: Invalid model reference format
        Got: invalid-format
        
      Expected formats:
        - hf:org/repo (Hugging Face)
        - file:///path/to/model.gguf (Local file)
        
      Example: hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
      """
    And the exit code is 1

  @error-handling @validation
  Scenario: EH-015b - Invalid backend name
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --backend quantum \
        --prompt "test"
      """
    Then rbee-keeper validates backend name
    And validation fails
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: Invalid backend
        Got: quantum
        
      Valid backends: ["cpu", "cuda", "metal"]
      
      Example: --backend cuda --device 0
      """
    And the exit code is 1

  @error-handling @validation
  Scenario: EH-015c - Device number out of range
    Given node "workstation" has 2 CUDA devices (0, 1)
    When I run:
      """
      rbee-keeper infer \
        --node workstation \
        --model hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
        --backend cuda \
        --device 5 \
        --prompt "test"
      """
    Then rbee-hive validates device number
    And validation fails
    And rbee-keeper displays:
      """
      [rbee-hive] ❌ Error: Device 5 not available
        Requested: cuda device 5
        Available: [0, 1]
        
      Try: --device 0 or --device 1
      """
    And the exit code is 1

  @error-handling @authentication
  Scenario: EH-017a - Missing API key
    Given rbee-hive requires API key
    When queen-rbee sends request without Authorization header
    Then rbee-hive returns:
      """
      HTTP/1.1 401 Unauthorized
      Content-Type: application/json

      {
        "error": {
          "code": "MISSING_API_KEY",
          "message": "Missing API key"
        }
      }
      """
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: Authentication required
        Missing API key for workstation.home.arpa
        
      Configure API key in:
        ~/.rbee/config.yaml
      """
    And the exit code is 1

  @error-handling @authentication
  Scenario: EH-017b - Invalid API key
    Given rbee-keeper uses API key "wrong_key"
    When rbee-keeper sends request with "Authorization: Bearer wrong_key"
    Then rbee-hive returns:
      """
      HTTP/1.1 401 Unauthorized
      Content-Type: application/json

      {
        "error": {
          "code": "INVALID_API_KEY",
          "message": "Invalid or missing API key"
        }
      }
      """
    And rbee-keeper displays:
      """
      [rbee-keeper] ❌ Error: Invalid API key for workstation.home.arpa
        
      Check API key in:
        ~/.rbee/config.yaml
      """
    And the exit code is 1

  Scenario: Error response structure validation
    Given an error occurs with code "VRAM_EXHAUSTED"
    When the error is returned to rbee-keeper
    Then the response format is:
      """
      {
        "error": {
          "code": "VRAM_EXHAUSTED",
          "message": "Insufficient VRAM: need 6000 MB, have 4000 MB",
          "details": {
            "required": 6000,
            "available": 4000
          }
        }
      }
      """
    And the error code is one of the defined error codes
    And the message is human-readable
    And the details provide actionable context
