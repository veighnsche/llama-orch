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

  @p0 @validation @log-injection @security
  Scenario: VAL-001 - Prevent log injection with newlines
    Given queen-rbee is running at "http://localhost:8080"
    When I send POST to "/v1/workers/spawn" with model_ref "hf:test\nINJECTED LOG LINE"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid model reference: contains newline characters"
    And log file does not contain "INJECTED LOG LINE" on separate line
    And validation error explains expected format

  @p0 @validation @log-injection @security
  Scenario: VAL-002 - Prevent log injection with ANSI escape codes
    Given queen-rbee is running at "http://localhost:8080"
    When I send POST to "/v1/workers/spawn" with model_ref "hf:test\x1b[31mRED TEXT\x1b[0m"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid model reference: contains ANSI escape codes"
    And log file does not contain ANSI escape sequences

  @p0 @validation @log-injection @security
  Scenario: VAL-003 - Prevent log injection with null bytes
    Given queen-rbee is running at "http://localhost:8080"
    When I send POST to "/v1/workers/spawn" with model_ref "hf:test\x00injected"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid model reference: contains null bytes"

  @p0 @validation @path-traversal @security
  Scenario: VAL-004 - Prevent path traversal (../../etc/passwd)
    Given rbee-hive is running at "http://localhost:8081"
    When I send request with model_path "../../etc/passwd"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid path: path traversal detected"
    And file system access is blocked

  @p0 @validation @path-traversal @security
  Scenario: VAL-005 - Prevent path traversal (absolute paths)
    Given rbee-hive is running at "http://localhost:8081"
    When I send request with model_path "/etc/passwd"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid path: absolute paths not allowed"

  @p0 @validation @path-traversal @security
  Scenario: VAL-006 - Prevent path traversal (symlinks)
    Given rbee-hive is running at "http://localhost:8081"
    And symlink exists at "/tmp/llorch-test/evil-link" pointing to "/etc/passwd"
    When I send request with model_path "/tmp/llorch-test/evil-link"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid path: symlinks not allowed"
    And symlink is not followed

  @p0 @validation @command-injection @security
  Scenario: VAL-007 - Prevent command injection (shell metacharacters)
    Given rbee-hive is running at "http://localhost:8081"
    When I send request with worker_id "worker-123; rm -rf /"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid worker ID: contains shell metacharacters"
    And no shell command is executed

  @p0 @validation @command-injection @security
  Scenario: VAL-008 - Prevent command injection (backticks)
    Given rbee-hive is running at "http://localhost:8081"
    When I send request with worker_id "worker-`whoami`"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid worker ID: contains backticks"

  @p0 @validation @command-injection @security
  Scenario: VAL-009 - Prevent command injection (pipe characters)
    Given rbee-hive is running at "http://localhost:8081"
    When I send request with worker_id "worker-123 | cat /etc/passwd"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid worker ID: contains pipe characters"

  @p0 @validation @format @security
  Scenario: VAL-010 - Validate model reference format (hf:org/repo)
    Given queen-rbee is running at "http://localhost:8080"
    When I send request with model_ref "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    Then request is accepted
    When I send request with model_ref "invalid-format"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid model reference format"

  @p0 @validation @format @security
  Scenario: VAL-011 - Validate model reference (no special chars)
    Given queen-rbee is running at "http://localhost:8080"
    When I send request with model_ref "hf:org/repo!@#$%"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid model reference: contains special characters"

  @p0 @validation @format
  Scenario: VAL-012 - Validate worker ID format (alphanumeric + dash)
    Given rbee-hive is running at "http://localhost:8081"
    When I send request with worker_id "worker-123-abc"
    Then request is accepted
    When I send request with worker_id "worker_123_abc"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid worker ID: only alphanumeric and dash allowed"

  @p0 @validation @format
  Scenario: VAL-013 - Validate worker ID length (max 64 chars)
    Given rbee-hive is running at "http://localhost:8081"
    When I send request with worker_id "worker-" repeated 20 times
    Then request is rejected with 400 Bad Request
    And error message is "Invalid worker ID: exceeds maximum length of 64 characters"

  @p0 @validation @format
  Scenario: VAL-014 - Validate backend name (cuda, metal, cpu only)
    Given rbee-hive is running at "http://localhost:8081"
    When I send request with backend "cuda"
    Then request is accepted
    When I send request with backend "metal"
    Then request is accepted
    When I send request with backend "cpu"
    Then request is accepted
    When I send request with backend "quantum"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid backend: must be one of [cpu, cuda, metal]"

  @p0 @validation @format
  Scenario: VAL-015 - Validate device ID (non-negative integer)
    Given rbee-hive is running at "http://localhost:8081"
    When I send request with device "0"
    Then request is accepted
    When I send request with device "5"
    Then request is accepted
    When I send request with device "-1"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid device ID: must be non-negative integer"

  @p0 @validation @format
  Scenario: VAL-016 - Validate port number (1024-65535)
    Given queen-rbee config has port "8080"
    Then queen-rbee starts successfully
    Given queen-rbee config has port "80"
    Then queen-rbee fails to start
    And displays error: "Invalid port: must be between 1024 and 65535"
    Given queen-rbee config has port "70000"
    Then queen-rbee fails to start
    And displays error: "Invalid port: must be between 1024 and 65535"

  @p0 @validation @format
  Scenario: VAL-017 - Validate node name (DNS-safe characters)
    Given queen-rbee is running at "http://localhost:8080"
    When I send request with node "workstation"
    Then request is accepted
    When I send request with node "workstation.home.arpa"
    Then request is accepted
    When I send request with node "work@station"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid node name: must be DNS-safe"

  @p0 @validation @sql-injection @security
  Scenario: VAL-018 - Reject SQL injection attempts
    Given model-catalog is running with SQLite database
    When I send request with model_ref "hf:'; DROP TABLE models; --"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid model reference format"
    And SQL injection is prevented
    And database tables are intact

  @p0 @validation @xss @security
  Scenario: VAL-019 - Reject XSS attempts
    Given queen-rbee is running at "http://localhost:8080"
    When I send request with model_ref "<script>alert('XSS')</script>"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid model reference: contains HTML/script tags"
    And response body does not contain script tags

  @p0 @validation @fuzzing @security
  Scenario: VAL-020 - Property-based testing (proptest fuzzing)
    Given queen-rbee is running at "http://localhost:8080"
    When I send 1000 requests with randomly generated inputs
    Then no request causes panic or crash
    And all invalid inputs are rejected with 400 Bad Request
    And all valid inputs are accepted
    And no memory leaks occur

  @p0 @validation @unicode @security
  Scenario: VAL-021 - Validate Unicode handling
    Given queen-rbee is running at "http://localhost:8080"
    When I send request with model_ref "hf:org/repo-émoji-🚀"
    Then request is rejected with 400 Bad Request
    And error message is "Invalid model reference: non-ASCII characters not allowed"

  @p0 @validation @size-limits @security
  Scenario: VAL-022 - Enforce request body size limits
    Given queen-rbee is running at "http://localhost:8080"
    When I send request with 10 MB body
    Then request is rejected with 413 Payload Too Large
    And error message is "Request body exceeds maximum size of 1 MB"

  @p0 @validation @rate-limiting @security
  Scenario: VAL-023 - Validate rate limiting on validation errors
    Given queen-rbee is running at "http://localhost:8080"
    When I send 100 invalid requests in 1 second
    Then requests are rate-limited after 50 failures
    And response status is 429 Too Many Requests
    And error message is "Too many validation errors"

  @p0 @validation @error-messages @security
  Scenario: VAL-024 - Safe error messages (no payload echoed)
    Given queen-rbee is running at "http://localhost:8080"
    When I send request with malicious payload "<script>alert('XSS')</script>"
    Then request is rejected with 400 Bad Request
    And error message does not contain "<script>alert('XSS')</script>"
    And error message is "Invalid model reference format"

  @p0 @validation @comprehensive
  Scenario: VAL-025 - Comprehensive validation on all endpoints
    Given queen-rbee is running at "http://localhost:8080"
    When I send malicious input to "/v1/workers/spawn"
    Then input is validated and rejected
    When I send malicious input to "/v1/workers/{id}"
    Then input is validated and rejected
    When I send malicious input to "/v1/inference"
    Then input is validated and rejected
    And all endpoints perform input validation
