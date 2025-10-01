Feature: VRAM Operation Event Validation
  As a security-conscious audit system
  I want to validate VRAM operation events
  So that I prevent injection attacks in security-critical VRAM operations

  Scenario: Accept valid VramSealed event
    Given a shard ID "shard-abc123"
    And a worker ID "worker-gpu-0"
    When I create a VramSealed event
    And I validate the event
    Then the validation should succeed

  Scenario: Reject VramSealed with ANSI escape in shard ID
    Given a shard ID "\x1b[31mshard-fake\x1b[0m"
    And a worker ID "worker-gpu-0"
    When I create a VramSealed event
    And I validate the event
    Then the validation should reject ANSI escape sequences

  Scenario: Reject VramSealed with null byte in worker ID
    Given a shard ID "shard-123"
    And a worker ID "worker-gpu-0\0malicious"
    When I create a VramSealed event
    And I validate the event
    Then the validation should reject null bytes

  Scenario: Reject VramSealed with control characters in shard ID
    Given a shard ID "shard-123\r\nFAKE LOG"
    And a worker ID "worker-gpu-0"
    When I create a VramSealed event
    And I validate the event
    Then the validation should reject control characters
