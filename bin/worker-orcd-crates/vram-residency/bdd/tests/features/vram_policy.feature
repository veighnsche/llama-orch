Feature: VRAM-Only Policy Enforcement
  As a worker-orcd service
  I want to enforce VRAM-only policy
  So that models never fall back to RAM during inference

  Scenario: Enforce VRAM-only policy at initialization
    Given a VramManager with 10MB capacity
    When I enforce VRAM-only policy
    Then the policy enforcement should succeed
    And unified memory should be disabled
    And zero-copy should be disabled
    And pinned host memory should be disabled

  Scenario: Detect policy violation
    Given a VramManager with unified memory enabled
    When I enforce VRAM-only policy
    Then the policy enforcement should fail
    And an audit event "PolicyViolation" should be emitted
    And the worker should transition to Stopped state
