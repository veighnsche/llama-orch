Feature: Core Narration Behaviors
  As a developer using narration-core
  I want to emit structured narration events
  So that I can debug and observe my application

  Background:
    Given a clean capture adapter

  Scenario: B-CORE-001 - Basic narration emits structured event
    When I narrate with actor orchestratord, action admission, target session-123, and human Accepted request
    Then the narration is captured
    And the captured narration has actor orchestratord
    And the captured narration has action admission
    And the captured narration has target session-123
    And the captured narration has human Accepted request

  Scenario: B-CORE-002 - Human field is automatically redacted
    When I narrate with human text Authorization: Bearer secret123
    Then the captured narration human text does not contain secret123
    And the captured narration human text contains [REDACTED]

  Scenario: B-CORE-006 - Correlation ID is included
    When I narrate with correlation_id req-xyz
    Then the captured narration has correlation_id req-xyz

  Scenario: B-CORE-007 - Session ID is included
    When I narrate with session_id session-abc
    Then the captured narration has session_id session-abc

  Scenario: B-CORE-010 - Pool ID is included
    When I narrate with pool_id default
    Then the captured narration has pool_id default

  Scenario: B-CORE-015 - Emitted by is included
    When I narrate with emitted_by orchestratord@0.1.0
    Then the captured narration has emitted_by orchestratord@0.1.0

  Scenario: B-CORE-016 - Emitted at timestamp is included
    When I narrate with emitted_at_ms 1234567890
    Then the captured narration has emitted_at_ms 1234567890

  Scenario: B-CORE-017 - Trace ID is included
    When I narrate with trace_id trace-123
    Then the captured narration has trace_id trace-123

  Scenario: B-CORE-100 - Legacy human() function works
    When I call legacy human() with actor test, action test, target test, message Test message
    Then the narration is captured
    And the captured narration has human Test message

  Scenario: B-CORE-103 - Legacy function defaults other fields to None
    When I call legacy human() with actor test, action test, target test, message Test
    Then the captured narration has no correlation_id
    And the captured narration has no session_id
