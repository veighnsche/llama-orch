Feature: Auto-Injection Behaviors
  As a developer in Cloud Profile deployments
  I want automatic provenance injection
  So that I don't have to manually set service identity and timestamps

  Background:
    Given a clean capture adapter

  Scenario: B-AUTO-001 - Service identity returns name@version format
    When I get the service identity
    Then it matches the pattern ".*@.*"
    And it contains "observability-narration-core"

  Scenario: B-AUTO-012 - Timestamps are monotonically increasing
    When I get timestamp 1
    And I wait 10 milliseconds
    And I get timestamp 2
    Then timestamp 2 is greater than or equal to timestamp 1

  Scenario: B-AUTO-020 - Inject provenance when emitted_by is None
    When I narrate_auto without emitted_by
    Then the captured narration has emitted_by set

  Scenario: B-AUTO-021 - Preserve existing emitted_by
    When I narrate_auto with emitted_by "custom-service@1.0.0"
    Then the captured narration has emitted_by "custom-service@1.0.0"

  Scenario: B-AUTO-022 - Inject provenance when emitted_at_ms is None
    When I narrate_auto without emitted_at_ms
    Then the captured narration has emitted_at_ms set

  Scenario: B-AUTO-023 - Preserve existing emitted_at_ms
    When I narrate_auto with emitted_at_ms 9999999999
    Then the captured narration has emitted_at_ms 9999999999

  Scenario: B-AUTO-032 - narrate_auto injects service identity
    When I narrate_auto without emitted_by
    Then the captured narration emitted_by contains "@"

  Scenario: B-AUTO-033 - narrate_auto injects timestamp
    When I narrate_auto without emitted_at_ms
    Then the captured narration has emitted_at_ms greater than 0

  Scenario: B-AUTO-043 - narrate_full injects trace_id when available
    When I narrate_full without trace_id
    Then the captured narration may have trace_id

  Scenario: B-AUTO-047 - narrate_full preserves existing trace_id
    When I narrate_full with trace_id "custom-trace-123"
    Then the captured narration has trace_id "custom-trace-123"
