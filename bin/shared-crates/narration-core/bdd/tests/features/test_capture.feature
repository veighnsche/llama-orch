Feature: Test Capture Adapter Behaviors
  As a BDD test writer
  I want to capture and assert on narration events
  So that I can verify observability coverage

  Scenario: B-CAP-001 - Install creates new adapter
    When I install a capture adapter
    Then the adapter is available

  Scenario: B-CAP-010 - Narration is captured when adapter installed
    Given a clean capture adapter
    When I narrate with human "Test message"
    Then the narration is captured
    And captured events count is 1

  Scenario: B-CAP-014 - captured() returns all events
    Given a clean capture adapter
    When I narrate with human "Message 1"
    And I narrate with human "Message 2"
    And I narrate with human "Message 3"
    Then captured events count is 3

  Scenario: B-CAP-020 - clear() empties event vector
    Given a clean capture adapter
    When I narrate with human "Test"
    And I clear the capture adapter
    Then captured events count is 0

  Scenario: B-CAP-030 - assert_includes passes when found
    Given a clean capture adapter
    When I narrate with human "Spawning engine llamacpp-v1"
    Then assert_includes "Spawning engine" passes

  Scenario: B-CAP-031 - assert_includes fails when not found
    Given a clean capture adapter
    When I narrate with human "Accepted request"
    Then assert_includes "Rejected" fails

  Scenario: B-CAP-032 - assert_field passes when found
    Given a clean capture adapter
    When I narrate with actor "orchestratord"
    Then assert_field "actor" "orchestratord" passes

  Scenario: B-CAP-033 - assert_field fails when not found
    Given a clean capture adapter
    When I narrate with actor "orchestratord"
    Then assert_field "actor" "pool-managerd" fails

  Scenario: B-CAP-035 - assert_correlation_id_present passes when found
    Given a clean capture adapter
    When I narrate with correlation_id "req-xyz"
    Then assert_correlation_id_present passes

  Scenario: B-CAP-036 - assert_correlation_id_present fails when not found
    Given a clean capture adapter
    When I narrate without correlation_id
    Then assert_correlation_id_present fails

  Scenario: B-CAP-037 - assert_provenance_present passes when emitted_by present
    Given a clean capture adapter
    When I narrate with emitted_by "test@1.0.0"
    Then assert_provenance_present passes

  Scenario: B-CAP-038 - assert_provenance_present fails when neither present
    Given a clean capture adapter
    When I narrate without provenance
    Then assert_provenance_present fails
