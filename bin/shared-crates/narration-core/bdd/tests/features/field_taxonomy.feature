Feature: Field Taxonomy Behaviors
  As a developer using narration-core
  I want a comprehensive field taxonomy
  So that I can include all relevant context in narration

  Background:
    Given a clean capture adapter

  Scenario: B-FIELD-001 - Actor field is required
    When I create NarrationFields with actor "orchestratord"
    Then the fields have actor "orchestratord"

  Scenario: B-FIELD-002 - Action field is required
    When I create NarrationFields with action "admission"
    Then the fields have action "admission"

  Scenario: B-FIELD-003 - Target field is required
    When I create NarrationFields with target "session-123"
    Then the fields have target "session-123"

  Scenario: B-FIELD-004 - Human field is required
    When I create NarrationFields with human "Accepted request"
    Then the fields have human "Accepted request"

  Scenario: B-FIELD-010 - Correlation ID is optional
    When I narrate with correlation_id "req-xyz"
    Then the captured narration has correlation_id "req-xyz"

  Scenario: B-FIELD-011 - Session ID is optional
    When I narrate with session_id "session-abc"
    Then the captured narration has session_id "session-abc"

  Scenario: B-FIELD-012 - Job ID is optional
    When I narrate with job_id "job-123"
    Then the captured narration has job_id "job-123"

  Scenario: B-FIELD-014 - Pool ID is optional
    When I narrate with pool_id "default"
    Then the captured narration has pool_id "default"

  Scenario: B-FIELD-015 - Replica ID is optional
    When I narrate with replica_id "r0"
    Then the captured narration has replica_id "r0"

  Scenario: B-FIELD-060 - Default actor is empty string
    When I create default NarrationFields
    Then actor is ""

  Scenario: B-FIELD-061 - Default action is empty string
    When I create default NarrationFields
    Then action is ""

  Scenario: B-FIELD-062 - Default target is empty string
    When I create default NarrationFields
    Then target is ""

  Scenario: B-FIELD-063 - Default human is empty string
    When I create default NarrationFields
    Then human is ""

  Scenario: B-FIELD-064 - Default Option fields are None
    When I create default NarrationFields
    Then all Option fields are None
