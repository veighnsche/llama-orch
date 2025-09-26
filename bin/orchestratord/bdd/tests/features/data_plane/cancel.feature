Feature: Cancel a queued task
  # Traceability: ORCH-2003 (cancel)
  Scenario: Client cancels queued task
    Given an existing queued task
    When I cancel the task
    Then I receive 204 No Content with correlation id
