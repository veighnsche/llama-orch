Feature: Deadlines and SSE metrics
  # Traceability: ORCH-3079..3081, ORCH-3029
  Scenario: Infeasible deadlines rejected
    Given a task with infeasible deadline
    When I enqueue a completion task with valid payload
    Then I receive error code DEADLINE_UNMET

  Scenario: SSE exposes on_time_probability
    Given an OrchQueue API endpoint
    When I stream task events
    Then SSE metrics include on_time_probability
    And started includes queue_position and predicted_start_ms
    And SSE event ordering is per stream
