Feature: Deadlines with preemption
  # Traceability: ORCH-3079, ORCH-3085, ORCH-3087

  Scenario: Infeasible deadlines with soft preemption
    Given a task with infeasible deadline
    And soft preemption is enabled
    When I enqueue a completion task with valid payload
    Then I receive error code DEADLINE_UNMET
    And SSE metrics include on_time_probability

  Scenario: Infeasible deadlines with hard preemption
    Given a task with infeasible deadline
    And hard preemption is enabled and adapter proves interruptible_decode
    When I stream task events
    Then preempted flag and resumable state are surfaced
    And SSE metrics include on_time_probability
