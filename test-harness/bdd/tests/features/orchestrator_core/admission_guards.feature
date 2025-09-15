Feature: Admission guardrails
  As a user of the data plane
  I want invalid requests to be rejected before enqueue
  So that capacity constraints are enforced early and fairly

  Scenario: Context length beyond model limit is rejected pre-admission
    Given a task with context length beyond model limit
    Then the request is rejected before enqueue

  Scenario: Token budget exceeding configured limit is rejected pre-admission
    Given a task with token budget exceeding configured limit
    Then the request is rejected before enqueue
