Feature: Budget headers
  # Verifies presence of budget headers on enqueue and stream responses

  Scenario: Enqueue returns budget headers
    Given an OrchQueue API endpoint
    When I enqueue a completion task with valid payload
    Then I receive 202 Accepted with correlation id
    And budget headers are present

  Scenario: Stream returns budget headers
    Given an OrchQueue API endpoint
    And I enqueue a completion task with valid payload
    Then I receive 202 Accepted with correlation id
    When I stream task events
    Then budget headers are present
