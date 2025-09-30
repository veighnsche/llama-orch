Feature: Prometheus Metrics
  # Traceability: B-OBS-001 through B-OBS-014
  # Metrics exposition in Prometheus text format

  Scenario: Metrics endpoint returns Prometheus format
    Given a metrics endpoint
    When I request /metrics
    Then I receive 200 OK
    And Content-Type is text/plain
    And the response includes TYPE headers
    And the response includes pre-registered metrics

  Scenario: Metrics include labels
    Given tasks have been enqueued
    When I request /metrics
    Then tasks_enqueued_total includes labels engine pool_id priority

  Scenario: Metrics conform to linter names and labels
    Given a metrics endpoint
    When I request /metrics
    Then metrics conform to linter names and labels
