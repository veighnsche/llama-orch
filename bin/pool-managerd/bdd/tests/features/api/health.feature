Feature: Daemon Health Check
  # Traceability: B-API-001, B-API-002, B-API-003
  # Spec: Basic daemon health endpoint
  
  Scenario: Health endpoint returns OK
    Given a running pool-managerd daemon
    When I request GET /health
    Then I receive 200 OK
    And the response includes status field
    And the response includes version field
    And the status field equals "ok"
  
  Scenario: Health endpoint is available when no pools exist
    Given a running pool-managerd daemon
    And no pools are registered
    When I request GET /health
    Then I receive 200 OK
    And the status field equals "ok"
  
  Scenario: Health response includes daemon version
    Given a running pool-managerd daemon
    When I request GET /health
    Then I receive 200 OK
    And the version field matches CARGO_PKG_VERSION
