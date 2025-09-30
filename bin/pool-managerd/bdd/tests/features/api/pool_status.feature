Feature: Pool Status Query
  # Traceability: B-API-020, B-API-021, B-API-022, B-API-023, B-API-024
  # Spec: GET /pools/{id}/status endpoint behaviors
  
  Scenario: Query status of existing pool
    Given a running pool-managerd daemon
    And a pool "test-pool" is registered
    And the pool has health live=true ready=true
    When I request GET /pools/test-pool/status
    Then I receive 200 OK
    And the response includes pool_id field
    And the response includes live field
    And the response includes ready field
    And the response includes active_leases field
    And the pool_id field equals "test-pool"
  
  Scenario: Query status of non-existent pool
    Given a running pool-managerd daemon
    And no pool "missing-pool" exists
    When I request GET /pools/missing-pool/status
    Then I receive 404 Not Found
    And the error message contains "pool missing-pool not found"
  
  Scenario: Status reflects current registry state
    Given a running pool-managerd daemon
    And a pool "test-pool" is registered
    And the pool has health live=true ready=false
    And the pool has active_leases 3
    And the pool has engine_version "llamacpp-v1.2.3"
    When I request GET /pools/test-pool/status
    Then I receive 200 OK
    And the live field equals true
    And the ready field equals false
    And the active_leases field equals 3
    And the engine_version field equals "llamacpp-v1.2.3"
  
  Scenario: Status includes optional engine_version
    Given a running pool-managerd daemon
    And a pool "test-pool" is registered
    And the pool has no engine_version set
    When I request GET /pools/test-pool/status
    Then I receive 200 OK
    And the engine_version field is null
