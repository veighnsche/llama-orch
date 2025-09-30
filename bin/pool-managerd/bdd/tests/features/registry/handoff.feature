Feature: Registry Handoff Registration
  # Traceability: B-REG-100 through B-REG-107
  # Spec: OC-POOL-3101, OC-POOL-3102 - register_ready_from_handoff API
  
  Scenario: Register ready from complete handoff
    Given an empty registry
    And a handoff JSON with all fields:
      """
      {
        "engine_version": "llamacpp-v1.2.3",
        "device_mask": "0,1",
        "slots_total": 8,
        "slots_free": 8
      }
      """
    When I call register_ready_from_handoff for pool "pool1"
    Then the pool health is live=true ready=true
    And the pool engine_version is "llamacpp-v1.2.3"
    And the pool device_mask is "0,1"
    And the pool slots_total is 8
    And the pool slots_free is 8
    And the pool last_error is cleared
    And the pool heartbeat is set
  
  Scenario: Register ready clears previous error
    Given an empty registry
    And a pool "pool1" with last_error "previous failure"
    And a handoff JSON with engine_version "llamacpp-v1.2.3"
    When I call register_ready_from_handoff for pool "pool1"
    Then the pool health is live=true ready=true
    And the pool last_error is None
  
  Scenario: Register ready with minimal handoff
    Given an empty registry
    And a handoff JSON with only engine_version "llamacpp-v1.0.0"
    When I call register_ready_from_handoff for pool "pool1"
    Then the pool health is live=true ready=true
    And the pool engine_version is "llamacpp-v1.0.0"
    And the pool device_mask is None
    And the pool slots_total is None
  
  Scenario: Register ready sets heartbeat to current time
    Given an empty registry
    And a handoff JSON with engine_version "llamacpp-v1.2.3"
    When I call register_ready_from_handoff for pool "pool1"
    Then the pool heartbeat is within 1000ms of current time
  
  Scenario: Register ready handles missing optional fields
    Given an empty registry
    And a handoff JSON with:
      """
      {
        "engine_version": "llamacpp-v1.2.3",
        "slots_total": 4
      }
      """
    When I call register_ready_from_handoff for pool "pool1"
    Then the pool health is live=true ready=true
    And the pool engine_version is "llamacpp-v1.2.3"
    And the pool slots_total is 4
    And the pool device_mask is None
    And the pool slots_free is None
  
  Scenario: Register ready updates existing pool
    Given an empty registry
    And a pool "pool1" with health live=false ready=false
    And a handoff JSON with engine_version "llamacpp-v2.0.0"
    When I call register_ready_from_handoff for pool "pool1"
    Then the pool health is live=true ready=true
    And the pool engine_version is "llamacpp-v2.0.0"
  
  Scenario: Register ready with device mask
    Given an empty registry
    And a handoff JSON with:
      """
      {
        "engine_version": "llamacpp-v1.2.3",
        "device_mask": "GPU0"
      }
      """
    When I call register_ready_from_handoff for pool "pool1"
    Then the pool device_mask is "GPU0"
  
  Scenario: Register ready with slots
    Given an empty registry
    And a handoff JSON with:
      """
      {
        "engine_version": "llamacpp-v1.2.3",
        "slots_total": 16,
        "slots_free": 12
      }
      """
    When I call register_ready_from_handoff for pool "pool1"
    Then the pool slots_total is 16
    And the pool slots_free is 12
