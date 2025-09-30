Feature: Registry Health State Management
  # Traceability: B-REG-001, B-REG-002, B-REG-003, B-REG-004
  # Spec: Health status tracking in registry
  
  Scenario: Set and get health status
    Given an empty registry
    When I set health for pool "pool1" to live=true ready=true
    And I get health for pool "pool1"
    Then the health status is live=true ready=true
  
  Scenario: Get health for non-existent pool
    Given an empty registry
    When I get health for pool "missing"
    Then the result is None
  
  Scenario: Health defaults to live=false ready=false
    Given an empty registry
    When I set health for pool "pool1" to live=false ready=false
    And I get health for pool "pool1"
    Then the health status is live=false ready=false
  
  Scenario: Update health status
    Given an empty registry
    And I set health for pool "pool1" to live=true ready=false
    When I set health for pool "pool1" to live=true ready=true
    And I get health for pool "pool1"
    Then the health status is live=true ready=true
