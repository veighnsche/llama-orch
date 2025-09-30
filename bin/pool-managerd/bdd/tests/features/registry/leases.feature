Feature: Registry Lease Accounting
  # Traceability: B-REG-090, B-REG-091, B-REG-092, B-REG-093, B-REG-094, B-REG-095, B-REG-096
  # Spec: Lease allocation and release tracking
  
  Scenario: Allocate lease increments count
    Given an empty registry
    And a pool "pool1" is registered
    When I allocate a lease for pool "pool1"
    Then the active_leases count is 1
  
  Scenario: Allocate multiple leases
    Given an empty registry
    And a pool "pool1" is registered
    When I allocate a lease for pool "pool1"
    And I allocate a lease for pool "pool1"
    And I allocate a lease for pool "pool1"
    Then the active_leases count is 3
  
  Scenario: Release lease decrements count
    Given an empty registry
    And a pool "pool1" is registered
    And the pool has 3 active leases
    When I release a lease for pool "pool1"
    Then the active_leases count is 2
  
  Scenario: Release lease never goes below zero
    Given an empty registry
    And a pool "pool1" is registered
    And the pool has 0 active leases
    When I release a lease for pool "pool1"
    Then the active_leases count is 0
  
  Scenario: Get active leases for non-existent pool
    Given an empty registry
    When I get active_leases for pool "missing"
    Then the active_leases count is 0
  
  Scenario: Allocate and release cycle
    Given an empty registry
    And a pool "pool1" is registered
    When I allocate a lease for pool "pool1"
    And I allocate a lease for pool "pool1"
    And I release a lease for pool "pool1"
    And I allocate a lease for pool "pool1"
    Then the active_leases count is 2
  
  Scenario: Multiple pools have independent lease counts
    Given an empty registry
    And a pool "pool1" is registered
    And a pool "pool2" is registered
    When I allocate a lease for pool "pool1"
    And I allocate a lease for pool "pool1"
    And I allocate a lease for pool "pool2"
    Then the active_leases count for "pool1" is 2
    And the active_leases count for "pool2" is 1
