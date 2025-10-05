Feature: Compute Backend Initialization
  As a worker implementation
  I want to initialize compute backends
  So that I can run inference on different devices

  Scenario: Initialize valid device
    Given a compute device with ID 0
    When I initialize the compute backend
    Then the backend should initialize successfully

  Scenario: Initialize invalid device
    Given a compute device with ID -1
    When I initialize the compute backend
    Then the initialization should fail
    And the error should be "DeviceNotFound"
