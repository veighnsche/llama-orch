Feature: Ready Callback
  As a worker implementation
  I want to send ready callbacks to pool manager
  So that the pool manager knows I'm ready for work

  Scenario: NVIDIA worker ready callback
    Given a worker ready callback
    And memory usage is 16000000000 bytes
    And memory architecture is "vram-only"
    When I send the ready callback
    Then the callback should include memory usage
    And the callback should include memory architecture

  Scenario: Apple ARM worker ready callback
    Given a worker ready callback
    And memory usage is 8000000000 bytes
    And memory architecture is "unified"
    When I send the ready callback
    Then the callback should include memory usage
    And the callback should include memory architecture
