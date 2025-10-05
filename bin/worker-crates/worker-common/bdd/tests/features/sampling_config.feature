Feature: Sampling Configuration
  As a worker implementation
  I want to configure sampling parameters
  So that I can control inference behavior

  Scenario: Greedy sampling (temperature = 0)
    Given a sampling config with temperature 0.0
    When I check if advanced sampling is enabled
    Then advanced sampling should be disabled
    And the sampling mode should be "greedy"

  Scenario: Advanced sampling enabled
    Given a sampling config with temperature 0.7
    And top_p is 0.9
    And top_k is 50
    When I check if advanced sampling is enabled
    Then advanced sampling should be enabled

  Scenario: Default sampling (no filtering)
    Given a sampling config with temperature 1.0
    And top_p is 1.0
    And top_k is 0
    When I check if advanced sampling is enabled
    Then advanced sampling should be disabled
