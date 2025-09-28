Feature: Deterministic outputs
  Running the engine with the same inputs and seed should produce identical logical results

  Scenario: Run twice with same seed and compare run_summary keys
    Given the inputs directory ../inputs
    And a fresh outputs directory
    And another fresh outputs directory
    When I run the engine CLI with pipelines "public,private" and seed 424242
    And I run the engine CLI with pipelines "public,private" and seed 424242 into the second outputs directory
    Then the command should exit with code 0
    And the run_summary key "seed" should match between outputs
    And the run_summary key "pipelines" should match between outputs
