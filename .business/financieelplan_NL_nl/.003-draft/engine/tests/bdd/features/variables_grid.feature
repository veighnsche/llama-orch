@wip
Feature: Variables grid and random replicates
  Engine expands low_to_high variables into a grid and redraws random variables per replicate

  Scenario: Low-to-high grid and random replicates are applied
    Given the inputs directory ../inputs
    And a fresh outputs directory
    And I copy the inputs to a temporary workspace
    And I set simulation.yaml key "run.random_runs_per_simulation" to 5
    When I run the engine CLI with pipelines "public,private" and seed 424242
    Then the command should exit with code 0
    And stdout should contain event "run_start"
    And the run_summary should contain keys "pipelines","seed"
