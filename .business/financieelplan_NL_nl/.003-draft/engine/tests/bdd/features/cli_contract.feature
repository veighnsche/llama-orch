Feature: Engine CLI contract
  As a user of the D3 engine
  I want a CLI that validates inputs and runs simulations
  So that I can produce artifacts and progress logs deterministically

  Scenario: Run CLI with inputs and outputs
    Given the inputs directory ../inputs
    And a fresh outputs directory
    When I run the engine CLI with pipelines "public,private" and seed 424242
    Then the command should exit with code 0
    And the outputs directory should contain file "run_summary.json"
    And stdout should contain JSONL records with keys "ts","level","event"
