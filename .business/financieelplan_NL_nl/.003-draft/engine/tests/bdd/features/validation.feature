@wip
Feature: Validation and warnings policy
  Engine validates schemas/paths and escalates warnings when configured

  Scenario: Unknown path in variables causes ERROR
    Given the inputs directory ../inputs
    And a fresh outputs directory
    And I copy the inputs to a temporary workspace
    # TODO: inject an invalid variable path into public_tap.csv
    When I run the engine CLI with pipelines "public" and seed 424242
    Then the command should exit with code 2

  Scenario: CSV>YAML shadowing escalates to ERROR when fail_on_warning is true
    Given the inputs directory ../inputs
    And a fresh outputs directory
    And I copy the inputs to a temporary workspace
    And I set simulation.yaml key "run.fail_on_warning" to true
    # TODO: create a known shadowing by duplicating a key in CSV overriding YAML
    When I run the engine CLI with pipelines "public" and seed 424242
    Then the command should exit with code 2
