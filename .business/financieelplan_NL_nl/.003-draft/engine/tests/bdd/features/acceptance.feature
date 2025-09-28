@wip
Feature: Acceptance criteria
  Validate growth, margins and capacity according to simulation targets

  Scenario: Public growth is monotonic and Private margins are healthy
    Given the inputs directory ../inputs
    And a fresh outputs directory
    When I run the engine CLI with pipelines "public,private" and seed 424242
    Then the command should exit with code 0
    And the outputs directory should contain CSV "public_tap_customers_by_month.csv" with headers month,active_customers
    And CSV "public_tap_customers_by_month.csv" column "active_customers" should be monotonic nondecreasing
    And the outputs directory should contain CSV "private_tap_economics.csv" with headers gpu,margin_pct
    And CSV "private_tap_economics.csv" column "margin_pct" should be all >= 20.0

  Scenario: Public capacity has no violations
    Given the inputs directory ../inputs
    And a fresh outputs directory
    When I run the engine CLI with pipelines "public" and seed 424242
    Then the command should exit with code 0
    And the outputs directory should contain CSV "public_tap_capacity_plan.csv" with headers model,gpu,instances_needed,capacity_violation
    And CSV "public_tap_capacity_plan.csv" column "capacity_violation" should be all "False"
