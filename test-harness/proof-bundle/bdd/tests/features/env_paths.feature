Feature: Proof bundle base path, run id, and type mapping
  # Traceability: PB-1001, PB-1002, PB-1003, PB-1010; PBV-2001, PBV-2002, PBV-2007
  Verify PB-1001..PB-1003 and PB-1010 behaviors for location, overrides, and dir mapping.

  Background:
    Given I clear proof bundle env overrides
    And I set current dir to the world root

  Scenario: Default base and run id (PBV-2001)
    When I open a bundle for type "integration"
    Then bundle root ends with type dir "integration" and a run id
    And bundle root is under world .proof_bundle
    And the run id matches regex "^[0-9]+(-[0-9a-f]{8})?$"

  Scenario: Env overrides honored (PBV-2002)
    Given I set env overrides with run id "BDD-ENV-1"
    When I open a bundle for type "unit"
    Then bundle root is under world root and ends with "unit"/"BDD-ENV-1"

  Scenario Outline: TestType mapping to directory names (PBV-2007)
    Then dir name mapping for type "<type>" equals "<dir>"

    Examples:
      | type         | dir                 |
      | unit         | unit                |
      | integration  | integration         |
      | contract     | contract            |
      | bdd          | bdd                 |
      | determinism  | determinism         |
      | smoke        | home-profile-smoke  |
      | e2e-haiku    | e2e-haiku           |
