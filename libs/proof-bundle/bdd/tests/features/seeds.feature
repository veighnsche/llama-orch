Feature: Seeds recorder behavior
  # Traceability: PB-1005, PB-1006; PBV-2006
  Verify PB-1005/PB-1006 and PBV-2006: seeds recorder appends and is newline-terminated.

  Background:
    Given a proof bundle of type "integration" with run id "BDD-SEEDS-1"

  Scenario: Record multiple seeds appends lines
    When I record seed "42"
    And I record seed "1337"
    Then seeds file exists
    And file "seeds.txt" has at least 2 lines
    And first line of "seeds.txt" contains "seed="
    And last line of "seeds.txt" equals "seed=1337"
