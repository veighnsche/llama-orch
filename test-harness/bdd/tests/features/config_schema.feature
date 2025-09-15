Feature: Config schema validation and generation
  # Traceability: OC-CONFIG-6001..6010
  Scenario: Valid example validates
    Given a valid example config
    Then schema validation passes

  Scenario: Strict mode rejects unknown field
    Given strict mode with unknown field
    Then validation rejects unknown fields

  Scenario: Schema generation is idempotent
    Given schema is generated twice
    Then outputs are identical
