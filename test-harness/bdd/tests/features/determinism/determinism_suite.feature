Feature: Determinism across replicas
  # Traceability: ORCH-3045, OC-TEST-7001..7003
  Scenario: Byte-exact token streams
    Given two replicas pin engine_version sampler_profile_version and model_digest
    And same prompt parameters and seed are used
    Then token streams are byte-exact across replicas
