Feature: Llama.cpp adapter contract
  # Traceability: ORCH-3054, ORCH-3058
  Scenario: Adapter implements required endpoints and metadata
    Given a worker adapter for llama.cpp
    Then the adapter implements health properties completion cancel metrics
    And OpenAI-compatible endpoints are internal only
    And adapter reports engine_version and model_digest
