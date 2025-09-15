Feature: TGI adapter contract
  # Traceability: ORCH-3056, ORCH-3058
  Scenario: Adapter implements required endpoints and metadata
    Given a worker adapter for tgi
    Then the adapter implements TGI custom API and metrics
    And OpenAI-compatible endpoints are internal only
    And adapter reports engine_version and model_digest
