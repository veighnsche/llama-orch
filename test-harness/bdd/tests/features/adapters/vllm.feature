Feature: vLLM adapter contract
  # Traceability: ORCH-3055, ORCH-3058
  Scenario: Adapter implements required endpoints and metadata
    Given a worker adapter for vllm
    Then the adapter implements health/properties/completion/cancel/metrics against vLLM
    And OpenAI-compatible endpoints are internal only
    And adapter reports engine_version and model_digest
