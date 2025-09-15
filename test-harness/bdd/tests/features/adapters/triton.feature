Feature: Triton/TensorRT-LLM adapter contract
  # Traceability: ORCH-3057, ORCH-3058
  Scenario: Adapter implements required endpoints and metadata
    Given a worker adapter for triton
    Then the adapter implements infer/streaming and metrics
    And OpenAI-compatible endpoints are internal only
    And adapter reports engine_version and model_digest
