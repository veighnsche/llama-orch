Feature: Model Adapters
  As a worker implementation
  I want to automatically detect model architecture
  So that I can use the correct inference pipeline

  Scenario: Detect and load Qwen model
    Given a GGUF file "qwen-2.5-0.5b.gguf"
    When I detect the model architecture
    Then the architecture should be "llama"
    When I create a model adapter
    Then the adapter should be "LlamaAdapter"
    And the adapter should support inference
    And the model should have vocab size 151936
    And the model should have 24 layers

  Scenario: Detect and load Phi-3 model
    Given a GGUF file "phi-3-mini.gguf"
    When I detect the model architecture
    Then the architecture should be "llama"
    When I create a model adapter
    Then the adapter should be "LlamaAdapter"
    And the adapter should support inference
    And the model should have vocab size 32064
    And the model should have 32 layers

  Scenario: Detect and load GPT-OSS-20B model
    Given a GGUF file "gpt-oss-20b.gguf"
    When I detect the model architecture
    Then the architecture should be "gpt"
    When I create a model adapter
    Then the adapter should be "GPTAdapter"
    And the adapter should support inference
    And the model should have vocab size 50257
    And the model should have 44 layers
