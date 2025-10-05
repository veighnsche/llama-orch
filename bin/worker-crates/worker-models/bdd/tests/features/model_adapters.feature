Feature: Model Adapters
  As a worker implementation
  I want to automatically detect model architecture
  So that I can use the correct inference pipeline

  Scenario: Detect Llama-style architecture
    Given a GGUF file "qwen-2.5-0.5b.gguf"
    When I detect the model architecture
    Then the architecture should be "llama"
    When I create a model adapter
    Then the adapter should be "LlamaAdapter"
    And the adapter should support inference

  Scenario: Detect GPT-style architecture
    Given a GGUF file "gpt2-small.gguf"
    When I detect the model architecture
    Then the architecture should be "gpt"
    When I create a model adapter
    Then the adapter should be "GPTAdapter"
    And the adapter should support inference
