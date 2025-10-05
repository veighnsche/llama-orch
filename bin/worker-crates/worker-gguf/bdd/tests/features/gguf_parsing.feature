Feature: GGUF File Parsing
  As a worker implementation
  I want to parse GGUF metadata
  So that I can understand model architecture and configuration

  Scenario: Parse Qwen model metadata
    Given a GGUF file "qwen-2.5-0.5b.gguf"
    When I parse the GGUF metadata
    Then the architecture should be "llama"
    And the vocabulary size should be 151936
    And the hidden dimension should be 896
    And the number of layers should be 24
    And the number of attention heads should be 14
    And the number of KV heads should be 2
    And the model should use GQA
    And the RoPE frequency base should be 1000000.0
    And the context length should be 32768

  Scenario: Parse Phi-3 model metadata
    Given a GGUF file "phi-3-mini.gguf"
    When I parse the GGUF metadata
    Then the architecture should be "llama"
    And the vocabulary size should be 32064
    And the hidden dimension should be 3072
    And the number of layers should be 32
    And the number of attention heads should be 32
    And the number of KV heads should be 32
    And the model should use MHA
    And the RoPE frequency base should be 10000.0
    And the context length should be 4096

  Scenario: Parse GPT-2 model metadata
    Given a GGUF file "gpt2-small.gguf"
    When I parse the GGUF metadata
    Then the architecture should be "gpt"
    And the vocabulary size should be 50257
    And the hidden dimension should be 768
    And the number of layers should be 12
