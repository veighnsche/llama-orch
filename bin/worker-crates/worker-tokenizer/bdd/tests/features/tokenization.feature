Feature: Tokenization
  As a worker implementation
  I want to encode and decode text
  So that I can process prompts and generate responses

  Scenario: Encode and decode simple text
    Given a "gguf-bpe" tokenizer
    And the text "Hello, world!"
    When I encode the text
    Then the token count should be 3
    When I decode the tokens
    Then the decoded text should match the original

  Scenario: UTF-8 boundary safety
    Given a "gguf-bpe" tokenizer
    And the text "Hello 世界 🌍"
    When I encode the text
    Then the encoding should be UTF-8 safe
    When I decode the tokens
    Then the decoded text should match the original
