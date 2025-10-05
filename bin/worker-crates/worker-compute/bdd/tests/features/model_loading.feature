Feature: Model Loading
  As a worker implementation
  I want to load models
  So that I can run inference

  Scenario: Load valid GGUF model
    Given an initialized compute backend
    And a model path "/models/llama-3.1-8b.gguf"
    When I load the model
    Then the model should load successfully
    And the memory usage should be 8000000000 bytes

  Scenario: Load model with invalid format
    Given an initialized compute backend
    And a model path "/models/model.bin"
    When I load the model
    Then the model loading should fail
    And the error should be "ModelLoadFailed"

  Scenario: Load model with empty path
    Given an initialized compute backend
    And a model path ""
    When I load the model
    Then the model loading should fail
    And the error should be "InvalidParameter"
