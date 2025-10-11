# Traceability: TEST-001 (split by TEAM-077, TEAM-078)
# Architecture: TEAM-037 (queen-rbee orchestration, SQLite model catalog)
# Components: rbee-hive (pool manager), ModelCatalog (SQLite)
# Created by: TEAM-078 (split from 020-model-provisioning.feature)
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/
# ⚠️ DO NOT use mock servers - wire up actual rbee-hive and ModelCatalog libraries

Feature: Model Catalog
  As a system managing model metadata
  I want to query and register models in SQLite catalog
  So that I can track available models and their locations

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"
    And queen-rbee is running at "http://localhost:8080"
    And the model catalog is SQLite at "~/.rbee/models.db"

  Scenario: Model found in SQLite catalog
    Given the model catalog contains:
      | provider | reference                                 | local_path                  |
      | hf       | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF    | /models/tinyllama-q4.gguf   |
    When rbee-hive checks the model catalog
    Then the query returns local_path "/models/tinyllama-q4.gguf"
    And rbee-hive skips model download
    And rbee-hive proceeds to worker preflight

  Scenario: Model not found in catalog
    Given the model is not in the catalog
    When rbee-hive checks the model catalog
    Then the query returns no results
    And rbee-hive triggers model download

  Scenario: Model catalog registration after download
    Given the model downloaded successfully to "/models/tinyllama-q4.gguf"
    And the model size is 5242880 bytes
    When rbee-hive registers the model in the catalog
    Then the SQLite INSERT statement is:
      """
      INSERT INTO models (id, provider, reference, local_path, size_bytes, downloaded_at_unix)
      VALUES ('tinyllama-q4', 'hf', 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF', 
              '/models/tinyllama-q4.gguf', 5242880, 1728508603)
      """
    And the catalog query now returns the model

  Scenario: Query models by provider
    Given the model catalog contains:
      | provider | reference                                 | local_path                  |
      | hf       | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF    | /models/tinyllama-q4.gguf   |
      | hf       | TheBloke/Llama-2-7B-GGUF                  | /models/llama2-7b-q4.gguf   |
      | local    | custom-model                              | /models/custom.gguf         |
    When rbee-hive queries models with provider "hf"
    Then the query returns 2 models
    And all returned models have provider "hf"

  Scenario: GGUF model size calculation
    Given a GGUF file at "/models/tinyllama-q4.gguf"
    When rbee-hive calculates model size
    Then the file size is read from disk
    And the size is used for RAM preflight checks
    And the size is stored in the model catalog
