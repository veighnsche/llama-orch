Feature: Security gates
  # Traceability: ORCH-3035; OC-CTRL-2040..2041
  Scenario: Missing API key
    Given an OrchQueue API endpoint
    And no API key is provided
    Then I receive 401 Unauthorized

  Scenario: Invalid API key
    Given an OrchQueue API endpoint
    And an invalid API key is provided
    Then I receive 403 Forbidden
