Feature: Typed Error taxonomy
  # Traceability: ORCH-2006, OC-CTRL-2030..2031
  Scenario: Invalid params yields 400
    Given an OrchQueue API endpoint
    When I trigger INVALID_PARAMS
    Then I receive 400 with correlation id and error envelope code INVALID_PARAMS
    And error envelope includes engine when applicable

  Scenario: Pool unavailable yields 503
    Given an OrchQueue API endpoint
    When I trigger POOL_UNAVAILABLE
    Then I receive 503 with correlation id and error envelope code POOL_UNAVAILABLE
    And error envelope includes engine when applicable

  Scenario: Internal error yields 500
    Given an OrchQueue API endpoint
    When I trigger INTERNAL error
    Then I receive 500 with correlation id and error envelope code INTERNAL
    And error envelope includes engine when applicable
