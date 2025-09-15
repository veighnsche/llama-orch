Feature: SSE details
  # Traceability: ORCH-2002, OC-CTRL-2020..2022
  Scenario: SSE frames and ordering
    Given an OrchQueue API endpoint
    When I stream task events
    Then I receive SSE metrics frames
    And started includes queue_position and predicted_start_ms
    And SSE event ordering is per stream
