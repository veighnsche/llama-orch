Feature: SSE transcript persistence
  # After streaming, the SSE transcript is persisted as an artifact with expected events

  Scenario: Streaming persists transcript
    Given an OrchQueue API endpoint
    When I stream task events
    Then SSE transcript artifact exists with events started token metrics end
