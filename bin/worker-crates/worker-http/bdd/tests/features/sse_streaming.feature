Feature: SSE Streaming
  As a worker implementation
  I want to stream tokens via SSE
  So that clients can receive real-time inference results

  Scenario: Stream multiple tokens
    Given an SSE stream
    When I send a token event
    And I send a token event
    And I send a token event
    Then the client should receive 3 events
    When I close the stream
    Then the stream should close cleanly

  Scenario: Handle client disconnect
    Given an SSE stream
    When I send a token event
    And the client disconnects
    Then the stream should close cleanly
    And no error should be logged
