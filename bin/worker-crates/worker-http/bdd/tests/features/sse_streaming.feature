Feature: SSE Streaming
  As a worker implementation
  I want to stream tokens via SSE
  So that clients can receive real-time inference results

  Scenario: Complete inference event stream
    Given an SSE event stream
    When I send a "started" event
    And I send a "token" event
    And I send a "token" event
    And I send a "token" event
    And I send a "end" event
    Then the stream should have 5 events
    And the stream should have a terminal event
    And the event order should be "started -> token -> token -> token -> end"

  Scenario: Error during inference
    Given an SSE event stream
    When I send a "started" event
    And I send a "token" event
    And I send a "error" event
    Then the stream should have 3 events
    And the stream should have a terminal event
    And the event order should be "started -> token -> error"

  Scenario: Metrics during inference
    Given an SSE event stream
    When I send a "started" event
    And I send a "token" event
    And I send a "metrics" event
    And I send a "token" event
    And I send a "end" event
    Then the stream should have 5 events
    And the event order should be "started -> token -> metrics -> token -> end"
