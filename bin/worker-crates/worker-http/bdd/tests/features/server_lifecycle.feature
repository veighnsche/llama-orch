Feature: HTTP Server Lifecycle
  As a worker implementation
  I want to manage HTTP server lifecycle
  So that I can handle requests and shut down gracefully

  Scenario: Start and stop server
    Given an HTTP server on port 8080
    When I start the server
    Then the server should be running
    When I send a shutdown signal
    Then the server should shut down gracefully

  Scenario: Bind failure handling
    Given an HTTP server on port 80
    When I start the server
    Then the server should fail to bind
    And the error should be "BindFailed"
