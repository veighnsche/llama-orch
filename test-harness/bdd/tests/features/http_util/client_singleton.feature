Feature: http-util client singleton (HTU-1001 subset)
  As a developer using http-util
  I want a reusable HTTP client
  So that connections are pooled and reused

  Scenario: client() returns a single shared instance
    Given no special http-util configuration
    When I get the http-util client twice
    Then both references point to the same client
