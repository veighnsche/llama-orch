Feature: http-util bearer injection (HTU-1005)
  As an adapter developer
  I want http-util to optionally inject Authorization header from env
  So that I can easily add bearer auth when configured

  Scenario: AUTH_TOKEN is set -> Authorization header present
    Given AUTH_TOKEN is set to "abc123"
    When I apply with_bearer_if_configured to a GET request
    Then the request has Authorization header "Bearer abc123"

  Scenario: AUTH_TOKEN is unset -> Authorization header absent
    Given AUTH_TOKEN is unset
    When I apply with_bearer_if_configured to a GET request
    Then the request has no Authorization header
