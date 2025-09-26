Feature: http-util client defaults (HTU-1001, Security)
  As an adapter developer
  I want sane client defaults (timeouts, HTTP/2 keep-alive, TLS verify)
  So that adapters behave consistently and securely

  Scenario: Default timeouts and TLS verification
    Given no special http-util configuration
    When I inspect http-util client defaults
    Then connect timeout is approximately 5s and request timeout approximately 30s
    And TLS verification is ON by default

  Scenario: HTTP/2 keep-alive reuse when server supports ALPN
    Given no special http-util configuration
    When I inspect http-util client defaults
    Then HTTP/2 keep-alive is enabled when server supports ALPN
