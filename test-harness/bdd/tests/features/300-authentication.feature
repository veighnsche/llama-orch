# Traceability: RC-P0-AUTH (Release Candidate P0 Authentication)
# Created by: TEAM-097
# Components: queen-rbee, rbee-hive, llm-worker-rbee
#
# ⚠️ CRITICAL: Step definitions MUST import and test REAL product code from /bin/

Feature: Authentication
  As a security-conscious system
  I want to validate API tokens on all HTTP endpoints
  So that unauthorized access is prevented

  Background:
    Given the following topology:
      | node        | hostname              | components                                      | capabilities           |
      | blep        | blep.home.arpa        | rbee-keeper, queen-rbee                         | cpu                    |
      | workstation | workstation.home.arpa | rbee-hive, llm-worker-rbee                      | cuda:0, cuda:1, cpu    |
    And I am on node "blep"

  @p0 @auth @security
  Scenario: AUTH-001 - Reject request without Authorization header
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "test-token-12345"
    When I send POST to "/v1/workers/spawn" without Authorization header
    Then response status is 401 Unauthorized
    And response body contains "Missing API key"
    And response header "WWW-Authenticate" is "Bearer"
    And log contains "auth failed" with reason "missing_header"

  @p0 @auth @security
  Scenario: AUTH-002 - Reject request with invalid token
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "correct-token-12345"
    When I send POST to "/v1/workers/spawn" with Authorization "Bearer wrong-token-99999"
    Then response status is 401 Unauthorized
    And response body contains "Invalid or missing API key"
    And log contains token fingerprint "wrong-token-99999" (not raw token)
    And log contains "auth failed"

  @p0 @auth @security
  Scenario: AUTH-003 - Accept request with valid Bearer token
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "valid-token-12345"
    When I send POST to "/v1/workers/spawn" with Authorization "Bearer valid-token-12345"
    And request body is:
      """
      {
        "model_ref": "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "backend": "cpu",
        "node": "workstation"
      }
      """
    Then response status is 200 OK or 202 Accepted
    And log contains token fingerprint "valid-token-12345" (not raw token)
    And log contains "authenticated"

  @p0 @auth @security @timing-attack
  Scenario: AUTH-004 - Timing-safe token comparison (variance < 10%)
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "correct-token-12345"
    When I send 100 requests with valid token "correct-token-12345"
    And I send 100 requests with invalid token "wrong-token-99999"
    Then timing variance between valid and invalid is < 10%
    And no timing side-channel is detectable

  @p0 @auth @dev-mode
  Scenario: AUTH-005 - Loopback bind without token works (dev mode)
    Given queen-rbee is running at "http://127.0.0.1:8080"
    And queen-rbee has no API token configured
    When I send POST to "/v1/workers/spawn" without Authorization header
    And request is from localhost
    Then response status is 200 OK or 202 Accepted
    And log contains "dev mode: auth not required on loopback"

  @p0 @auth @security
  Scenario: AUTH-006 - Public bind requires token or fails to start
    Given queen-rbee config has bind address "0.0.0.0:8080"
    And queen-rbee config has no API token
    When I start queen-rbee
    Then queen-rbee fails to start
    And displays error: "API token required for non-loopback bind"
    And exit code is 1

  @p0 @auth @logging
  Scenario: AUTH-007 - Token fingerprinting in logs (never raw tokens)
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "secret-token-abc123xyz"
    When I send POST to "/v1/workers/spawn" with Authorization "Bearer secret-token-abc123xyz"
    Then log file does not contain "secret-token-abc123xyz"
    And log file contains token fingerprint (6-char SHA-256 prefix)
    And log entry format is: identity="token:a3f2c1"

  @p0 @auth @multi-component
  Scenario: AUTH-008 - Multiple components all require auth
    Given queen-rbee is running with auth at "http://0.0.0.0:8080"
    And rbee-hive is running with auth at "http://workstation.home.arpa:8081"
    And llm-worker-rbee is running with auth at "http://workstation.home.arpa:8082"
    When I send request to queen-rbee without auth
    Then response status is 401 Unauthorized
    When I send request to rbee-hive without auth
    Then response status is 401 Unauthorized
    When I send request to llm-worker-rbee without auth
    Then response status is 401 Unauthorized

  @p0 @auth @endpoints
  Scenario: AUTH-009 - Token validation on all HTTP endpoints
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "test-token-12345"
    When I send GET to "/health" without Authorization header
    Then response status is 200 OK
    When I send GET to "/v1/workers" without Authorization header
    Then response status is 401 Unauthorized
    When I send POST to "/v1/workers/spawn" without Authorization header
    Then response status is 401 Unauthorized
    When I send DELETE to "/v1/workers/worker-123" without Authorization header
    Then response status is 401 Unauthorized

  @p0 @auth @validation
  Scenario: AUTH-010 - Invalid token format rejected
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    When I send POST to "/v1/workers/spawn" with Authorization "InvalidFormat"
    Then response status is 401 Unauthorized
    And response body contains "Invalid Authorization header format"
    And log contains "auth failed" with reason "invalid_format"

  @p0 @auth @validation
  Scenario: AUTH-011 - Empty token rejected
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    When I send POST to "/v1/workers/spawn" with Authorization "Bearer "
    Then response status is 401 Unauthorized
    And response body contains "Missing API key"
    And log contains "auth failed" with reason "empty_token"

  @p0 @auth @special-chars
  Scenario: AUTH-012 - Token with special characters handled correctly
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "token-with-special!@#$%^&*()_+-=[]{}|;:,.<>?"
    When I send POST to "/v1/workers/spawn" with Authorization "Bearer token-with-special!@#$%^&*()_+-=[]{}|;:,.<>?"
    Then response status is 200 OK or 202 Accepted
    And log contains token fingerprint (not raw token)

  @p0 @auth @concurrency
  Scenario: AUTH-013 - Concurrent auth requests (no race conditions)
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "concurrent-test-token"
    When I send 50 concurrent requests with valid token
    And I send 50 concurrent requests with invalid token
    Then all 50 valid requests return 200 or 202
    And all 50 invalid requests return 401
    And no race conditions occur
    And all responses arrive within 5 seconds

  @p0 @auth @logging
  Scenario: AUTH-014 - Auth failure logged with fingerprint
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "correct-token"
    When I send POST to "/v1/workers/spawn" with Authorization "Bearer wrong-token"
    Then log contains:
      """
      level=WARN identity="token:abc123" msg="auth failed"
      """
    And log does not contain "wrong-token"

  @p0 @auth @logging
  Scenario: AUTH-015 - Auth success logged with fingerprint
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "valid-token-12345"
    When I send POST to "/v1/workers/spawn" with Authorization "Bearer valid-token-12345"
    Then log contains:
      """
      level=INFO identity="token:a3f2c1" msg="authenticated"
      """
    And log does not contain "valid-token-12345"

  @p0 @auth @bearer-parsing
  Scenario: AUTH-016 - Bearer token parsing edge cases
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    When I send request with Authorization "bearer lowercase-bearer"
    Then response status is 401 Unauthorized
    When I send request with Authorization "BEARER UPPERCASE-BEARER"
    Then response status is 401 Unauthorized
    When I send request with Authorization "Bearer  double-space-token"
    Then response status is 401 Unauthorized
    When I send request with Authorization "BearerNoSpace"
    Then response status is 401 Unauthorized

  @p0 @auth @http-methods
  Scenario: AUTH-017 - Auth required for all HTTP methods
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "test-token"
    When I send GET to "/v1/workers" without Authorization
    Then response status is 401 Unauthorized
    When I send POST to "/v1/workers/spawn" without Authorization
    Then response status is 401 Unauthorized
    When I send PUT to "/v1/workers/worker-123" without Authorization
    Then response status is 401 Unauthorized
    When I send DELETE to "/v1/workers/worker-123" without Authorization
    Then response status is 401 Unauthorized
    When I send PATCH to "/v1/workers/worker-123" without Authorization
    Then response status is 401 Unauthorized

  @p0 @auth @error-response
  Scenario: AUTH-018 - Consistent error response format
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    When I send POST to "/v1/workers/spawn" without Authorization header
    Then response status is 401 Unauthorized
    And response Content-Type is "application/json"
    And response body matches schema:
      """
      {
        "error": {
          "code": "MISSING_API_KEY",
          "message": "Missing API key"
        }
      }
      """

  @p0 @auth @integration
  Scenario: AUTH-019 - End-to-end auth flow (queen → hive → worker)
    Given queen-rbee is running with auth at "http://0.0.0.0:8080"
    And rbee-hive is running with auth at "http://workstation.home.arpa:8081"
    And queen-rbee has API token "queen-token-123"
    And rbee-hive has API token "hive-token-456"
    When rbee-keeper sends inference request to queen-rbee with token "queen-token-123"
    Then queen-rbee authenticates rbee-keeper successfully
    And queen-rbee forwards request to rbee-hive with token "hive-token-456"
    And rbee-hive authenticates queen-rbee successfully
    And inference completes successfully
    And all auth events are logged with fingerprints

  @p0 @auth @performance
  Scenario: AUTH-020 - Auth overhead < 1ms per request
    Given queen-rbee is running with auth enabled at "http://0.0.0.0:8080"
    And queen-rbee expects API token "perf-test-token"
    When I send 1000 authenticated requests
    Then average auth overhead is < 1ms per request
    And p99 auth latency is < 5ms
    And no performance degradation over time
