Feature: http-util retries/backoff policy (HTU-1002)
  As an adapter developer
  I want http-util to perform capped jittered retries for idempotent requests
  So that transient failures are handled within sane limits

  Background:
    Given no special http-util configuration

  Scenario: Transient 503 then success follows default policy
    Given a transient upstream that returns 503 then succeeds
    When I invoke with_retries around an idempotent request
    Then attempts follow default policy base 100ms multiplier 2.0 cap 2s max attempts 4

  Scenario: Non-retriable 400 does not retry
    Given an upstream that returns 400 Bad Request
    When I invoke with_retries around an idempotent request
    Then no retry occurs
