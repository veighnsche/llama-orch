Feature: http-util streaming decode (HTU-1003)
  As an adapter developer
  I want http-util to decode SSE-like token streams preserving ordering
  So that adapters can stream tokens reliably with minimal allocations

  Scenario: Decode started → token* → metrics? → end ordering
    Given a body stream with started token token metrics end
    When I decode with stream_decode
    Then ordering is preserved and token indices are strictly increasing
