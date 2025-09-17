Feature: Session management
  # Traceability: ORCH-2004 (get session), ORCH-2005 (delete session)
  Scenario: Client queries and deletes session
    Given a session id
    When I query the session
    Then I receive session info with ttl_ms_remaining turns kv_bytes kv_warmth
    And I delete the session
    And I receive 204 No Content with correlation id
