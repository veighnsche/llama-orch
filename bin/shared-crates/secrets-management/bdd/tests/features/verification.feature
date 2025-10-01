Feature: Secret Verification
  As a security-conscious system
  I want to verify secrets using timing-safe comparison
  So that I prevent timing attacks

  Scenario: Verify matching secret
    Given a secret file containing "correct-token"
    When I verify the secret with "correct-token"
    Then the verification should succeed

  Scenario: Reject non-matching secret
    Given a secret file containing "correct-token"
    When I verify the secret with "wrong-token"
    Then the verification should fail

  Scenario: Reject secret with different length
    Given a secret file containing "short"
    When I verify the secret with "much-longer-token"
    Then the verification should fail

  Scenario: Timing-safe comparison for early mismatch
    Given a secret file containing "correct-token-abc"
    When I verify the secret with "wrong-token-abc"
    Then the verification should fail

  Scenario: Timing-safe comparison for late mismatch
    Given a secret file containing "correct-token-abc"
    When I verify the secret with "correct-token-xyz"
    Then the verification should fail
