Feature: Apply Commit mode
  As an operator
  I want to run preflight then apply in Commit mode
  So that requested changes are applied only when I choose to commit

  Scenario: Preflight then Apply in Commit mode
    When I run preflight and apply in Commit mode
    Then side effects are not performed (DryRun is default)

  Scenario: Preflight detects a critical violation and blocks apply
    Given a critical compatibility violation is detected in preflight
    When I run the engine with default policy
    Then side effects are not performed (DryRun is default)
