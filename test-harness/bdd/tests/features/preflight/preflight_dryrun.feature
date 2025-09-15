Feature: Preflight DryRun default
  As an operator
  I want preflight runs to be non-destructive by default
  So that no side effects occur unless I explicitly commit

  Scenario: Preflight runs with DryRun default
    When when preflight runs
    Then side effects are not performed (DryRun is default)
