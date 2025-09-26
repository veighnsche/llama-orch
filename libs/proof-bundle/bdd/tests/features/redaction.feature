Feature: Redaction non-goal
  Verify PB-1007: the crate does not redact content; callers are responsible.

  Background:
    Given a proof bundle of type "integration" with run id "BDD-REDACT-1"

  Scenario: Markdown content is written verbatim (no redaction)
    When I overwrite markdown "logs/run_log_redacted.md" with body "token=PLACEHOLDER_SECRET"
    Then file "logs/run_log_redacted.md" contains "token=PLACEHOLDER_SECRET"

  Scenario: JSON content is written verbatim (no redaction)
    When I write json file base "logs/raw" value {"token":"PLACEHOLDER_SECRET"}
    Then json file "logs/raw.json" has field "token" equals "PLACEHOLDER_SECRET"
