Feature: Watchdog timeouts
  As an operator
  I want the watchdog to abort stuck or long-running tasks
  So that capacity is protected and SLAs are enforced

  Scenario: Watchdog aborts a task exceeding thresholds
    Given a running task exceeding watchdog thresholds
    Then the watchdog aborts the task
