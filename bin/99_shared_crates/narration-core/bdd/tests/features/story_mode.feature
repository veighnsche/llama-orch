Feature: Story Mode Behaviors
  As a developer using narration-core
  I want dialogue-focused story narration
  So that I can understand conversations between components

  Background:
    Given a clean capture adapter

  Scenario: B-STORY-001 - Story field is optional
    When I narrate with story "\"Hello!\" said orchestratord."
    Then the captured narration has story field

  Scenario: B-STORY-002 - Story field can be None
    When I narrate without story field
    Then the captured narration has no story field

  Scenario: B-STORY-003 - Story field supports dialogue
    When I narrate with story "\"Do you have capacity?\" asked orchestratord. \"Yes!\" replied pool-managerd."
    Then the captured story includes "asked orchestratord"
    And the captured story includes "replied pool-managerd"

  Scenario: B-STORY-004 - Story field supports multi-party dialogue
    When I narrate with story "\"Who has capacity?\" asked orchestratord. \"I do!\" said pool-1. \"Me too!\" said pool-2."
    Then the captured story includes "asked orchestratord"
    And the captured story includes "said pool-1"
    And the captured story includes "said pool-2"

  Scenario: B-STORY-005 - Story field is redacted for secrets
    When I narrate with story "\"Here's the token: Bearer abc123\" said worker."
    Then the captured story includes "[REDACTED]"
    And the captured story does not include "Bearer abc123"

  Scenario: B-STORY-006 - Story field supports request-response pattern
    When I narrate with story "\"Can you handle job-456?\" asked orchestratord. \"Absolutely!\" replied worker-gpu0-r1."
    Then the captured story includes "asked orchestratord"
    And the captured story includes "replied worker-gpu0-r1"
    And the captured story includes "job-456"

  Scenario: B-STORY-007 - Story field supports error dialogue
    When I narrate with story "\"Processing...\" said worker. Suddenly: \"ERROR! Out of memory!\""
    Then the captured story includes "Processing"
    And the captured story includes "ERROR"
    And the captured story includes "Out of memory"

  Scenario: B-STORY-008 - Story field supports success celebration
    When I narrate with story "\"Job done!\" announced worker. \"Excellent!\" replied orchestratord."
    Then the captured story includes "Job done"
    And the captured story includes "announced worker"
    And the captured story includes "Excellent"

  Scenario: B-STORY-009 - Story field can coexist with human and cute
    When I narrate with all three modes:
      | human | Requesting 2GB VRAM on GPU 0 |
      | cute  | Orchestratord asks for a cozy spot! 🏠 |
      | story | "Do you have 2GB?" asked orchestratord. "Yes!" replied pool-managerd. |
    Then the captured narration has human field
    And the captured narration has cute field
    And the captured narration has story field

  Scenario: B-STORY-010 - Story field supports heartbeat dialogue
    When I narrate with story "\"You still alive?\" asked pool-managerd. \"Yep, all good!\" replied worker."
    Then the captured story includes "asked pool-managerd"
    And the captured story includes "replied worker"

  Scenario: B-STORY-011 - Story field supports denial responses
    When I narrate with story "\"Do you have 4GB VRAM?\" asked orchestratord. \"No,\" replied pool-managerd sadly, \"only 512MB free.\""
    Then the captured story includes "asked orchestratord"
    And the captured story includes "No"
    And the captured story includes "512MB free"

  Scenario: B-STORY-012 - Story field supports announcement pattern
    When I narrate with story "\"I'm ready!\" announced worker-gpu0-r1. \"Great!\" said pool-managerd."
    Then the captured story includes "announced worker-gpu0-r1"
    And the captured story includes "said pool-managerd"

  Scenario: B-STORY-013 - Story field is emitted in structured logs
    When I narrate with story "\"Hello!\" said orchestratord."
    Then the tracing event includes story field

  Scenario: B-STORY-014 - Story field supports quoted speech with single quotes
    When I narrate with story "'Can you help?' asked orchestratord. 'Sure!' replied worker."
    Then the captured story includes "asked orchestratord"
    And the captured story includes "replied worker"

  Scenario: B-STORY-015 - Story field supports mixed quote styles
    When I narrate with story "\"Do you have capacity?\" asked orchestratord. 'Yes!' replied pool-managerd."
    Then the captured story includes "asked orchestratord"
    And the captured story includes "replied pool-managerd"

  Scenario: B-STORY-016 - CaptureAdapter can assert story presence
    When I narrate with story "\"Hello!\" said orchestratord."
    Then assert_story_present succeeds

  Scenario: B-STORY-017 - CaptureAdapter can assert story includes substring
    When I narrate with story "\"Do you have capacity?\" asked orchestratord."
    Then assert_story_includes "asked orchestratord" succeeds

  Scenario: B-STORY-018 - CaptureAdapter can assert dialogue presence
    When I narrate with story "\"Hello!\" said orchestratord."
    Then assert_story_has_dialogue succeeds

  Scenario: B-STORY-019 - Story field without dialogue fails dialogue assertion
    When I narrate with story "Worker is processing the job"
    Then assert_story_has_dialogue fails

  Scenario: B-STORY-020 - Story field supports cancellation dialogue
    When I narrate with story "\"Can you cancel job-789?\" asked orchestratord. \"Sure thing!\" replied worker, \"Stopping now.\""
    Then the captured story includes "cancel job-789"
    And the captured story includes "Stopping now"
