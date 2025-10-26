# TEAM-307: Updated for n!() macro and context propagation
Feature: Story Mode Narration
  As a developer debugging distributed systems
  I want dialogue-based narration
  So that multi-service interactions are crystal clear

  Background:
    Given the narration capture adapter is installed
    And the capture buffer is empty

  Scenario: Basic story narration with n!() macro
    When I emit narration with n!("check", story: "\"Do you have 2GB VRAM?\" asked orchestratord. \"Yes!\" replied pool-managerd.")
    Then the captured narration should include story field
    And the story field should contain "asked orchestratord"
    And the story field should contain "replied pool-managerd"

  Scenario: Story narration with context
    Given a narration context with job_id "job-story-123"
    When I emit narration with n!("dialogue", story: "\"Starting job!\" announced the system.") in context
    Then the captured narration should have 1 event
    And event 1 should have job_id "job-story-123"
    And the story field should contain "announced the system"

  Scenario: Story mode is optional
    When I narrate without story field
    Then the captured narration should have 1 event
    And the story field should be absent

  Scenario: Story with multiple speakers
    When I narrate with:
      | field  | value                                                                                    |
      | actor  | orchestratord                                                                            |
      | action | capacity_check                                                                           |
      | target | all-pools                                                                                |
      | human  | Checking capacity across all pools                                                       |
      | story  | \"Who has capacity?\" asked orchestratord. \"I do!\" said pool-1. \"Me too!\" said pool-2. |
    Then the story field should contain "asked orchestratord"
    And the story field should contain "said pool-1"
    And the story field should contain "said pool-2"

  Scenario: Story with error dialogue
    When I narrate with story field "\"Processing job-999...\" said worker. Suddenly: \"ERROR! Out of memory!\" \"What happened?\" asked orchestratord. \"CUDA OOM,\" replied worker sadly."
    Then the story field should contain "said worker"
    And the story field should contain "asked orchestratord"
    And the story field should contain "replied worker sadly"

  Scenario: Story with redaction
    When I narrate with story field "\"Here's the token: Bearer abc123\" said auth-service."
    Then the story field should contain "[REDACTED]"
    And the story field should not contain "abc123"

  Scenario: Story narration length guideline
    When I narrate with story field that is 200 characters long
    Then the story field should be present
    And the story field length should be at most 200

  Scenario: Triple narration (human + cute + story)
    When I narrate with:
      | field  | value                                                                                |
      | actor  | orchestratord                                                                        |
      | action | vram_request                                                                         |
      | target | pool-managerd-3                                                                      |
      | human  | Requesting 2048 MB VRAM on GPU 0 for model 'llama-7b'                                |
      | cute   | Orchestratord politely asks pool-managerd-3 for a cozy 2GB spot! 🏠                  |
      | story  | \"Do you have 2GB VRAM on GPU0?\" asked orchestratord. \"Yes!\" replied pool-managerd-3. |
    Then the captured narration should have 1 event
    And the human field should contain "Requesting 2048 MB VRAM"
    And the cute field should contain "cozy 2GB spot"
    And the story field should contain "asked orchestratord"

  Scenario: Story mode with correlation ID tracking
    Given a correlation ID "req-story-456"
    When I narrate with story field "\"Can you handle this?\" asked orchestratord." and correlation ID
    Then the captured narration should include correlation ID "req-story-456"
    And the story field should contain "Can you handle this"

  Scenario: Story with success celebration
    When I narrate with story field "\"Job done!\" announced worker proudly. \"How'd it go?\" asked orchestratord. \"Perfect! 150 tokens in 2.5s!\""
    Then the story field should contain "announced worker proudly"
    And the story field should contain "asked orchestratord"
    And the story field should contain "Perfect!"

  Scenario: Story mode at different levels
    When I narrate at WARN level with story field "\"Capacity low!\" warned pool-managerd."
    Then the captured narration should have 1 event
    And the story field should contain "warned pool-managerd"

  Scenario: Story with heartbeat dialogue
    When I narrate with story field "\"You still alive?\" asked pool-managerd-3. \"Yep, all good here!\" replied worker-gpu0-r1."
    Then the story field should contain "asked pool-managerd-3"
    And the story field should contain "replied worker-gpu0-r1"
