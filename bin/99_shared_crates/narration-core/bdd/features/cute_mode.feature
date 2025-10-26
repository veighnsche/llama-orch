# TEAM-307: Updated for n!() macro and context propagation
Feature: Cute Mode Narration
  As a developer debugging distributed systems
  I want whimsical children's book narration
  So that debugging can be delightful

  Background:
    Given the narration capture adapter is installed
    And the capture buffer is empty

  Scenario: Basic cute narration with n!() macro
    When I emit narration with n!("deploy", cute: "Tucked the model safely into GPU0's cozy VRAM blanket! 🛏️✨")
    Then the captured narration should include cute field
    And the cute field should contain "cozy VRAM blanket"
    And the cute field should contain "🛏️"

  Scenario: Cute narration with simple n!() macro
    When I emit narration with n!("action", "Human message")
    Then the captured narration should have 1 event
    And the human field should contain "Human message"

  Scenario: Cute narration with emoji
    When I narrate with:
      | field  | value                                                    |
      | actor  | vram-residency                                           |
      | action | seal                                                     |
      | target | llama-7b                                                 |
      | human  | Sealed model shard 'llama-7b' in 2048 MB VRAM on GPU 0   |
      | cute   | Tucked llama-7b safely into GPU0's warm 2GB nest! 🛏️✨   |
    Then the captured narration should have 1 event
    And the cute field should contain "warm 2GB nest"
    And the cute field should contain "🛏️✨"

  Scenario: Cute field is optional
    When I narrate without cute field
    Then the captured narration should have 1 event
    And the cute field should be absent


  Scenario: Multiple cute narrations
    When I narrate with cute field "First cute message! 🎀"
    And I narrate with cute field "Second cute message! ✨"
    And I narrate with cute field "Third cute message! 💝"
    Then the captured narration should have 3 events
    And event 1 cute field should contain "First cute message"
    And event 2 cute field should contain "Second cute message"
    And event 3 cute field should contain "Third cute message"

  Scenario: Cute narration length guideline
    When I narrate with cute field that is 150 characters long
    Then the cute field should be present
    And the cute field length should be at most 150

  Scenario: Cute mode with context (job_id)
    Given a narration context with job_id "job-cute-123"
    When I emit narration with n!("process", cute: "Processing cutely! 🎀") in context
    Then the captured narration should have 1 event
    And event 1 should have job_id "job-cute-123"
    And the cute field should contain "cutely"

  Scenario: Cute mode with correlation ID in context
    Given a narration context with correlation_id "req-cute-456"
    When I emit narration with n!("process", cute: "Processing request cutely! 🎀") in context
    Then the captured narration should have 1 event
    And event 1 should have correlation_id "req-cute-456"
    And the cute field should contain "cutely"

  Scenario: Cute narration with WARN level
    When I narrate at WARN level with cute field "Oh no! Something's not quite right! 😟"
    Then the captured narration should have 1 event
    And the cute field should contain "not quite right"
    And the cute field should contain "😟"

  Scenario: Cute narration with ERROR level
    When I narrate at ERROR level with cute field "Oops! The model had a little accident! 😟🔍"
    Then the captured narration should have 1 event
    And the cute field should contain "little accident"
    And the cute field should contain "😟🔍"
