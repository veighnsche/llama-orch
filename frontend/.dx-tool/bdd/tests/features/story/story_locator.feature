# Created by: TEAM-DX-003
# PRIORITY 0: Story file locator

Feature: Story File Locator
  As a frontend engineer without browser access
  I want to find which file defines a Storybook story
  So I can quickly make changes

  Background:
    Given Storybook is running on port 6006

  Scenario: Locate Button story file
    When I run story-file with URL "http://localhost:6006/story/stories-atoms-button-button-story-vue"
    Then I should see story file path "stories/atoms/Button/Button.story.vue"
    And I should see component file path "stories/atoms/Button/Button.vue"
    And the files should exist on disk

  Scenario: Locate story with variant ID
    When I run story-file with URL "http://localhost:6006/story/stories-atoms-button-button-story-vue?variantId=stories-atoms-button-button-story-vue-0"
    Then I should see story file path "stories/atoms/Button/Button.story.vue"
    And the variant ID should be ignored

  Scenario: Invalid story URL
    When I run story-file with URL "http://localhost:6006/invalid"
    Then I should see an error "Could not parse story path from URL"

  Scenario: Story file not found
    When I run story-file with URL "http://localhost:6006/story/nonexistent-story"
    Then I should see an error "Story file not found"
