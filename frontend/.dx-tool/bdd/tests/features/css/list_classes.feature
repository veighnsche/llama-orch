# Created by: TEAM-DX-003
# Test CSS class listing

Feature: CSS Class Listing
  As a frontend developer
  I want to list all classes for a selector
  So that I can verify component styling

  Background:
    Given Storybook is running on port 6006

  Scenario: List classes for button
    When I list classes for selector 'button'
    Then I should see at least 1 class
