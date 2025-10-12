# Created by: TEAM-DX-003
# Test CSS selector style extraction

Feature: CSS Selector Styles
  As a frontend developer
  I want to extract computed styles for selectors
  So that I can verify styling is applied correctly

  Background:
    Given Storybook is running on port 6006

  Scenario: Get styles for button
    When I get styles for selector 'button'
    Then I should see style 'cursor' with value 'pointer'

  Scenario: Get styles for body element
    When I get styles for selector 'body'
    Then I should see style 'display' with value 'block'
