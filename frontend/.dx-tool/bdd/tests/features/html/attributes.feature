# Created by: TEAM-DX-003
# Test HTML attribute extraction

Feature: HTML Attributes
  As a frontend developer
  I want to extract element attributes
  So that I can verify component props

  Background:
    Given Storybook is running on port 6006

  Scenario: Get attributes for button
    When I get attributes for selector 'button'
    Then I should see attribute 'type'
