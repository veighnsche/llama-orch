# Created by: TEAM-DX-003
# Test DOM tree extraction

Feature: DOM Tree
  As a frontend developer
  I want to visualize DOM structure
  So that I can understand component hierarchy

  Background:
    Given Storybook is running on port 6006

  Scenario: Get DOM tree for body
    When I get DOM tree for selector 'body' with depth 2
    Then the DOM tree should contain 'body'
