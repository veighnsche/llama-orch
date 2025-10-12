# Created by: TEAM-DX-003
# Test HTML selector queries

Feature: HTML Selector Queries
  As a frontend developer
  I want to query DOM structure
  So that I can verify components are rendering

  Background:
    Given Storybook is running on port 6006

  Scenario: Query button elements
    When I query selector 'button'
    Then I should find at least 1 element
    And the element tag should be 'button'

  Scenario: Query body element
    When I query selector 'body'
    Then I should find 1 element
    And the element tag should be 'body'
