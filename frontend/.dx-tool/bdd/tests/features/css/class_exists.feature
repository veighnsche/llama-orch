# Created by: TEAM-DX-003
# Test CSS class existence checking

Feature: CSS Class Existence
  As a frontend developer
  I want to check if CSS classes exist in Storybook
  So that I can verify Tailwind is generating my classes

  Background:
    Given Storybook is running on port 6006

  Scenario: Check for existing Tailwind class
    When I check if class 'flex' exists
    Then the class should exist

  Scenario: Check for non-existent class
    When I check if class 'nonexistent-class-xyz-12345' exists
    Then the class should not exist

  Scenario: Check for common utility class
    When I check if class 'cursor-pointer' exists
    Then the class should exist
