/**
 * TEAM-351: Tests for narration parser
 * 
 * Behavioral tests covering:
 * - Valid JSON parsing
 * - [DONE] marker handling
 * - Empty/whitespace handling
 * - SSE format handling
 * - Malformed JSON handling
 * - Statistics tracking
 */

import { describe, it, expect, beforeEach } from 'vitest'
import {
  parseNarrationLine,
  getParseStats,
  resetParseStats,
} from './parser'

describe('@rbee/narration-client - parser', () => {
  beforeEach(() => {
    resetParseStats()
  })

  describe('parseNarrationLine() - Valid JSON', () => {
    it('should parse valid narration event', () => {
      const line = '{"actor":"test","action":"test_action","human":"Test message"}'
      const result = parseNarrationLine(line)
      
      expect(result).toEqual({
        actor: 'test',
        action: 'test_action',
        human: 'Test message',
      })
    })

    it('should parse event with data: prefix', () => {
      const line = 'data: {"actor":"test","action":"test_action","human":"Test"}'
      const result = parseNarrationLine(line)
      
      expect(result).toEqual({
        actor: 'test',
        action: 'test_action',
        human: 'Test',
      })
    })

    it('should parse event without data: prefix', () => {
      const line = '{"actor":"test","action":"test_action","human":"Test"}'
      const result = parseNarrationLine(line)
      
      expect(result).not.toBeNull()
      expect(result?.actor).toBe('test')
    })

    it('should handle formatted field', () => {
      const line = '{"actor":"test","action":"test","human":"Test","formatted":"Formatted"}'
      const result = parseNarrationLine(line)
      
      expect(result?.formatted).toBe('Formatted')
    })

    it('should handle optional fields', () => {
      const line = '{"actor":"test","action":"test","human":"Test","level":"info","job_id":"123"}'
      const result = parseNarrationLine(line)
      
      expect(result?.level).toBe('info')
      expect(result?.job_id).toBe('123')
    })
  })

  describe('parseNarrationLine() - [DONE] marker', () => {
    it('should skip [DONE] marker', () => {
      const result = parseNarrationLine('[DONE]')
      expect(result).toBeNull()
    })

    it('should skip [DONE] with whitespace', () => {
      const result = parseNarrationLine('  [DONE]  ')
      expect(result).toBeNull()
    })

    it('should count [DONE] markers', () => {
      parseNarrationLine('[DONE]')
      const stats = getParseStats()
      
      expect(stats.doneMarkers).toBe(1)
    })
  })

  describe('parseNarrationLine() - Empty/whitespace', () => {
    it('should skip empty string', () => {
      const result = parseNarrationLine('')
      expect(result).toBeNull()
    })

    it('should skip whitespace-only line', () => {
      const result = parseNarrationLine('   ')
      expect(result).toBeNull()
    })

    it('should skip empty data: line', () => {
      const result = parseNarrationLine('data: ')
      expect(result).toBeNull()
    })

    it('should count empty lines', () => {
      parseNarrationLine('')
      parseNarrationLine('   ')
      const stats = getParseStats()
      
      expect(stats.emptyLines).toBe(2)
    })
  })

  describe('parseNarrationLine() - SSE format', () => {
    it('should skip SSE comment lines', () => {
      const result = parseNarrationLine(': this is a comment')
      expect(result).toBeNull()
    })

    it('should skip event: lines', () => {
      const result = parseNarrationLine('event: message')
      expect(result).toBeNull()
    })

    it('should skip id: lines', () => {
      const result = parseNarrationLine('id: 123')
      expect(result).toBeNull()
    })

    it('should skip retry: lines', () => {
      const result = parseNarrationLine('retry: 1000')
      expect(result).toBeNull()
    })

    it('should handle data: with JSON', () => {
      const line = 'data: {"actor":"test","action":"test","human":"Test"}'
      const result = parseNarrationLine(line)
      
      expect(result).not.toBeNull()
    })
  })

  describe('parseNarrationLine() - Malformed JSON', () => {
    it('should return null for invalid JSON', () => {
      const result = parseNarrationLine('{invalid json}')
      expect(result).toBeNull()
    })

    it('should return null for incomplete JSON', () => {
      const result = parseNarrationLine('{"actor":"test"')
      expect(result).toBeNull()
    })

    it('should return null for missing required fields', () => {
      const result = parseNarrationLine('{"actor":"test"}')
      expect(result).toBeNull()
    })

    it('should return null for wrong field types', () => {
      const result = parseNarrationLine('{"actor":123,"action":"test","human":"test"}')
      expect(result).toBeNull()
    })

    it('should count malformed JSON as failed', () => {
      parseNarrationLine('{invalid}')
      const stats = getParseStats()
      
      expect(stats.failed).toBe(1)
    })
  })

  describe('getParseStats() - Statistics tracking', () => {
    it('should track successful parses', () => {
      parseNarrationLine('{"actor":"test","action":"test","human":"Test"}')
      parseNarrationLine('{"actor":"test","action":"test","human":"Test"}')
      const stats = getParseStats()
      
      expect(stats.total).toBe(2)
      expect(stats.success).toBe(2)
    })

    it('should track failed parses', () => {
      parseNarrationLine('{invalid}')
      parseNarrationLine('{also invalid}')
      const stats = getParseStats()
      
      expect(stats.failed).toBe(2)
    })

    it('should track done markers', () => {
      parseNarrationLine('[DONE]')
      parseNarrationLine('[DONE]')
      const stats = getParseStats()
      
      expect(stats.doneMarkers).toBe(2)
    })

    it('should track empty lines', () => {
      parseNarrationLine('')
      parseNarrationLine('   ')
      const stats = getParseStats()
      
      expect(stats.emptyLines).toBe(2)
    })

    it('should calculate correct totals', () => {
      parseNarrationLine('{"actor":"test","action":"test","human":"Test"}') // success
      parseNarrationLine('{invalid}') // failed
      parseNarrationLine('[DONE]') // done marker
      parseNarrationLine('') // empty
      const stats = getParseStats()
      
      expect(stats.total).toBe(4)
      expect(stats.success).toBe(1)
      expect(stats.failed).toBe(1)
      expect(stats.doneMarkers).toBe(1)
      expect(stats.emptyLines).toBe(1)
    })
  })

  describe('resetParseStats() - Statistics reset', () => {
    it('should reset all statistics to zero', () => {
      parseNarrationLine('{"actor":"test","action":"test","human":"Test"}')
      parseNarrationLine('{invalid}')
      
      resetParseStats()
      const stats = getParseStats()
      
      expect(stats.total).toBe(0)
      expect(stats.success).toBe(0)
      expect(stats.failed).toBe(0)
      expect(stats.doneMarkers).toBe(0)
      expect(stats.emptyLines).toBe(0)
    })
  })

  describe('Edge cases', () => {
    it('should handle very long lines', () => {
      const longMessage = 'a'.repeat(10000)
      const line = `{"actor":"test","action":"test","human":"${longMessage}"}`
      const result = parseNarrationLine(line)
      
      expect(result?.human).toBe(longMessage)
    })

    it('should handle special characters', () => {
      const line = '{"actor":"test","action":"test","human":"Test\\n\\t\\"quote\\""}'
      const result = parseNarrationLine(line)
      
      expect(result).not.toBeNull()
    })

    it('should handle unicode', () => {
      const line = '{"actor":"test","action":"test","human":"Test 🎉 emoji"}'
      const result = parseNarrationLine(line)
      
      expect(result?.human).toBe('Test 🎉 emoji')
    })

    it('should handle nested objects gracefully', () => {
      const line = '{"actor":"test","action":"test","human":"Test","extra":{"nested":"value"}}'
      const result = parseNarrationLine(line)
      
      expect(result).not.toBeNull()
    })
  })
})
