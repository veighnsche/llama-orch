/**
 * TEAM-351: Tests for iframe message types
 * 
 * Behavioral tests covering:
 * - Type guards and validation
 * - Message structure validation
 * - Detailed validation feedback
 * - Edge cases
 */

import { describe, it, expect } from 'vitest'
import {
  isValidBaseMessage,
  isValidIframeMessage,
  validateMessage,
  type NarrationMessage,
  type CommandMessage,
  type ResponseMessage,
  type ErrorMessage,
} from './types'

describe('@rbee/iframe-bridge - types', () => {
  describe('isValidBaseMessage()', () => {
    it('should validate valid base message', () => {
      const msg = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
      }
      
      expect(isValidBaseMessage(msg)).toBe(true)
    })

    it('should reject null', () => {
      expect(isValidBaseMessage(null)).toBe(false)
    })

    it('should reject non-object', () => {
      expect(isValidBaseMessage('string')).toBe(false)
      expect(isValidBaseMessage(123)).toBe(false)
      expect(isValidBaseMessage(true)).toBe(false)
    })

    it('should reject missing type', () => {
      const msg = {
        source: 'test',
        timestamp: Date.now(),
      }
      
      expect(isValidBaseMessage(msg)).toBe(false)
    })

    it('should reject missing source', () => {
      const msg = {
        type: 'NARRATION_EVENT',
        timestamp: Date.now(),
      }
      
      expect(isValidBaseMessage(msg)).toBe(false)
    })

    it('should reject missing timestamp', () => {
      const msg = {
        type: 'NARRATION_EVENT',
        source: 'test',
      }
      
      expect(isValidBaseMessage(msg)).toBe(false)
    })

    it('should reject wrong type field type', () => {
      const msg = {
        type: 123,
        source: 'test',
        timestamp: Date.now(),
      }
      
      expect(isValidBaseMessage(msg)).toBe(false)
    })

    it('should accept optional fields', () => {
      const msg = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
        id: 'msg-123',
        version: '1.0.0',
      }
      
      expect(isValidBaseMessage(msg)).toBe(true)
    })
  })

  describe('isValidIframeMessage() - NARRATION_EVENT', () => {
    it('should validate valid narration message', () => {
      const msg: NarrationMessage = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
        payload: {
          actor: 'test',
          action: 'test_action',
          human: 'Test message',
        },
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should reject narration without payload', () => {
      const msg = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
      }
      
      expect(isValidIframeMessage(msg)).toBe(false)
    })

    it('should reject narration with missing actor', () => {
      const msg = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
        payload: {
          action: 'test_action',
          human: 'Test',
        },
      }
      
      expect(isValidIframeMessage(msg)).toBe(false)
    })

    it('should reject narration with missing action', () => {
      const msg = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
        payload: {
          actor: 'test',
          human: 'Test',
        },
      }
      
      expect(isValidIframeMessage(msg)).toBe(false)
    })

    it('should reject narration with missing human', () => {
      const msg = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
        payload: {
          actor: 'test',
          action: 'test_action',
        },
      }
      
      expect(isValidIframeMessage(msg)).toBe(false)
    })

    it('should accept narration with additional fields', () => {
      const msg = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
        payload: {
          actor: 'test',
          action: 'test_action',
          human: 'Test',
          level: 'info',
          job_id: '123',
        },
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })
  })

  describe('isValidIframeMessage() - COMMAND', () => {
    it('should validate valid command message', () => {
      const msg: CommandMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test_command',
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should accept command with args', () => {
      const msg: CommandMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test_command',
        args: { key: 'value' },
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should reject command without command field', () => {
      const msg = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
      }
      
      expect(isValidIframeMessage(msg)).toBe(false)
    })
  })

  describe('isValidIframeMessage() - RESPONSE', () => {
    it('should validate valid response message', () => {
      const msg: ResponseMessage = {
        type: 'RESPONSE',
        source: 'test',
        timestamp: Date.now(),
        requestId: 'req-123',
        success: true,
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should accept response with data', () => {
      const msg: ResponseMessage = {
        type: 'RESPONSE',
        source: 'test',
        timestamp: Date.now(),
        requestId: 'req-123',
        success: true,
        data: { result: 'ok' },
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should accept response with error', () => {
      const msg: ResponseMessage = {
        type: 'RESPONSE',
        source: 'test',
        timestamp: Date.now(),
        requestId: 'req-123',
        success: false,
        error: 'Failed',
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should reject response without requestId', () => {
      const msg = {
        type: 'RESPONSE',
        source: 'test',
        timestamp: Date.now(),
        success: true,
      }
      
      expect(isValidIframeMessage(msg)).toBe(false)
    })

    it('should reject response without success', () => {
      const msg = {
        type: 'RESPONSE',
        source: 'test',
        timestamp: Date.now(),
        requestId: 'req-123',
      }
      
      expect(isValidIframeMessage(msg)).toBe(false)
    })
  })

  describe('isValidIframeMessage() - ERROR', () => {
    it('should validate valid error message', () => {
      const msg: ErrorMessage = {
        type: 'ERROR',
        source: 'test',
        timestamp: Date.now(),
        error: 'Something failed',
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should accept error with code', () => {
      const msg: ErrorMessage = {
        type: 'ERROR',
        source: 'test',
        timestamp: Date.now(),
        error: 'Failed',
        code: 'ERR_TEST',
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should accept error with details', () => {
      const msg: ErrorMessage = {
        type: 'ERROR',
        source: 'test',
        timestamp: Date.now(),
        error: 'Failed',
        details: { reason: 'timeout' },
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should reject error without error field', () => {
      const msg = {
        type: 'ERROR',
        source: 'test',
        timestamp: Date.now(),
      }
      
      expect(isValidIframeMessage(msg)).toBe(false)
    })
  })

  describe('isValidIframeMessage() - Invalid types', () => {
    it('should reject unknown message type', () => {
      const msg = {
        type: 'UNKNOWN',
        source: 'test',
        timestamp: Date.now(),
      }
      
      expect(isValidIframeMessage(msg)).toBe(false)
    })

    it('should reject invalid base message', () => {
      const msg = {
        type: 'NARRATION_EVENT',
        // missing source and timestamp
      }
      
      expect(isValidIframeMessage(msg)).toBe(false)
    })
  })

  describe('validateMessage() - Detailed validation', () => {
    it('should validate valid message', () => {
      const msg = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      const result = validateMessage(msg)
      expect(result.valid).toBe(true)
      expect(result.error).toBeUndefined()
    })

    it('should reject null', () => {
      const result = validateMessage(null)
      expect(result.valid).toBe(false)
      expect(result.error).toBe('Message must be an object')
    })

    it('should reject non-object', () => {
      const result = validateMessage('string')
      expect(result.valid).toBe(false)
      expect(result.error).toBe('Message must be an object')
    })

    it('should report missing fields', () => {
      const msg = {
        type: 'COMMAND',
        // missing source and timestamp
      }
      
      const result = validateMessage(msg)
      expect(result.valid).toBe(false)
      expect(result.error).toBe('Missing required fields')
      expect(result.missing).toEqual(['source', 'timestamp'])
    })

    it('should report invalid structure', () => {
      const msg = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        // missing command field
      }
      
      const result = validateMessage(msg)
      expect(result.valid).toBe(false)
      expect(result.error).toContain('Invalid message type or structure')
    })
  })

  describe('Edge cases', () => {
    it('should handle very large payloads', () => {
      const msg: NarrationMessage = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
        payload: {
          actor: 'test',
          action: 'test',
          human: 'a'.repeat(100000),
        },
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should handle special characters', () => {
      const msg: NarrationMessage = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
        payload: {
          actor: 'test',
          action: 'test',
          human: 'Test\n\t"quote"',
        },
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })

    it('should handle unicode', () => {
      const msg: NarrationMessage = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
        payload: {
          actor: 'test',
          action: 'test',
          human: 'Test 🎉 emoji',
        },
      }
      
      expect(isValidIframeMessage(msg)).toBe(true)
    })
  })
})
