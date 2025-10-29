/**
 * TEAM-351: Tests for message sender
 * 
 * Behavioral tests covering:
 * - Message sending
 * - Validation
 * - Statistics tracking
 * - Retry logic
 * - Error handling
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  createMessageSender,
  getSendStats,
  resetSendStats,
  isValidSenderConfig,
  type SenderConfig,
} from './sender'
import type { IframeMessage } from './types'

const mockPostMessage = vi.fn()

beforeEach(() => {
  vi.clearAllMocks()
  resetSendStats()
  
  // Mock window.parent as a different object
  Object.defineProperty(window, 'parent', {
    writable: true,
    configurable: true,
    value: {
      postMessage: mockPostMessage,
    },
  })
})

describe('@rbee/iframe-bridge - sender', () => {
  describe('isValidSenderConfig()', () => {
    it('should validate valid config', () => {
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
      }
      
      expect(isValidSenderConfig(config)).toBe(true)
    })

    it('should accept wildcard', () => {
      const config: SenderConfig = {
        targetOrigin: '*',
      }
      
      expect(isValidSenderConfig(config)).toBe(true)
    })

    it('should reject invalid origin format', () => {
      const config = {
        targetOrigin: 'not-a-url',
      }
      
      expect(isValidSenderConfig(config)).toBe(false)
    })

    it('should reject missing targetOrigin', () => {
      const config = {}
      expect(isValidSenderConfig(config)).toBe(false)
    })

    it('should reject null', () => {
      expect(isValidSenderConfig(null)).toBe(false)
    })
  })

  describe('createMessageSender()', () => {
    it('should create sender function', () => {
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
      }
      
      const sender = createMessageSender(config)
      expect(typeof sender).toBe('function')
    })

    it('should throw on invalid config', () => {
      const config = {
        targetOrigin: 'not-a-url',
      } as SenderConfig
      
      expect(() => createMessageSender(config)).toThrow('Invalid sender config')
    })

    it('should send valid message', () => {
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
      }
      
      const sender = createMessageSender(config)
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      const result = sender(message)
      
      expect(result).toBe(true)
      expect(mockPostMessage).toHaveBeenCalledTimes(1)
      expect(mockPostMessage).toHaveBeenCalledWith(message, 'http://localhost:3000')
    })

    it('should return false when parent is same window', () => {
      global.window.parent = global.window as any
      
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
      }
      
      const sender = createMessageSender(config)
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      const result = sender(message)
      expect(result).toBe(false)
    })

    it('should validate message when validate=true', () => {
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
        validate: true,
      }
      
      const sender = createMessageSender(config)
      const invalidMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        // missing command field
      } as any
      
      const result = sender(invalidMessage)
      expect(result).toBe(false)
      expect(mockPostMessage).not.toHaveBeenCalled()
    })

    it('should skip validation when validate=false', () => {
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
        validate: false,
      }
      
      const sender = createMessageSender(config)
      const invalidMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
      } as any
      
      const result = sender(invalidMessage)
      expect(result).toBe(true)
      expect(mockPostMessage).toHaveBeenCalled()
    })

    it('should handle postMessage errors', () => {
      mockPostMessage.mockImplementation(() => { throw new Error('Failed') })
      
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
      }
      
      const sender = createMessageSender(config)
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      const result = sender(message)
      expect(result).toBe(false)
    })

    it('should attempt retry on failure when retry=true', async () => {
      let callCount = 0
      mockPostMessage.mockImplementation(() => {
        callCount++
        if (callCount === 1) throw new Error('Failed')
      })
      
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
        retry: true,
      }
      
      const sender = createMessageSender(config)
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      const result = sender(message)
      expect(result).toBe(false)
      expect(mockPostMessage).toHaveBeenCalledTimes(1)
      
      // Wait for retry
      await new Promise(resolve => setTimeout(resolve, 150))
      expect(mockPostMessage).toHaveBeenCalledTimes(2)
    })
  })

  describe('Statistics tracking', () => {
    it('should track successful sends', () => {
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
      }
      
      const sender = createMessageSender(config)
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      sender(message)
      sender(message)
      
      const stats = getSendStats()
      expect(stats.total).toBe(2)
      expect(stats.success).toBe(2)
      expect(stats.failed).toBe(0)
    })

    it('should track failed sends', () => {
      mockPostMessage.mockImplementation(() => { throw new Error('Failed') })
      
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
      }
      
      const sender = createMessageSender(config)
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      sender(message)
      
      const stats = getSendStats()
      expect(stats.total).toBe(1)
      expect(stats.success).toBe(0)
      expect(stats.failed).toBe(1)
    })

    it('should reset statistics', () => {
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
      }
      
      const sender = createMessageSender(config)
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      sender(message)
      resetSendStats()
      
      const stats = getSendStats()
      expect(stats.total).toBe(0)
      expect(stats.success).toBe(0)
      expect(stats.failed).toBe(0)
    })
  })

  describe('Edge cases', () => {
    it('should handle very large messages', () => {
      // Reset mock to ensure it doesn't throw
      mockPostMessage.mockImplementation(() => {})
      
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
      }
      
      const sender = createMessageSender(config)
      const message: IframeMessage = {
        type: 'NARRATION_EVENT',
        source: 'test',
        timestamp: Date.now(),
        payload: {
          actor: 'test',
          action: 'test',
          human: 'a'.repeat(100000),
        },
      }
      
      const result = sender(message)
      expect(result).toBe(true)
    })

    it('should handle SSR environment', () => {
      // In jsdom, window is always defined, so we test the parent === window case instead
      Object.defineProperty(window, 'parent', {
        writable: true,
        configurable: true,
        value: window, // parent is same as window
      })
      
      const config: SenderConfig = {
        targetOrigin: 'http://localhost:3000',
      }
      
      const sender = createMessageSender(config)
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      const result = sender(message)
      expect(result).toBe(false)
    })
  })
})
