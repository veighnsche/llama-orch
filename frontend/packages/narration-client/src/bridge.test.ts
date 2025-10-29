/**
 * TEAM-351: Tests for narration bridge
 * 
 * Behavioral tests covering:
 * - Message sending to parent
 * - Origin detection
 * - Retry logic
 * - Error handling
 * - Message validation
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest'
import {
  sendToParent,
  createStreamHandler,
} from './bridge'
import { SERVICES } from './config'

// Mock window.parent.postMessage
const mockPostMessage = vi.fn()
const originalWindow = global.window

beforeEach(() => {
  vi.clearAllMocks()
  
  // Setup window mock
  global.window = {
    parent: {
      postMessage: mockPostMessage,
    },
    location: {
      port: '7834', // Queen dev port
    },
  } as any
})

afterEach(() => {
  global.window = originalWindow
})

describe('@rbee/narration-client - bridge', () => {
  describe('sendToParent() - Basic sending', () => {
    it('should send valid narration event', () => {
      const event = {
        actor: 'test',
        action: 'test_action',
        human: 'Test message',
      }
      
      const result = sendToParent(event, SERVICES.queen)
      
      expect(result).toBe(true)
      expect(mockPostMessage).toHaveBeenCalledTimes(1)
      const [message, origin] = mockPostMessage.mock.calls[0]
      
      expect(message.type).toBe('NARRATION_EVENT')
      expect(message.payload).toEqual(event)
    })

    it('should include timestamp', () => {
      const event = {
        actor: 'test',
        action: 'test_action',
        human: 'Test',
      }
      
      const before = Date.now()
      sendToParent(event, SERVICES.queen)
      const after = Date.now()
      
      const [message] = mockPostMessage.mock.calls[0]
      expect(message.timestamp).toBeGreaterThanOrEqual(before)
      expect(message.timestamp).toBeLessThanOrEqual(after)
    })

    it('should include protocol version', () => {
      const event = {
        actor: 'test',
        action: 'test_action',
        human: 'Test',
      }
      
      sendToParent(event, SERVICES.queen)
      
      const [message] = mockPostMessage.mock.calls[0]
      expect(message.version).toBe('1.0.0')
    })
  })

  describe('sendToParent() - Origin detection', () => {
    it('should use keeper dev origin for dev port', () => {
      global.window.location.port = '7834' // Queen dev
      
      sendToParent({ actor: 'test', action: 'test', human: 'Test' }, SERVICES.queen)
      
      const [, origin] = mockPostMessage.mock.calls[0]
      expect(origin).toBe('http://localhost:5173')
    })

    it('should use wildcard for prod port', () => {
      global.window.location.port = '7833' // Queen prod
      
      sendToParent({ actor: 'test', action: 'test', human: 'Test' }, SERVICES.queen)
      
      const [, origin] = mockPostMessage.mock.calls[0]
      expect(origin).toBe('*')
    })

    it('should handle hive dev port', () => {
      global.window.location.port = '7836' // Hive dev
      
      sendToParent({ actor: 'test', action: 'test', human: 'Test' }, SERVICES.hive)
      
      const [, origin] = mockPostMessage.mock.calls[0]
      expect(origin).toBe('http://localhost:5173')
    })

    it('should handle worker dev port', () => {
      global.window.location.port = '7837' // Worker dev
      
      sendToParent({ actor: 'test', action: 'test', human: 'Test' }, SERVICES.worker)
      
      const [, origin] = mockPostMessage.mock.calls[0]
      expect(origin).toBe('http://localhost:5173')
    })
  })

  describe('sendToParent() - Retry logic', () => {
    it('should attempt retry on failure', async () => {
      let callCount = 0
      mockPostMessage.mockImplementation(() => {
        callCount++
        if (callCount === 1) throw new Error('Failed')
        // Second call succeeds
      })
      
      const result = sendToParent(
        { actor: 'test', action: 'test', human: 'Test' },
        SERVICES.queen,
        { retry: true }
      )
      
      expect(result).toBe(false)
      expect(mockPostMessage).toHaveBeenCalledTimes(1)
      
      // Wait for retry setTimeout
      await new Promise(resolve => setTimeout(resolve, 150))
      expect(mockPostMessage).toHaveBeenCalledTimes(2)
    })

    it('should not retry when retry=false', () => {
      mockPostMessage.mockImplementation(() => { throw new Error('Failed') })
      
      const result = sendToParent(
        { actor: 'test', action: 'test', human: 'Test' },
        SERVICES.queen,
        { retry: false }
      )
      
      expect(result).toBe(false)
      expect(mockPostMessage).toHaveBeenCalledTimes(1)
    })
  })

  describe('sendToParent() - Error handling', () => {
    it('should handle postMessage errors gracefully', () => {
      mockPostMessage.mockImplementation(() => { throw new Error('Failed') })
      
      expect(() => {
        sendToParent({ actor: 'test', action: 'test', human: 'Test' }, SERVICES.queen)
      }).not.toThrow()
    })

    it('should handle missing window.parent', () => {
      global.window.parent = global.window as any
      
      expect(() => {
        sendToParent({ actor: 'test', action: 'test', human: 'Test' }, SERVICES.queen)
      }).not.toThrow()
    })

    it('should not send when parent is same window', () => {
      global.window.parent = global.window as any
      
      const result = sendToParent({ actor: 'test', action: 'test', human: 'Test' }, SERVICES.queen)
      
      expect(result).toBe(false)
      expect(mockPostMessage).not.toHaveBeenCalled()
    })
  })

  describe('createStreamHandler() - Stream handling', () => {
    it('should parse and send valid lines', () => {
      const handler = createStreamHandler(SERVICES.queen)
      
      handler('{"actor":"test","action":"test","human":"Test"}')
      
      expect(mockPostMessage).toHaveBeenCalledTimes(1)
    })

    it('should skip [DONE] marker', () => {
      const handler = createStreamHandler(SERVICES.queen)
      
      handler('[DONE]')
      
      expect(mockPostMessage).not.toHaveBeenCalled()
    })

    it('should skip empty lines', () => {
      const handler = createStreamHandler(SERVICES.queen)
      
      handler('')
      handler('   ')
      
      expect(mockPostMessage).not.toHaveBeenCalled()
    })

    it('should handle multiple lines', () => {
      const handler = createStreamHandler(SERVICES.queen)
      
      handler('{"actor":"test","action":"test1","human":"Test1"}')
      handler('{"actor":"test","action":"test2","human":"Test2"}')
      
      expect(mockPostMessage).toHaveBeenCalledTimes(2)
    })

    it('should call onLocal callback', () => {
      const onLocal = vi.fn()
      const handler = createStreamHandler(SERVICES.queen, onLocal)
      
      handler('{"actor":"test","action":"test","human":"Test"}')
      
      expect(onLocal).toHaveBeenCalledTimes(1)
      expect(onLocal).toHaveBeenCalledWith({
        actor: 'test',
        action: 'test',
        human: 'Test',
      })
    })

    it('should send to parent and call onLocal', () => {
      const onLocal = vi.fn()
      const handler = createStreamHandler(SERVICES.queen, onLocal)
      
      handler('{"actor":"test","action":"test","human":"Test"}')
      
      expect(mockPostMessage).toHaveBeenCalledTimes(1)
      expect(onLocal).toHaveBeenCalledTimes(1)
    })
  })

  // Note: Stats tracking not implemented in bridge module

  describe('Edge cases', () => {
    it('should handle SSR environment', () => {
      const originalWindow = global.window
      delete (global as any).window
      
      const result = sendToParent({ actor: 'test', action: 'test', human: 'Test' }, SERVICES.queen)
      
      expect(result).toBe(false)
      
      global.window = originalWindow
    })

    it('should handle very large messages', () => {
      mockPostMessage.mockImplementation(() => {}) // Reset to success
      const largeMessage = 'a'.repeat(100000)
      
      const result = sendToParent({ actor: 'test', action: 'test', human: largeMessage }, SERVICES.queen)
      
      expect(result).toBe(true)
      expect(mockPostMessage).toHaveBeenCalledTimes(1)
    })

    it('should handle special characters in messages', () => {
      mockPostMessage.mockImplementation(() => {}) // Reset to success
      const result = sendToParent({
        actor: 'test',
        action: 'test',
        human: 'Test\n\t"quote"',
      }, SERVICES.queen)
      
      expect(result).toBe(true)
      expect(mockPostMessage).toHaveBeenCalledTimes(1)
    })
  })
})
