/**
 * TEAM-351: Tests for message receiver
 * 
 * Behavioral tests covering:
 * - Message receiving
 * - Origin validation
 * - Message validation
 * - Statistics tracking
 * - Memory leak prevention
 * - Error handling
 */

import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  createMessageReceiver,
  getReceiveStats,
  resetReceiveStats,
  getActiveReceiverCount,
  cleanupAllReceivers,
  type ReceiverConfig,
  type IframeMessage,
} from './receiver'

beforeEach(() => {
  resetReceiveStats()
  cleanupAllReceivers()
})

describe('@rbee/iframe-bridge - receiver', () => {
  describe('createMessageReceiver()', () => {
    it('should create receiver and return cleanup function', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      const cleanup = createMessageReceiver(config)
      expect(typeof cleanup).toBe('function')
    })

    it('should throw on invalid config', () => {
      const config = {
        allowedOrigins: ['http://localhost:3000'],
        // missing onMessage
      } as any
      
      expect(() => createMessageReceiver(config)).toThrow('onMessage must be a function')
    })

    it('should throw on invalid origin config', () => {
      const config = {
        allowedOrigins: [],
        onMessage: vi.fn(),
      } as ReceiverConfig
      
      expect(() => createMessageReceiver(config)).toThrow('Invalid origin config')
    })

    it('should receive valid message from allowed origin', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      createMessageReceiver(config)
      
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: message,
        origin: 'http://localhost:3000',
      }))
      
      expect(onMessage).toHaveBeenCalledTimes(1)
      expect(onMessage).toHaveBeenCalledWith(message)
    })

    it('should reject message from non-allowed origin', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      createMessageReceiver(config)
      
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: message,
        origin: 'http://localhost:4000',
      }))
      
      expect(onMessage).not.toHaveBeenCalled()
    })

    it('should reject invalid message structure', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      createMessageReceiver(config)
      
      const invalidMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        // missing command field
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: invalidMessage,
        origin: 'http://localhost:3000',
      }))
      
      expect(onMessage).not.toHaveBeenCalled()
    })

    it('should skip validation when validate=false', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
        validate: false,
      }
      
      createMessageReceiver(config)
      
      const invalidMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: invalidMessage,
        origin: 'http://localhost:3000',
      }))
      
      expect(onMessage).toHaveBeenCalled()
    })

    it('should call onError when onMessage throws', () => {
      const onMessage = vi.fn().mockImplementation(() => {
        throw new Error('Handler failed')
      })
      const onError = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
        onError,
      }
      
      createMessageReceiver(config)
      
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: message,
        origin: 'http://localhost:3000',
      }))
      
      expect(onError).toHaveBeenCalledTimes(1)
      expect(onError.mock.calls[0][0]).toBeInstanceOf(Error)
    })

    it('should cleanup event listener', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      const cleanup = createMessageReceiver(config)
      cleanup()
      
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: message,
        origin: 'http://localhost:3000',
      }))
      
      expect(onMessage).not.toHaveBeenCalled()
    })
  })

  describe('Statistics tracking', () => {
    it('should track accepted messages', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      createMessageReceiver(config)
      
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: message,
        origin: 'http://localhost:3000',
      }))
      
      const stats = getReceiveStats()
      expect(stats.total).toBe(1)
      expect(stats.accepted).toBe(1)
      expect(stats.rejected).toBe(0)
    })

    it('should track rejected origins', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      createMessageReceiver(config)
      
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: message,
        origin: 'http://localhost:4000',
      }))
      
      const stats = getReceiveStats()
      expect(stats.total).toBe(1)
      expect(stats.invalidOrigin).toBe(1)
    })

    it('should track invalid messages', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      createMessageReceiver(config)
      
      const invalidMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: invalidMessage,
        origin: 'http://localhost:3000',
      }))
      
      const stats = getReceiveStats()
      expect(stats.total).toBe(1)
      expect(stats.invalidMessage).toBe(1)
    })

    it('should track handler errors', () => {
      const onMessage = vi.fn().mockImplementation(() => {
        throw new Error('Failed')
      })
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      createMessageReceiver(config)
      
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: message,
        origin: 'http://localhost:3000',
      }))
      
      const stats = getReceiveStats()
      expect(stats.errors).toBe(1)
    })
  })

  describe('Memory leak prevention', () => {
    it('should track active receivers', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      expect(getActiveReceiverCount()).toBe(0)
      
      const cleanup1 = createMessageReceiver(config)
      expect(getActiveReceiverCount()).toBe(1)
      
      const cleanup2 = createMessageReceiver(config)
      expect(getActiveReceiverCount()).toBe(2)
      
      cleanup1()
      expect(getActiveReceiverCount()).toBe(1)
      
      cleanup2()
      expect(getActiveReceiverCount()).toBe(0)
    })

    it('should cleanup all receivers', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      createMessageReceiver(config)
      createMessageReceiver(config)
      createMessageReceiver(config)
      
      expect(getActiveReceiverCount()).toBe(3)
      
      cleanupAllReceivers()
      expect(getActiveReceiverCount()).toBe(0)
    })
  })

  describe('Edge cases', () => {
    it('should handle messages without data', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      createMessageReceiver(config)
      
      window.dispatchEvent(new MessageEvent('message', {
        data: null,
        origin: 'http://localhost:3000',
      }))
      
      expect(onMessage).not.toHaveBeenCalled()
    })

    it('should handle messages without type', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['http://localhost:3000'],
        onMessage,
      }
      
      createMessageReceiver(config)
      
      window.dispatchEvent(new MessageEvent('message', {
        data: { source: 'test' },
        origin: 'http://localhost:3000',
      }))
      
      expect(onMessage).not.toHaveBeenCalled()
    })

    it('should handle wildcard origin', () => {
      const onMessage = vi.fn()
      const config: ReceiverConfig = {
        allowedOrigins: ['*'],
        onMessage,
      }
      
      createMessageReceiver(config)
      
      const message: IframeMessage = {
        type: 'COMMAND',
        source: 'test',
        timestamp: Date.now(),
        command: 'test',
      }
      
      window.dispatchEvent(new MessageEvent('message', {
        data: message,
        origin: 'http://anything.com',
      }))
      
      expect(onMessage).toHaveBeenCalled()
    })
  })
})
