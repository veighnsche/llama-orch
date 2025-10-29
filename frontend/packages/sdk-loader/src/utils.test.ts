/**
 * TEAM-356: Tests for utility functions
 */

import { describe, it, expect, vi } from 'vitest'
import { sleep, addJitter, withTimeout, calculateBackoff } from './utils'

describe('utils', () => {
  describe('sleep', () => {
    it('should resolve after specified delay', async () => {
      const start = Date.now()
      await sleep(100)
      const elapsed = Date.now() - start
      expect(elapsed).toBeGreaterThanOrEqual(90) // Allow some variance
      expect(elapsed).toBeLessThan(150)
    })

    it('should resolve immediately for 0ms', async () => {
      const start = Date.now()
      await sleep(0)
      const elapsed = Date.now() - start
      expect(elapsed).toBeLessThan(50)
    })
  })

  describe('addJitter', () => {
    it('should add random jitter within range', () => {
      const baseMs = 100
      const maxJitterMs = 50
      
      for (let i = 0; i < 100; i++) {
        const result = addJitter(baseMs, maxJitterMs)
        expect(result).toBeGreaterThanOrEqual(baseMs)
        expect(result).toBeLessThanOrEqual(baseMs + maxJitterMs)
      }
    })

    it('should return base value when jitter is 0', () => {
      const result = addJitter(100, 0)
      expect(result).toBe(100)
    })

    it('should handle negative base values', () => {
      const result = addJitter(-100, 50)
      expect(result).toBeGreaterThanOrEqual(-100)
      expect(result).toBeLessThanOrEqual(-50)
    })
  })

  describe('withTimeout', () => {
    it('should resolve if promise completes before timeout', async () => {
      const promise = Promise.resolve('success')
      const result = await withTimeout(promise, 1000, 'test operation')
      expect(result).toBe('success')
    })

    it('should reject if timeout exceeded', async () => {
      const promise = new Promise(resolve => setTimeout(resolve, 200))
      await expect(
        withTimeout(promise, 50, 'slow operation')
      ).rejects.toThrow('Timeout after 50ms: slow operation')
    })

    it('should propagate promise rejection', async () => {
      const promise = Promise.reject(new Error('original error'))
      await expect(
        withTimeout(promise, 1000, 'test operation')
      ).rejects.toThrow('original error')
    })

    it('should include operation name in timeout error', async () => {
      const promise = new Promise(resolve => setTimeout(resolve, 200))
      await expect(
        withTimeout(promise, 50, 'custom operation name')
      ).rejects.toThrow('custom operation name')
    })
  })

  describe('calculateBackoff', () => {
    it('should calculate exponential backoff for attempt 1', () => {
      const result = calculateBackoff(1, 100, 50)
      // 2^0 * 100 = 100, plus jitter (0-50)
      expect(result).toBeGreaterThanOrEqual(100)
      expect(result).toBeLessThanOrEqual(150)
    })

    it('should calculate exponential backoff for attempt 2', () => {
      const result = calculateBackoff(2, 100, 50)
      // 2^1 * 100 = 200, plus jitter (0-50)
      expect(result).toBeGreaterThanOrEqual(200)
      expect(result).toBeLessThanOrEqual(250)
    })

    it('should calculate exponential backoff for attempt 3', () => {
      const result = calculateBackoff(3, 100, 50)
      // 2^2 * 100 = 400, plus jitter (0-50)
      expect(result).toBeGreaterThanOrEqual(400)
      expect(result).toBeLessThanOrEqual(450)
    })

    it('should handle large attempt numbers', () => {
      const result = calculateBackoff(10, 100, 50)
      // 2^9 * 100 = 51200, plus jitter (0-50)
      expect(result).toBeGreaterThanOrEqual(51200)
      expect(result).toBeLessThanOrEqual(51250)
    })

    it('should handle zero jitter', () => {
      const result = calculateBackoff(2, 100, 0)
      expect(result).toBe(200) // 2^1 * 100, no jitter
    })
  })
})
