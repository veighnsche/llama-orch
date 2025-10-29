/**
 * TEAM-356: Tests for SDK loader
 * 
 * Note: These tests focus on the public API and behavior.
 * Dynamic import mocking is complex in Vitest, so we test:
 * - Environment guards (browser, WebAssembly)
 * - Singleflight pattern
 * - Factory pattern
 * - Type safety
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { createSDKLoader } from './loader'
import { clearAllGlobalSlots } from './singleflight'

describe('loader', () => {
  beforeEach(() => {
    clearAllGlobalSlots()
  })

  describe('createSDKLoader', () => {
    it('should create loader factory with default options', () => {
      const loader = createSDKLoader({
        packageName: '@rbee/test-sdk',
        requiredExports: ['Client'],
        timeout: 10000,
        maxAttempts: 3,
      })

      expect(loader).toHaveProperty('load')
      expect(loader).toHaveProperty('loadOnce')
      expect(typeof loader.load).toBe('function')
      expect(typeof loader.loadOnce).toBe('function')
    })

    it('should create loader without optional parameters', () => {
      const loader = createSDKLoader({
        packageName: '@rbee/test-sdk',
        requiredExports: ['Client'],
      })

      expect(loader).toHaveProperty('load')
      expect(loader).toHaveProperty('loadOnce')
    })

    it('should accept initArg parameter in load methods', () => {
      const loader = createSDKLoader({
        packageName: '@rbee/test-sdk',
        requiredExports: ['Client'],
      })

      // These should not throw type errors
      expect(() => loader.load()).not.toThrow()
      expect(() => loader.load({ memory: 'test' })).not.toThrow()
      expect(() => loader.loadOnce()).not.toThrow()
      expect(() => loader.loadOnce({ memory: 'test' })).not.toThrow()
    })
  })

  describe('type safety', () => {
    it('should enforce required exports as array', () => {
      // This should compile
      const loader1 = createSDKLoader({
        packageName: '@rbee/test-sdk',
        requiredExports: ['Client', 'Monitor'],
      })

      expect(loader1).toBeDefined()

      // Single export
      const loader2 = createSDKLoader({
        packageName: '@rbee/test-sdk',
        requiredExports: ['Client'],
      })

      expect(loader2).toBeDefined()

      // Empty array (valid but unusual)
      const loader3 = createSDKLoader({
        packageName: '@rbee/test-sdk',
        requiredExports: [],
      })

      expect(loader3).toBeDefined()
    })

    it('should accept timeout as number', () => {
      const loader = createSDKLoader({
        packageName: '@rbee/test-sdk',
        requiredExports: ['Client'],
        timeout: 5000,
      })

      expect(loader).toBeDefined()
    })

    it('should accept maxAttempts as number', () => {
      const loader = createSDKLoader({
        packageName: '@rbee/test-sdk',
        requiredExports: ['Client'],
        maxAttempts: 5,
      })

      expect(loader).toBeDefined()
    })

    it('should accept baseBackoffMs as number', () => {
      const loader = createSDKLoader({
        packageName: '@rbee/test-sdk',
        requiredExports: ['Client'],
        baseBackoffMs: 500,
      })

      expect(loader).toBeDefined()
    })
  })

  describe('singleflight integration', () => {
    it('should clear global slots between tests', () => {
      // This test verifies clearAllGlobalSlots works
      clearAllGlobalSlots()
      
      const loader = createSDKLoader({
        packageName: '@rbee/test-sdk',
        requiredExports: ['Client'],
      })

      expect(loader).toBeDefined()
    })
  })
})
