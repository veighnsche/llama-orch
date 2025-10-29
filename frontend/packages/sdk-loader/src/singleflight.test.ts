/**
 * TEAM-356: Tests for singleflight pattern
 */

import { describe, it, expect, beforeEach } from 'vitest'
import { getGlobalSlot, clearGlobalSlot, clearAllGlobalSlots } from './singleflight'
import type { SDKLoadResult } from './types'

describe('singleflight', () => {
  beforeEach(() => {
    clearAllGlobalSlots()
  })

  describe('getGlobalSlot', () => {
    it('should create new slot for package', () => {
      const slot = getGlobalSlot('@rbee/test-sdk')
      expect(slot).toBeDefined()
      expect(slot.value).toBeUndefined()
      expect(slot.error).toBeUndefined()
      expect(slot.promise).toBeUndefined()
    })

    it('should return same slot for same package', () => {
      const slot1 = getGlobalSlot('@rbee/test-sdk')
      const slot2 = getGlobalSlot('@rbee/test-sdk')
      expect(slot1).toBe(slot2)
    })

    it('should create different slots for different packages', () => {
      const slot1 = getGlobalSlot('@rbee/sdk-1')
      const slot2 = getGlobalSlot('@rbee/sdk-2')
      expect(slot1).not.toBe(slot2)
    })

    it('should preserve slot state across calls', () => {
      const slot = getGlobalSlot<any>('@rbee/test-sdk')
      const mockResult: SDKLoadResult<any> = {
        sdk: { test: true },
        loadTime: 100,
        attempts: 1,
      }
      slot.value = mockResult

      const slot2 = getGlobalSlot<any>('@rbee/test-sdk')
      expect(slot2.value).toBe(mockResult)
    })
  })

  describe('clearGlobalSlot', () => {
    it('should clear specific slot', () => {
      const slot = getGlobalSlot('@rbee/test-sdk')
      slot.value = { sdk: {}, loadTime: 100, attempts: 1 }

      clearGlobalSlot('@rbee/test-sdk')

      const newSlot = getGlobalSlot('@rbee/test-sdk')
      expect(newSlot.value).toBeUndefined()
    })

    it('should not affect other slots', () => {
      const slot1 = getGlobalSlot<any>('@rbee/sdk-1')
      const slot2 = getGlobalSlot<any>('@rbee/sdk-2')
      
      slot1.value = { sdk: { id: 1 }, loadTime: 100, attempts: 1 }
      slot2.value = { sdk: { id: 2 }, loadTime: 200, attempts: 2 }

      clearGlobalSlot('@rbee/sdk-1')

      expect(getGlobalSlot('@rbee/sdk-1').value).toBeUndefined()
      expect(getGlobalSlot<any>('@rbee/sdk-2').value?.sdk.id).toBe(2)
    })

    it('should handle clearing non-existent slot', () => {
      expect(() => clearGlobalSlot('@rbee/non-existent')).not.toThrow()
    })
  })

  describe('clearAllGlobalSlots', () => {
    it('should clear all slots', () => {
      const slot1 = getGlobalSlot<any>('@rbee/sdk-1')
      const slot2 = getGlobalSlot<any>('@rbee/sdk-2')
      
      slot1.value = { sdk: {}, loadTime: 100, attempts: 1 }
      slot2.value = { sdk: {}, loadTime: 200, attempts: 2 }

      clearAllGlobalSlots()

      expect(getGlobalSlot('@rbee/sdk-1').value).toBeUndefined()
      expect(getGlobalSlot('@rbee/sdk-2').value).toBeUndefined()
    })

    it('should handle clearing when no slots exist', () => {
      expect(() => clearAllGlobalSlots()).not.toThrow()
    })
  })

  describe('slot state management', () => {
    it('should store successful load result', () => {
      const slot = getGlobalSlot<{ test: boolean }>('@rbee/test-sdk')
      const result: SDKLoadResult<{ test: boolean }> = {
        sdk: { test: true },
        loadTime: 150,
        attempts: 2,
      }
      
      slot.value = result
      
      expect(slot.value.sdk.test).toBe(true)
      expect(slot.value.loadTime).toBe(150)
      expect(slot.value.attempts).toBe(2)
    })

    it('should store error from failed load', () => {
      const slot = getGlobalSlot('@rbee/test-sdk')
      const error = new Error('Load failed')
      
      slot.error = error
      
      expect(slot.error).toBe(error)
      expect(slot.error.message).toBe('Load failed')
    })

    it('should store in-progress promise', () => {
      const slot = getGlobalSlot('@rbee/test-sdk')
      const promise = Promise.resolve({ sdk: {}, loadTime: 100, attempts: 1 })
      
      slot.promise = promise
      
      expect(slot.promise).toBe(promise)
    })
  })
})
