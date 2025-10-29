/**
 * TEAM-351: Tests for narration config
 * 
 * Behavioral tests covering:
 * - Service configuration
 * - Port configuration (imported from shared-config)
 * - Service name validation
 */

import { describe, it, expect } from 'vitest'
import { SERVICES } from './config'
import { PORTS } from '@rbee/shared-config'

describe('@rbee/narration-client - config', () => {
  describe('SERVICES constant', () => {
    it('should have all service configurations', () => {
      expect(SERVICES.queen).toBeDefined()
      expect(SERVICES.hive).toBeDefined()
      expect(SERVICES.worker).toBeDefined()
    })

    it('should have correct service names', () => {
      expect(SERVICES.queen.name).toBe('queen-rbee')
      expect(SERVICES.hive.name).toBe('rbee-hive')
      expect(SERVICES.worker.name).toBe('llm-worker')
    })
  })

  describe('Port configuration integration', () => {
    it('should use ports from @rbee/shared-config for queen', () => {
      expect(SERVICES.queen.devPort).toBe(PORTS.queen.dev)
      expect(SERVICES.queen.prodPort).toBe(PORTS.queen.prod)
    })

    it('should use ports from @rbee/shared-config for hive', () => {
      expect(SERVICES.hive.devPort).toBe(PORTS.hive.dev)
      expect(SERVICES.hive.prodPort).toBe(PORTS.hive.prod)
    })

    it('should use ports from @rbee/shared-config for worker', () => {
      expect(SERVICES.worker.devPort).toBe(PORTS.worker.dev)
      expect(SERVICES.worker.prodPort).toBe(PORTS.worker.prod)
    })

    it('should use keeper dev port for all services', () => {
      expect(SERVICES.queen.keeperDevPort).toBe(PORTS.keeper.dev)
      expect(SERVICES.hive.keeperDevPort).toBe(PORTS.keeper.dev)
      expect(SERVICES.worker.keeperDevPort).toBe(PORTS.keeper.dev)
    })
  })

  describe('Keeper origin configuration', () => {
    it('should use wildcard for keeper prod origin', () => {
      expect(SERVICES.queen.keeperProdOrigin).toBe('*')
      expect(SERVICES.hive.keeperProdOrigin).toBe('*')
      expect(SERVICES.worker.keeperProdOrigin).toBe('*')
    })
  })

  describe('Service configuration structure', () => {
    it('should have all required fields for queen', () => {
      expect(SERVICES.queen).toHaveProperty('name')
      expect(SERVICES.queen).toHaveProperty('devPort')
      expect(SERVICES.queen).toHaveProperty('prodPort')
      expect(SERVICES.queen).toHaveProperty('keeperDevPort')
      expect(SERVICES.queen).toHaveProperty('keeperProdOrigin')
    })

    it('should have all required fields for hive', () => {
      expect(SERVICES.hive).toHaveProperty('name')
      expect(SERVICES.hive).toHaveProperty('devPort')
      expect(SERVICES.hive).toHaveProperty('prodPort')
      expect(SERVICES.hive).toHaveProperty('keeperDevPort')
      expect(SERVICES.hive).toHaveProperty('keeperProdOrigin')
    })

    it('should have all required fields for worker', () => {
      expect(SERVICES.worker).toHaveProperty('name')
      expect(SERVICES.worker).toHaveProperty('devPort')
      expect(SERVICES.worker).toHaveProperty('prodPort')
      expect(SERVICES.worker).toHaveProperty('keeperDevPort')
      expect(SERVICES.worker).toHaveProperty('keeperProdOrigin')
    })
  })

  describe('Port consistency', () => {
    it('should have valid port numbers', () => {
      expect(SERVICES.queen.devPort).toBeGreaterThan(0)
      expect(SERVICES.queen.devPort).toBeLessThanOrEqual(65535)
      expect(SERVICES.queen.prodPort).toBeGreaterThan(0)
      expect(SERVICES.queen.prodPort).toBeLessThanOrEqual(65535)
    })

    it('should have different dev and prod ports for queen', () => {
      expect(SERVICES.queen.devPort).not.toBe(SERVICES.queen.prodPort)
    })

    it('should have unique dev ports across services', () => {
      const devPorts = [
        SERVICES.queen.devPort,
        SERVICES.hive.devPort,
        SERVICES.worker.devPort,
      ]
      const uniquePorts = new Set(devPorts)
      
      expect(uniquePorts.size).toBe(devPorts.length)
    })
  })
})
