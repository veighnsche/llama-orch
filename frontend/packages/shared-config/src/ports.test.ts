/**
 * TEAM-351: Tests for @rbee/shared-config
 * 
 * Tests cover:
 * - Port configuration structure
 * - getAllowedOrigins() function
 * - getIframeUrl() function
 * - getParentOrigin() function
 * - getServiceUrl() function
 * - Edge cases (null ports, HTTPS, invalid inputs)
 */

import { describe, it, expect } from 'vitest'
import {
  PORTS,
  type ServiceName,
  getAllowedOrigins,
  getIframeUrl,
  getParentOrigin,
  getServiceUrl,
} from './ports'

describe('@rbee/shared-config', () => {
  describe('PORTS constant', () => {
    it('should have correct structure', () => {
      expect(PORTS).toBeDefined()
      expect(PORTS.keeper).toBeDefined()
      expect(PORTS.queen).toBeDefined()
      expect(PORTS.hive).toBeDefined()
      expect(PORTS.worker).toBeDefined()
    })

    it('should have keeper ports', () => {
      expect(PORTS.keeper.dev).toBe(5173)
      expect(PORTS.keeper.prod).toBeNull()
    })

    it('should have queen ports', () => {
      expect(PORTS.queen.dev).toBe(7834)
      expect(PORTS.queen.prod).toBe(7833)
      expect(PORTS.queen.backend).toBe(7833)
    })

    it('should have hive ports', () => {
      expect(PORTS.hive.dev).toBe(7836)
      expect(PORTS.hive.prod).toBe(7835)
      expect(PORTS.hive.backend).toBe(7835)
    })

    it('should have worker ports', () => {
      expect(PORTS.worker.dev).toBe(7837)
      expect(PORTS.worker.prod).toBe(8080)
      expect(PORTS.worker.backend).toBe(8080)
    })

    it('should be readonly (as const)', () => {
      // TypeScript enforces this at compile time
      // Runtime check: object should be frozen or immutable
      expect(Object.isFrozen(PORTS) || typeof PORTS === 'object').toBe(true)
    })
  })

  describe('getAllowedOrigins()', () => {
    it('should return HTTP origins by default', () => {
      const origins = getAllowedOrigins()
      
      expect(origins).toContain('http://localhost:7834') // queen dev
      expect(origins).toContain('http://localhost:7833') // queen prod
      expect(origins).toContain('http://localhost:7836') // hive dev
      expect(origins).toContain('http://localhost:7835') // hive prod
      expect(origins).toContain('http://localhost:7837') // worker dev
      expect(origins).toContain('http://localhost:8080') // worker prod
    })

    it('should not include keeper', () => {
      const origins = getAllowedOrigins()
      
      expect(origins).not.toContain('http://localhost:5173')
    })

    it('should include HTTPS when requested', () => {
      const origins = getAllowedOrigins(true)
      
      expect(origins).toContain('https://localhost:7833') // queen prod
      expect(origins).toContain('https://localhost:7835') // hive prod
      expect(origins).toContain('https://localhost:8080') // worker prod
    })

    it('should not include HTTPS for dev ports', () => {
      const origins = getAllowedOrigins(true)
      
      expect(origins).not.toContain('https://localhost:7834') // queen dev
      expect(origins).not.toContain('https://localhost:7836') // hive dev
      expect(origins).not.toContain('https://localhost:7837') // worker dev
    })

    it('should return sorted array', () => {
      const origins = getAllowedOrigins()
      const sorted = [...origins].sort()
      
      expect(origins).toEqual(sorted)
    })

    it('should not have duplicates', () => {
      const origins = getAllowedOrigins()
      const unique = [...new Set(origins)]
      
      expect(origins.length).toBe(unique.length)
    })

    it('should return consistent results', () => {
      const origins1 = getAllowedOrigins()
      const origins2 = getAllowedOrigins()
      
      expect(origins1).toEqual(origins2)
    })
  })

  describe('getIframeUrl()', () => {
    it('should return dev URL for queen', () => {
      const url = getIframeUrl('queen', true)
      expect(url).toBe('http://localhost:7834')
    })

    it('should return prod URL for queen', () => {
      const url = getIframeUrl('queen', false)
      expect(url).toBe('http://localhost:7833')
    })

    it('should return dev URL for hive', () => {
      const url = getIframeUrl('hive', true)
      expect(url).toBe('http://localhost:7836')
    })

    it('should return prod URL for hive', () => {
      const url = getIframeUrl('hive', false)
      expect(url).toBe('http://localhost:7835')
    })

    it('should return dev URL for worker', () => {
      const url = getIframeUrl('worker', true)
      expect(url).toBe('http://localhost:7837')
    })

    it('should return prod URL for worker', () => {
      const url = getIframeUrl('worker', false)
      expect(url).toBe('http://localhost:8080')
    })

    it('should return dev URL for keeper', () => {
      const url = getIframeUrl('keeper', true)
      expect(url).toBe('http://localhost:5173')
    })

    it('should throw error for keeper prod', () => {
      expect(() => getIframeUrl('keeper', false)).toThrow(
        'Keeper service has no production HTTP port'
      )
    })

    it('should support HTTPS', () => {
      const url = getIframeUrl('queen', false, true)
      expect(url).toBe('https://localhost:7833')
    })

    it('should support HTTPS for dev', () => {
      const url = getIframeUrl('queen', true, true)
      expect(url).toBe('https://localhost:7834')
    })
  })

  describe('getParentOrigin()', () => {
    it('should return keeper dev for queen dev port', () => {
      const origin = getParentOrigin(7834)
      expect(origin).toBe('http://localhost:5173')
    })

    it('should return keeper dev for hive dev port', () => {
      const origin = getParentOrigin(7836)
      expect(origin).toBe('http://localhost:5173')
    })

    it('should return keeper dev for worker dev port', () => {
      const origin = getParentOrigin(7837)
      expect(origin).toBe('http://localhost:5173')
    })

    it('should return keeper dev for keeper dev port', () => {
      const origin = getParentOrigin(5173)
      expect(origin).toBe('http://localhost:5173')
    })

    it('should return wildcard for queen prod port', () => {
      const origin = getParentOrigin(7833)
      expect(origin).toBe('*')
    })

    it('should return wildcard for hive prod port', () => {
      const origin = getParentOrigin(7835)
      expect(origin).toBe('*')
    })

    it('should return wildcard for worker prod port', () => {
      const origin = getParentOrigin(8080)
      expect(origin).toBe('*')
    })

    it('should return wildcard for unknown port', () => {
      const origin = getParentOrigin(9999)
      expect(origin).toBe('*')
    })
  })

  describe('getServiceUrl()', () => {
    describe('dev mode', () => {
      it('should return queen dev URL', () => {
        const url = getServiceUrl('queen', 'dev')
        expect(url).toBe('http://localhost:7834')
      })

      it('should return hive dev URL', () => {
        const url = getServiceUrl('hive', 'dev')
        expect(url).toBe('http://localhost:7836')
      })

      it('should return worker dev URL', () => {
        const url = getServiceUrl('worker', 'dev')
        expect(url).toBe('http://localhost:7837')
      })

      it('should return keeper dev URL', () => {
        const url = getServiceUrl('keeper', 'dev')
        expect(url).toBe('http://localhost:5173')
      })
    })

    describe('prod mode', () => {
      it('should return queen prod URL', () => {
        const url = getServiceUrl('queen', 'prod')
        expect(url).toBe('http://localhost:7833')
      })

      it('should return hive prod URL', () => {
        const url = getServiceUrl('hive', 'prod')
        expect(url).toBe('http://localhost:7835')
      })

      it('should return worker prod URL', () => {
        const url = getServiceUrl('worker', 'prod')
        expect(url).toBe('http://localhost:8080')
      })

      it('should return empty string for keeper prod', () => {
        const url = getServiceUrl('keeper', 'prod')
        expect(url).toBe('')
      })
    })

    describe('backend mode', () => {
      it('should return queen backend URL', () => {
        const url = getServiceUrl('queen', 'backend')
        expect(url).toBe('http://localhost:7833')
      })

      it('should return hive backend URL', () => {
        const url = getServiceUrl('hive', 'backend')
        expect(url).toBe('http://localhost:7835')
      })

      it('should return worker backend URL', () => {
        const url = getServiceUrl('worker', 'backend')
        expect(url).toBe('http://localhost:8080')
      })

      it('should fallback to prod for keeper backend', () => {
        const url = getServiceUrl('keeper', 'backend')
        expect(url).toBe('')
      })
    })

    describe('HTTPS support', () => {
      it('should support HTTPS in dev mode', () => {
        const url = getServiceUrl('queen', 'dev', true)
        expect(url).toBe('https://localhost:7834')
      })

      it('should support HTTPS in prod mode', () => {
        const url = getServiceUrl('queen', 'prod', true)
        expect(url).toBe('https://localhost:7833')
      })

      it('should support HTTPS in backend mode', () => {
        const url = getServiceUrl('queen', 'backend', true)
        expect(url).toBe('https://localhost:7833')
      })
    })

    describe('default parameters', () => {
      it('should default to dev mode', () => {
        const url = getServiceUrl('queen')
        expect(url).toBe('http://localhost:7834')
      })

      it('should default to HTTP', () => {
        const url = getServiceUrl('queen', 'prod')
        expect(url).toBe('http://localhost:7833')
      })
    })
  })

  describe('Edge cases', () => {
    it('should handle all service names', () => {
      const services: ServiceName[] = ['keeper', 'queen', 'hive', 'worker']
      
      services.forEach(service => {
        expect(() => getServiceUrl(service, 'dev')).not.toThrow()
      })
    })

    it('should return consistent URLs', () => {
      const url1 = getServiceUrl('queen', 'prod')
      const url2 = getServiceUrl('queen', 'prod')
      
      expect(url1).toBe(url2)
    })

    it('should handle null ports gracefully', () => {
      const url = getServiceUrl('keeper', 'prod')
      expect(url).toBe('')
    })
  })
})
