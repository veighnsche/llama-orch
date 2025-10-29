/**
 * TEAM-351: Tests for origin validation
 * 
 * Behavioral tests covering:
 * - Origin format validation
 * - Localhost detection
 * - Origin validation against config
 * - Config validation
 * - Validator creation
 */

import { describe, it, expect } from 'vitest'
import {
  isValidOriginFormat,
  isLocalhostOrigin,
  validateOrigin,
  isValidOriginConfig,
  createOriginValidator,
  type OriginConfig,
} from './validator'

describe('@rbee/iframe-bridge - validator', () => {
  describe('isValidOriginFormat()', () => {
    it('should accept wildcard', () => {
      expect(isValidOriginFormat('*')).toBe(true)
    })

    it('should accept valid HTTP origin', () => {
      expect(isValidOriginFormat('http://localhost:3000')).toBe(true)
    })

    it('should accept valid HTTPS origin', () => {
      expect(isValidOriginFormat('https://example.com')).toBe(true)
    })

    it('should reject origin without protocol', () => {
      expect(isValidOriginFormat('localhost:3000')).toBe(false)
    })

    it('should reject origin with path', () => {
      expect(isValidOriginFormat('http://localhost:3000/path')).toBe(false)
    })

    it('should reject origin with query', () => {
      expect(isValidOriginFormat('http://localhost:3000?query=1')).toBe(false)
    })

    it('should reject origin with hash', () => {
      expect(isValidOriginFormat('http://localhost:3000#hash')).toBe(false)
    })

    it('should reject empty string', () => {
      expect(isValidOriginFormat('')).toBe(false)
    })

    it('should reject null', () => {
      expect(isValidOriginFormat(null as any)).toBe(false)
    })

    it('should reject non-string', () => {
      expect(isValidOriginFormat(123 as any)).toBe(false)
    })
  })

  describe('isLocalhostOrigin()', () => {
    it('should detect localhost hostname', () => {
      expect(isLocalhostOrigin('http://localhost:3000')).toBe(true)
    })

    it('should detect 127.0.0.1', () => {
      expect(isLocalhostOrigin('http://127.0.0.1:3000')).toBe(true)
    })

    it('should detect IPv6 localhost', () => {
      expect(isLocalhostOrigin('http://[::1]:3000')).toBe(true)
    })

    it('should reject wildcard', () => {
      expect(isLocalhostOrigin('*')).toBe(false)
    })

    it('should reject non-localhost', () => {
      expect(isLocalhostOrigin('http://example.com')).toBe(false)
    })

    it('should reject invalid URL', () => {
      expect(isLocalhostOrigin('not-a-url')).toBe(false)
    })
  })

  describe('validateOrigin()', () => {
    const config: OriginConfig = {
      allowedOrigins: ['http://localhost:3000', 'https://example.com'],
    }

    it('should accept allowed origin', () => {
      expect(validateOrigin('http://localhost:3000', config)).toBe(true)
    })

    it('should reject non-allowed origin', () => {
      expect(validateOrigin('http://localhost:4000', config)).toBe(false)
    })

    it('should reject invalid format', () => {
      expect(validateOrigin('not-a-url', config)).toBe(false)
    })

    it('should accept wildcard in non-strict mode', () => {
      const wildcardConfig: OriginConfig = {
        allowedOrigins: ['*'],
      }
      
      expect(validateOrigin('http://anything.com', wildcardConfig)).toBe(true)
    })

    it('should reject wildcard in strict mode', () => {
      const strictConfig: OriginConfig = {
        allowedOrigins: ['*'],
        strictMode: true,
      }
      
      expect(validateOrigin('http://anything.com', strictConfig)).toBe(false)
    })

    it('should allow any localhost when allowLocalhost=true', () => {
      const localhostConfig: OriginConfig = {
        allowedOrigins: ['http://localhost:3000'],
        allowLocalhost: true,
      }
      
      expect(validateOrigin('http://localhost:4000', localhostConfig)).toBe(true)
      expect(validateOrigin('http://127.0.0.1:5000', localhostConfig)).toBe(true)
    })

    it('should not allow localhost without localhost in allowed list', () => {
      const noLocalhostConfig: OriginConfig = {
        allowedOrigins: ['https://example.com'],
        allowLocalhost: true,
      }
      
      expect(validateOrigin('http://localhost:3000', noLocalhostConfig)).toBe(false)
    })

    it('should reject empty allowedOrigins', () => {
      const emptyConfig: OriginConfig = {
        allowedOrigins: [],
      }
      
      expect(validateOrigin('http://localhost:3000', emptyConfig)).toBe(false)
    })
  })

  describe('isValidOriginConfig()', () => {
    it('should validate valid config', () => {
      const config: OriginConfig = {
        allowedOrigins: ['http://localhost:3000'],
      }
      
      expect(isValidOriginConfig(config)).toBe(true)
    })

    it('should accept config with options', () => {
      const config: OriginConfig = {
        allowedOrigins: ['http://localhost:3000'],
        strictMode: true,
        allowLocalhost: true,
      }
      
      expect(isValidOriginConfig(config)).toBe(true)
    })

    it('should reject null', () => {
      expect(isValidOriginConfig(null)).toBe(false)
    })

    it('should reject non-object', () => {
      expect(isValidOriginConfig('string')).toBe(false)
    })

    it('should reject missing allowedOrigins', () => {
      const config = {}
      expect(isValidOriginConfig(config)).toBe(false)
    })

    it('should reject empty allowedOrigins', () => {
      const config = {
        allowedOrigins: [],
      }
      
      expect(isValidOriginConfig(config)).toBe(false)
    })

    it('should reject non-array allowedOrigins', () => {
      const config = {
        allowedOrigins: 'not-an-array',
      }
      
      expect(isValidOriginConfig(config)).toBe(false)
    })

    it('should reject invalid origin in array', () => {
      const config = {
        allowedOrigins: ['http://localhost:3000', 'not-a-url'],
      }
      
      expect(isValidOriginConfig(config)).toBe(false)
    })
  })

  describe('createOriginValidator()', () => {
    it('should create validator function', () => {
      const config: OriginConfig = {
        allowedOrigins: ['http://localhost:3000'],
      }
      
      const validator = createOriginValidator(config)
      expect(typeof validator).toBe('function')
    })

    it('should create working validator', () => {
      const config: OriginConfig = {
        allowedOrigins: ['http://localhost:3000'],
      }
      
      const validator = createOriginValidator(config)
      expect(validator('http://localhost:3000')).toBe(true)
      expect(validator('http://localhost:4000')).toBe(false)
    })

    it('should throw on invalid config', () => {
      const invalidConfig = {
        allowedOrigins: [],
      } as OriginConfig
      
      expect(() => createOriginValidator(invalidConfig)).toThrow('Invalid origin config')
    })

    it('should throw on null config', () => {
      expect(() => createOriginValidator(null as any)).toThrow()
    })
  })

  describe('Edge cases', () => {
    it('should handle multiple allowed origins', () => {
      const config: OriginConfig = {
        allowedOrigins: [
          'http://localhost:3000',
          'http://localhost:4000',
          'https://example.com',
        ],
      }
      
      expect(validateOrigin('http://localhost:3000', config)).toBe(true)
      expect(validateOrigin('http://localhost:4000', config)).toBe(true)
      expect(validateOrigin('https://example.com', config)).toBe(true)
      expect(validateOrigin('http://localhost:5000', config)).toBe(false)
    })

    it('should be case-sensitive', () => {
      const config: OriginConfig = {
        allowedOrigins: ['http://localhost:3000'],
      }
      
      expect(validateOrigin('HTTP://LOCALHOST:3000', config)).toBe(false)
    })

    it('should handle ports correctly', () => {
      const config: OriginConfig = {
        allowedOrigins: ['http://localhost:3000'],
      }
      
      expect(validateOrigin('http://localhost:3000', config)).toBe(true)
      expect(validateOrigin('http://localhost:3001', config)).toBe(false)
    })
  })
})
