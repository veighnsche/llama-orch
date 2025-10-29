/**
 * TEAM-351: Origin validation for iframe communication
 * TEAM-351: Bug fixes - URL validation, security, edge cases
 */

export interface OriginConfig {
  allowedOrigins: string[]
  strictMode?: boolean
  allowLocalhost?: boolean  // Allow any localhost port
}

/**
 * Validate origin URL format
 * TEAM-351: Ensure origin is a valid URL
 */
export function isValidOriginFormat(origin: string): boolean {
  if (!origin || typeof origin !== 'string') {
    return false
  }
  
  // Wildcard is valid
  if (origin === '*') {
    return true
  }
  
  // Must start with protocol
  if (!origin.startsWith('http://') && !origin.startsWith('https://')) {
    return false
  }
  
  // Try to parse as URL
  try {
    const url = new URL(origin)
    // Origin should not have path, query, or hash
    return url.pathname === '/' && !url.search && !url.hash
  } catch {
    return false
  }
}

/**
 * Check if origin is localhost
 * TEAM-351: Localhost detection for development
 */
export function isLocalhostOrigin(origin: string): boolean {
  if (origin === '*') return false
  
  try {
    const url = new URL(origin)
    return (
      url.hostname === 'localhost' ||
      url.hostname === '127.0.0.1' ||
      url.hostname === '[::1]'
    )
  } catch {
    return false
  }
}

/**
 * Validate origin against config
 * 
 * TEAM-351: Bug fixes - Format validation, localhost support, security
 * 
 * @param origin - Origin to validate
 * @param config - Origin configuration
 * @returns true if origin is allowed
 */
export function validateOrigin(
  origin: string,
  config: OriginConfig
): boolean {
  // TEAM-351: Validate origin format first
  if (!isValidOriginFormat(origin)) {
    return false
  }
  
  // TEAM-351: Validate config
  if (!config.allowedOrigins || config.allowedOrigins.length === 0) {
    return false
  }
  
  // TEAM-351: Check for wildcard (only in non-strict mode)
  if (!config.strictMode && config.allowedOrigins.includes('*')) {
    return true
  }
  
  // TEAM-351: Allow any localhost in development (if enabled)
  if (config.allowLocalhost && isLocalhostOrigin(origin)) {
    // Still check if at least one localhost origin is in allowed list
    const hasLocalhostAllowed = config.allowedOrigins.some(isLocalhostOrigin)
    if (hasLocalhostAllowed) {
      return true
    }
  }
  
  // TEAM-351: Exact match required
  return config.allowedOrigins.includes(origin)
}

/**
 * Validate origin config
 * TEAM-351: Ensure config is valid
 */
export function isValidOriginConfig(config: any): config is OriginConfig {
  return (
    config !== null &&
    typeof config === 'object' &&
    Array.isArray(config.allowedOrigins) &&
    config.allowedOrigins.length > 0 &&
    config.allowedOrigins.every((origin: any) => 
      typeof origin === 'string' && isValidOriginFormat(origin)
    )
  )
}

/**
 * Create origin validator with config validation
 * 
 * TEAM-351: Bug fixes - Config validation at creation time
 * 
 * @param config - Origin configuration
 * @returns Validator function
 * @throws Error if config is invalid
 */
export function createOriginValidator(config: OriginConfig) {
  // TEAM-351: Validate config at creation time
  if (!isValidOriginConfig(config)) {
    throw new Error('Invalid origin config: allowedOrigins must be non-empty array of valid origin URLs')
  }
  
  return (origin: string) => validateOrigin(origin, config)
}
