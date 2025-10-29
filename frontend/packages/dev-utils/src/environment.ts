/**
 * TEAM-351: Environment detection utilities
 * TEAM-351: Bug fixes - Validation, SSR support, HTTPS detection, types
 */

/**
 * Environment information
 * TEAM-351: Type-safe environment data
 */
export interface EnvironmentInfo {
  isDev: boolean
  isProd: boolean
  isSSR: boolean
  port: number
  protocol: 'http' | 'https' | 'unknown'
  hostname: string
  url: string
}

/**
 * Port validation result
 * TEAM-351: For port validation feedback
 */
export interface PortValidation {
  valid: boolean
  port: number
  error?: string
}

/**
 * Check if running in development mode
 * TEAM-351: Browser-safe environment detection
 */
export function isDevelopment(): boolean {
  return (import.meta as any).env?.DEV ?? false
}

/**
 * Check if running in production mode
 * TEAM-351: Browser-safe environment detection
 */
export function isProduction(): boolean {
  return (import.meta as any).env?.PROD ?? false
}

/**
 * Check if running in SSR (server-side rendering)
 * TEAM-351: SSR detection
 */
export function isSSR(): boolean {
  return typeof window === 'undefined'
}

/**
 * Get current port with validation
 * 
 * TEAM-351: Bug fixes - Handle NaN, HTTPS default, validation
 * 
 * @returns Port number (80 for HTTP, 443 for HTTPS if not specified)
 */
export function getCurrentPort(): number {
  // TEAM-351: SSR-safe
  if (isSSR()) {
    return 0
  }
  
  const portStr = window.location.port
  
  // TEAM-351: If port is specified, parse it
  if (portStr) {
    const port = parseInt(portStr, 10)
    // TEAM-351: Validate parsed port
    if (isNaN(port) || port < 1 || port > 65535) {
      console.warn(`[dev-utils] Invalid port: ${portStr}`)
      return 0
    }
    return port
  }
  
  // TEAM-351: Default ports based on protocol
  return window.location.protocol === 'https:' ? 443 : 80
}

/**
 * Get current protocol
 * TEAM-351: Protocol detection
 */
export function getProtocol(): 'http' | 'https' | 'unknown' {
  if (isSSR()) {
    return 'unknown'
  }
  
  const protocol = window.location.protocol
  if (protocol === 'http:') return 'http'
  if (protocol === 'https:') return 'https'
  return 'unknown'
}

/**
 * Get current hostname
 * TEAM-351: Hostname detection
 */
export function getHostname(): string {
  if (isSSR()) {
    return ''
  }
  return window.location.hostname
}

/**
 * Validate port number
 * 
 * TEAM-351: Port validation with feedback
 * 
 * @param port - Port number to validate
 * @returns Validation result
 */
export function validatePort(port: number): PortValidation {
  if (typeof port !== 'number') {
    return {
      valid: false,
      port: 0,
      error: 'Port must be a number',
    }
  }
  
  if (isNaN(port)) {
    return {
      valid: false,
      port: 0,
      error: 'Port is NaN',
    }
  }
  
  if (port < 1 || port > 65535) {
    return {
      valid: false,
      port,
      error: `Port must be between 1 and 65535 (got ${port})`,
    }
  }
  
  return {
    valid: true,
    port,
  }
}

/**
 * Check if running on specific port
 * 
 * TEAM-351: Bug fixes - Validation, SSR support
 * 
 * @param port - Port number to check
 * @returns true if running on specified port
 */
export function isRunningOnPort(port: number): boolean {
  // TEAM-351: Validate port
  const validation = validatePort(port)
  if (!validation.valid) {
    console.warn(`[dev-utils] ${validation.error}`)
    return false
  }
  
  return getCurrentPort() === port
}

/**
 * Check if running on localhost
 * TEAM-351: Localhost detection
 */
export function isLocalhost(): boolean {
  if (isSSR()) {
    return false
  }
  
  const hostname = window.location.hostname
  return (
    hostname === 'localhost' ||
    hostname === '127.0.0.1' ||
    hostname === '[::1]'
  )
}

/**
 * Check if running with HTTPS
 * TEAM-351: HTTPS detection
 */
export function isHTTPS(): boolean {
  return getProtocol() === 'https'
}

/**
 * Get complete environment information
 * 
 * TEAM-351: Comprehensive environment data
 * 
 * @returns Environment information object
 */
export function getEnvironmentInfo(): EnvironmentInfo {
  return {
    isDev: isDevelopment(),
    isProd: isProduction(),
    isSSR: isSSR(),
    port: getCurrentPort(),
    protocol: getProtocol(),
    hostname: getHostname(),
    url: isSSR() ? '' : window.location.href,
  }
}
