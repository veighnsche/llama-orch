/**
 * TEAM-351: Startup logging utilities
 * TEAM-351: Bug fixes - Validation, log levels, timestamps, types
 */

import type { EnvironmentInfo } from './environment'
import { validatePort } from './environment'

/**
 * Log level
 * TEAM-351: Type-safe log levels
 */
export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

/**
 * Log options
 * TEAM-351: Configurable logging
 */
export interface LogOptions {
  timestamp?: boolean
  level?: LogLevel
  prefix?: string
  color?: boolean
}

/**
 * Startup log options
 * TEAM-351: Startup-specific options
 */
export interface StartupLogOptions extends LogOptions {
  showUrl?: boolean
  showProtocol?: boolean
  showHostname?: boolean
}

/**
 * Format timestamp
 * TEAM-351: Timestamp formatting
 */
function formatTimestamp(): string {
  const now = new Date()
  return now.toISOString().split('T')[1].split('.')[0] // HH:MM:SS
}

/**
 * Get log level emoji
 * TEAM-351: Visual log levels
 */
function getLogLevelEmoji(level: LogLevel): string {
  switch (level) {
    case 'debug': return '🐛'
    case 'info': return 'ℹ️'
    case 'warn': return '⚠️'
    case 'error': return '❌'
    default: return ''
  }
}

/**
 * Log with level and options
 * 
 * TEAM-351: Generic logging utility
 * 
 * @param level - Log level
 * @param message - Message to log
 * @param options - Log options
 */
export function log(
  level: LogLevel,
  message: string,
  options: LogOptions = {}
): void {
  const { timestamp = false, prefix = '', color = true } = options
  
  const parts: string[] = []
  
  // TEAM-351: Add timestamp
  if (timestamp) {
    parts.push(`[${formatTimestamp()}]`)
  }
  
  // TEAM-351: Add level emoji
  if (color) {
    parts.push(getLogLevelEmoji(level))
  }
  
  // TEAM-351: Add prefix
  if (prefix) {
    parts.push(`[${prefix}]`)
  }
  
  // TEAM-351: Add message
  parts.push(message)
  
  const fullMessage = parts.join(' ')
  
  // TEAM-351: Log at appropriate level
  switch (level) {
    case 'debug':
      console.debug(fullMessage)
      break
    case 'info':
      console.log(fullMessage)
      break
    case 'warn':
      console.warn(fullMessage)
      break
    case 'error':
      console.error(fullMessage)
      break
  }
}

/**
 * Log startup mode with validation and options
 * 
 * TEAM-351: Bug fixes - Validation, options, HTTPS support
 * 
 * @param serviceName - Service name (e.g., 'QUEEN UI')
 * @param isDev - Development mode flag
 * @param port - Port number (optional)
 * @param options - Startup log options
 */
export function logStartupMode(
  serviceName: string,
  isDev: boolean,
  port?: number,
  options: StartupLogOptions = {}
): void {
  const {
    timestamp = false,
    showUrl = true,
    showProtocol = false,
    showHostname = false,
  } = options
  
  // TEAM-351: Validate service name
  if (!serviceName || typeof serviceName !== 'string') {
    console.warn('[dev-utils] Invalid service name')
    return
  }
  
  // TEAM-351: Validate port if provided
  if (port !== undefined) {
    const validation = validatePort(port)
    if (!validation.valid) {
      console.warn(`[dev-utils] ${validation.error}`)
      port = undefined
    }
  }
  
  const emoji = isDev ? '🔧' : '🚀'
  const mode = isDev ? 'DEVELOPMENT' : 'PRODUCTION'
  
  // TEAM-351: Main log line
  const mainMessage = `${emoji} [${serviceName}] Running in ${mode} mode`
  if (timestamp) {
    console.log(`[${formatTimestamp()}] ${mainMessage}`)
  } else {
    console.log(mainMessage)
  }
  
  // TEAM-351: Development mode details
  if (isDev && port) {
    console.log(`   - Vite dev server active (hot reload enabled)`)
    
    if (showUrl) {
      const protocol = showProtocol ? 'http' : ''
      const hostname = showHostname ? window.location.hostname : 'localhost'
      const url = protocol ? `${protocol}://${hostname}:${port}` : `http://localhost:${port}`
      console.log(`   - Running on: ${url}`)
    }
  }
  
  // TEAM-351: Production mode details
  if (!isDev) {
    console.log(`   - Serving embedded static files`)
    
    if (showProtocol && typeof window !== 'undefined') {
      console.log(`   - Protocol: ${window.location.protocol.replace(':', '')}`)
    }
  }
}

/**
 * Log environment information
 * 
 * TEAM-351: Comprehensive environment logging
 * 
 * @param serviceName - Service name
 * @param envInfo - Environment information
 * @param options - Log options
 */
export function logEnvironmentInfo(
  serviceName: string,
  envInfo: EnvironmentInfo,
  options: LogOptions = {}
): void {
  const { timestamp = false } = options
  
  const prefix = timestamp ? `[${formatTimestamp()}]` : ''
  
  console.log(`${prefix} 🌍 [${serviceName}] Environment Information:`)
  console.log(`   - Mode: ${envInfo.isDev ? 'Development' : 'Production'}`)
  console.log(`   - SSR: ${envInfo.isSSR ? 'Yes' : 'No'}`)
  console.log(`   - Protocol: ${envInfo.protocol}`)
  console.log(`   - Hostname: ${envInfo.hostname || 'N/A'}`)
  console.log(`   - Port: ${envInfo.port || 'N/A'}`)
  if (envInfo.url) {
    console.log(`   - URL: ${envInfo.url}`)
  }
}

/**
 * Create a logger with prefix
 * 
 * TEAM-351: Logger factory
 * 
 * @param prefix - Logger prefix
 * @param options - Default log options
 * @returns Logger functions
 */
export function createLogger(prefix: string, options: LogOptions = {}) {
  return {
    debug: (message: string) => log('debug', message, { ...options, prefix }),
    info: (message: string) => log('info', message, { ...options, prefix }),
    warn: (message: string) => log('warn', message, { ...options, prefix }),
    error: (message: string) => log('error', message, { ...options, prefix }),
  }
}
