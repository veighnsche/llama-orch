/**
 * TEAM-356: SDK loader types
 * 
 * Type definitions for WASM/SDK loading with retry logic and timeout handling.
 */

/**
 * Options for loading an SDK/WASM module
 */
export interface LoadOptions {
  /** Package name to import (e.g., '@rbee/queen-rbee-sdk') */
  packageName: string
  
  /** Required exports to validate (e.g., ['Client', 'Monitor']) */
  requiredExports: string[]
  
  /** Timeout in milliseconds (default: 15000) */
  timeout?: number
  
  /** Max retry attempts (default: 3) */
  maxAttempts?: number
  
  /** Base backoff delay in ms (default: 300) */
  baseBackoffMs?: number
  
  /** Initialization argument (for WASM init) */
  initArg?: any
}

/**
 * Result of SDK loading operation
 */
export interface SDKLoadResult<T> {
  /** Loaded SDK module */
  sdk: T
  
  /** Total load time in milliseconds */
  loadTime: number
  
  /** Number of attempts required */
  attempts: number
}

/**
 * Global slot for singleflight pattern
 * Ensures only one load operation happens at a time per package
 */
export interface GlobalSlot<T> {
  /** Successfully loaded SDK result */
  value?: SDKLoadResult<T>
  
  /** Error from failed load attempt */
  error?: Error
  
  /** In-progress load promise */
  promise?: Promise<SDKLoadResult<T>>
}
