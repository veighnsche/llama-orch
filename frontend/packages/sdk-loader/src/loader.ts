/**
 * TEAM-356: Core SDK loader with retry logic
 * 
 * Loads WASM/SDK modules with exponential backoff, timeout handling,
 * and export validation. Supports singleflight pattern to prevent
 * duplicate loads.
 */

import type { LoadOptions, SDKLoadResult } from './types'
import { withTimeout, sleep, calculateBackoff } from './utils'
import { getGlobalSlot } from './singleflight'

/**
 * Load SDK with retry logic and timeout
 * 
 * @param options - Load options
 * @returns SDK load result with timing information
 * @throws Error if load fails after all retry attempts
 * 
 * @example
 * ```typescript
 * const result = await loadSDK({
 *   packageName: '@rbee/queen-rbee-sdk',
 *   requiredExports: ['Client', 'Monitor'],
 *   timeout: 15000,
 *   maxAttempts: 3,
 * })
 * console.log(`Loaded in ${result.loadTime}ms after ${result.attempts} attempts`)
 * ```
 */
export async function loadSDK<T>(options: LoadOptions): Promise<SDKLoadResult<T>> {
  const {
    packageName,
    requiredExports,
    timeout = 15000,
    maxAttempts = 3,
    baseBackoffMs = 300,
    initArg,
  } = options

  // Environment guards
  if (typeof window === 'undefined') {
    throw new Error('SDK can only be loaded in browser environment')
  }

  if (typeof WebAssembly === 'undefined') {
    throw new Error('WebAssembly not supported in this browser')
  }

  const startTime = Date.now()
  let lastError: Error | undefined

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      // TEAM-377: Debug logging for SDK loading
      console.log(`[sdk-loader] Attempt ${attempt}/${maxAttempts}: Importing ${packageName}`)
      
      // Dynamic import with timeout
      const mod = await withTimeout(
        import(/* @vite-ignore */ packageName),
        timeout,
        `SDK import (attempt ${attempt}/${maxAttempts})`
      )
      
      console.log(`[sdk-loader] ✅ Import successful for ${packageName}`, mod)

      // Handle ESM/CJS shims (default export vs named exports)
      const wasmModule = (mod as any).default ?? mod

      // Initialize WASM if init function exists
      if (typeof wasmModule.init === 'function') {
        await withTimeout(
          wasmModule.init(initArg),
          timeout,
          'WASM initialization'
        )
      }

      // Validate required exports
      for (const exportName of requiredExports) {
        if (!wasmModule[exportName]) {
          throw new Error(`SDK missing required export: ${exportName}`)
        }
      }

      const loadTime = Date.now() - startTime
      return {
        sdk: wasmModule as T,
        loadTime,
        attempts: attempt,
      }
    } catch (err) {
      lastError = err as Error

      // Don't retry on last attempt
      if (attempt < maxAttempts) {
        const backoffMs = calculateBackoff(attempt, baseBackoffMs, baseBackoffMs)
        console.warn(
          `[sdk-loader] Attempt ${attempt}/${maxAttempts} failed, retrying in ${backoffMs}ms:`,
          lastError.message
        )
        await sleep(backoffMs)
      }
    }
  }

  throw lastError || new Error(`SDK load failed after ${maxAttempts} attempts`)
}

/**
 * Load SDK once (singleflight pattern)
 * 
 * Ensures only one load operation happens at a time per package.
 * If multiple callers request the same package simultaneously,
 * only one load executes and all callers receive the same result.
 * 
 * @param options - Load options
 * @returns SDK load result
 * @throws Error if load fails or previous load failed
 * 
 * @example
 * ```typescript
 * // Multiple concurrent calls - only one load executes
 * const [result1, result2] = await Promise.all([
 *   loadSDKOnce({ packageName: '@rbee/sdk', requiredExports: ['Client'] }),
 *   loadSDKOnce({ packageName: '@rbee/sdk', requiredExports: ['Client'] }),
 * ])
 * // result1.sdk === result2.sdk (same instance)
 * ```
 */
export async function loadSDKOnce<T>(options: LoadOptions): Promise<SDKLoadResult<T>> {
  const slot = getGlobalSlot<T>(options.packageName)

  // Already loaded successfully
  if (slot.value) {
    return slot.value
  }

  // Previous load failed
  if (slot.error) {
    throw slot.error
  }

  // Load in progress
  if (slot.promise) {
    return slot.promise
  }

  // Start new load
  slot.promise = loadSDK<T>(options)
    .then(result => {
      slot.value = result
      slot.promise = undefined
      return result
    })
    .catch(err => {
      slot.error = err
      slot.promise = undefined
      throw err
    })

  return slot.promise
}

/**
 * Create SDK loader factory with default options
 * 
 * @param defaultOptions - Default load options (without initArg)
 * @returns Factory object with load and loadOnce methods
 * 
 * @example
 * ```typescript
 * const queenLoader = createSDKLoader({
 *   packageName: '@rbee/queen-rbee-sdk',
 *   requiredExports: ['QueenClient', 'HeartbeatMonitor'],
 * })
 * 
 * // Load with optional init arg
 * const { sdk } = await queenLoader.loadOnce()
 * ```
 */
export function createSDKLoader<T>(defaultOptions: Omit<LoadOptions, 'initArg'>) {
  return {
    /**
     * Load SDK (may load multiple times)
     */
    load: (initArg?: any) => loadSDK<T>({ ...defaultOptions, initArg }),
    
    /**
     * Load SDK once (singleflight pattern)
     */
    loadOnce: (initArg?: any) => loadSDKOnce<T>({ ...defaultOptions, initArg }),
  }
}
