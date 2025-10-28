// TEAM-291: Suspense-compatible hook for rbee SDK

'use client'

import { getGlobalSlot } from '../globalSlot'
import { loadSDKOnce } from '../loader'
import type { LoadOptions, RbeeSDK } from '../types'

/**
 * Suspense-compatible hook for loading the rbee WASM SDK
 *
 * Throws a promise until ready, then returns the SDK. Use inside a Suspense boundary.
 *
 * @param options - Load configuration (timeoutMs, maxAttempts, baseBackoffMs, initArg, onReady)
 * @returns RbeeSDK instance
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const sdk = useRbeeSDKSuspense();
 *   const client = new sdk.RbeeClient('http://localhost:7833');
 *   return <div>Ready!</div>;
 * }
 * ```
 */
export function useRbeeSDKSuspense(options?: LoadOptions): RbeeSDK {
  const slot = getGlobalSlot()

  // Already loaded
  if (slot.value) {
    return slot.value.sdk
  }

  // Previous error
  if (slot.error) {
    throw slot.error
  }

  // Throw promise for Suspense
  throw loadSDKOnce(options).then((r) => r.sdk)
}
