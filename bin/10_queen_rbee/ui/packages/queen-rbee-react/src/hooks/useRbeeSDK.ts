// TEAM-377: React hook for Queen WASM SDK
// Uses @rbee/sdk-loader to load @rbee/queen-rbee-sdk with retry logic

'use client'

import { useEffect, useRef, useState } from 'react'
import { createSDKLoader } from '@rbee/sdk-loader'
import type { RbeeSDK } from '../types'

// TEAM-377: Create loader instance for Queen SDK (renamed from generic "rbee")
const queenSDKLoader = createSDKLoader<RbeeSDK>({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor', 'OperationBuilder', 'RhaiClient'],
  timeout: 15000,
  maxAttempts: 3,
})

/**
 * React hook for loading the Queen WASM SDK (client-only, retries with backoff)
 *
 * TEAM-377: Renamed from useRbeeSDK to useQueenSDK for clarity
 * - There is no generic "rbee SDK"
 * - This specifically loads @rbee/queen-rbee-sdk
 * - Hive has its own SDK: @rbee/rbee-hive-sdk
 *
 * @returns Object with sdk (RbeeSDK | null), loading (boolean), error (Error | null)
 *
 * @example
 * ```tsx
 * const { sdk, loading, error } = useQueenSDK();
 * if (loading) return <div>Loading...</div>;
 * if (error) return <div>Error: {error.message}</div>;
 * const client = new sdk.QueenClient('http://localhost:7833');
 * ```
 */
export function useQueenSDK() {
  const [sdk, setSDK] = useState<RbeeSDK | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true

    // TEAM-352: Use shared loader with singleflight pattern
    queenSDKLoader
      .loadOnce()
      .then((result) => {
        if (mountedRef.current) {
          setSDK(result.sdk)
          setLoading(false)
        }
      })
      .catch((err) => {
        if (mountedRef.current) {
          setError(err)
          setLoading(false)
        }
      })

    return () => {
      mountedRef.current = false
    }
  }, [])

  return { sdk, loading, error }
}

// TEAM-377: Backward compatibility alias (deprecated)
// TODO: Remove this after all consumers are updated
/**
 * @deprecated Use useQueenSDK instead. This alias will be removed in a future version.
 */
export const useRbeeSDK = useQueenSDK;
