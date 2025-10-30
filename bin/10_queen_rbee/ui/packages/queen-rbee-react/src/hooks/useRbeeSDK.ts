// TEAM-291: Standard React hook for rbee SDK
// TEAM-352: Migrated to use @rbee/sdk-loader directly (no wrapper)

'use client'

import { useEffect, useRef, useState } from 'react'
import { createSDKLoader } from '@rbee/sdk-loader'
import type { RbeeSDK } from '../types'

// TEAM-352: Create loader instance for Queen SDK
const queenSDKLoader = createSDKLoader<RbeeSDK>({
  packageName: '@rbee/queen-rbee-sdk',
  requiredExports: ['QueenClient', 'HeartbeatMonitor', 'OperationBuilder', 'RhaiClient'],
  timeout: 15000,
  maxAttempts: 3,
})

/**
 * React hook for loading the rbee WASM SDK (client-only, retries with backoff)
 *
 * @returns Object with sdk (RbeeSDK | null), loading (boolean), error (Error | null)
 *
 * @example
 * ```tsx
 * const { sdk, loading, error } = useRbeeSDK();
 * if (loading) return <div>Loading...</div>;
 * if (error) return <div>Error: {error.message}</div>;
 * const client = new sdk.QueenClient('http://localhost:7833');
 * ```
 */
export function useRbeeSDK() {
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
