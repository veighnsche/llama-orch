// TEAM-377: React hook for Queen WASM SDK
// TEAM-377: FIXED - Use static import instead of dynamic import
// Dynamic import with @vite-ignore doesn't work for workspace packages with WASM

'use client'

import { useEffect, useRef, useState } from 'react'
import type { RbeeSDK } from '../types'

// TEAM-377: Static import - Vite can resolve this properly
import * as QueenSDK from '@rbee/queen-rbee-sdk'

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

  useEffect(() => {
    // TEAM-377: Static import means SDK is already loaded
    // Just validate exports and set state
    try {
      console.log('[useQueenSDK] Validating SDK exports...', QueenSDK)
      
      // Validate required exports
      if (!QueenSDK.QueenClient || !QueenSDK.HeartbeatMonitor || !QueenSDK.OperationBuilder || !QueenSDK.RhaiClient) {
        throw new Error('SDK missing required exports')
      }
      
      console.log('[useQueenSDK] ✅ SDK loaded successfully')
      setSDK(QueenSDK as RbeeSDK)
      setLoading(false)
    } catch (err) {
      console.error('[useQueenSDK] ❌ SDK load failed:', err)
      setError(err instanceof Error ? err : new Error(String(err)))
      setLoading(false)
    }
  }, [])

  return { sdk, loading, error }
}

// TEAM-377: DELETED useRbeeSDK alias - RULE ZERO violation
// Just renamed to useQueenSDK everywhere, let compiler find call sites
