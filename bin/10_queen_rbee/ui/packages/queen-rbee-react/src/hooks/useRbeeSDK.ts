// TEAM-291: Standard React hook for rbee SDK

'use client';

import { useState, useEffect, useRef } from 'react';
import type { RbeeSDK, LoadOptions } from '../types';
import { loadSDKOnce } from '../loader';

/**
 * React hook for loading the rbee WASM SDK (client-only, retries with backoff)
 *
 * @param options - Load configuration (timeoutMs, maxAttempts, baseBackoffMs, initArg, onReady)
 * @returns Object with sdk (RbeeSDK | null), loading (boolean), error (Error | null)
 *
 * @example
 * ```tsx
 * const { sdk, loading, error } = useRbeeSDK({ onReady: (s) => console.log('Ready!') });
 * if (loading) return <div>Loading...</div>;
 * if (error) return <div>Error: {error.message}</div>;
 * const client = new sdk.RbeeClient('http://localhost:7833');
 * ```
 */
export function useRbeeSDK(options?: LoadOptions) {
  const [sdk, setSDK] = useState<RbeeSDK | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;

    // Options not in deps: we load once per mount, not on option changes
    loadSDKOnce(options)
      .then((result) => {
        if (mountedRef.current) {
          setSDK(result.sdk);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (mountedRef.current) {
          setError(err);
          setLoading(false);
        }
      });

    return () => {
      mountedRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { sdk, loading, error };
}
