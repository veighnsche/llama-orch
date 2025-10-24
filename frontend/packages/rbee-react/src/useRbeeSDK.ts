// TEAM-291: ⚠️ DEPRECATED - This monolithic file has been split into modules
// TODO: DELETE THIS FILE after verifying build works
// New structure:
//   - src/types.ts
//   - src/utils.ts
//   - src/globalSlot.ts
//   - src/loader.ts
//   - src/hooks/useRbeeSDK.ts
//   - src/hooks/useRbeeSDKSuspense.ts

'use client';

import { useState, useEffect, useRef } from 'react';
import type { RbeeClient, HeartbeatMonitor, OperationBuilder } from '@rbee/sdk';

// TEAM-291: Public types
export interface RbeeSDK {
  RbeeClient: typeof RbeeClient;
  HeartbeatMonitor: typeof HeartbeatMonitor;
  OperationBuilder: typeof OperationBuilder;
}

export type LoadOptions = {
  timeoutMs?: number;
  maxAttempts?: number;
  baseBackoffMs?: number;
  initArg?: unknown;
  onReady?: (sdk: RbeeSDK) => void;
};

// TEAM-291: Global singleton slot (HMR-safe, singleflight)
type GlobalSlot = {
  promise?: Promise<{ sdk: RbeeSDK }>;
  value?: { sdk: RbeeSDK };
  error?: Error;
};

declare global {
  // eslint-disable-next-line no-var
  var __rbeeSDKInit_v1__: GlobalSlot | undefined;
}

function getGlobalSlot(): GlobalSlot {
  if (!globalThis.__rbeeSDKInit_v1__) {
    globalThis.__rbeeSDKInit_v1__ = {};
  }
  return globalThis.__rbeeSDKInit_v1__;
}

// TEAM-291: Timeout helper
function withTimeout<T>(p: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    p,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)
    ),
  ]);
}

// TEAM-291: Sleep helper
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// TEAM-291: Core loader with retries
async function actuallyLoadSDK(opts: Required<LoadOptions>): Promise<{ sdk: RbeeSDK }> {
  // Environment guards
  if (typeof window === 'undefined') {
    throw new Error('rbee SDK can only be initialized in the browser (client component).');
  }

  const hasWasm =
    typeof WebAssembly === 'object' && typeof WebAssembly.instantiate === 'function';
  if (!hasWasm) {
    throw new Error('WebAssembly is not supported in this environment.');
  }

  let lastError: Error | undefined;

  for (let attempt = 1; attempt <= opts.maxAttempts; attempt++) {
    try {
      // Import with timeout
      const mod = await withTimeout(
        import('@rbee/sdk'),
        opts.timeoutMs,
        `SDK import (attempt ${attempt}/${opts.maxAttempts})`
      );

      // Handle ESM/CJS default shims
      const wasmModule = (mod as any).default ?? mod;

      // Initialize if present
      if (typeof wasmModule.init === 'function') {
        const initResult = wasmModule.init(opts.initArg);
        // Await if thenable
        if (initResult && typeof initResult.then === 'function') {
          await withTimeout(
            initResult,
            opts.timeoutMs,
            `SDK init (attempt ${attempt}/${opts.maxAttempts})`
          );
        }
      }

      // Validate exports
      if (
        !wasmModule.RbeeClient ||
        !wasmModule.HeartbeatMonitor ||
        !wasmModule.OperationBuilder
      ) {
        throw new Error(
          'SDK exports missing: expected { RbeeClient, HeartbeatMonitor, OperationBuilder }.'
        );
      }

      // Success
      return {
        sdk: {
          RbeeClient: wasmModule.RbeeClient,
          HeartbeatMonitor: wasmModule.HeartbeatMonitor,
          OperationBuilder: wasmModule.OperationBuilder,
        },
      };
    } catch (err) {
      lastError = err as Error;

      // Don't retry on last attempt
      if (attempt < opts.maxAttempts) {
        // Jittered exponential backoff
        const baseDelay = Math.pow(2, attempt - 1) * opts.baseBackoffMs;
        const jitter = Math.random() * opts.baseBackoffMs;
        await sleep(baseDelay + jitter);
      }
    }
  }

  throw lastError || new Error('SDK load failed after all retry attempts.');
}

// TEAM-291: Singleflight loader
function loadSDKOnce(options?: LoadOptions): Promise<{ sdk: RbeeSDK }> {
  const slot = getGlobalSlot();

  // Already loaded
  if (slot.value) {
    return Promise.resolve(slot.value);
  }

  // Previous error
  if (slot.error) {
    return Promise.reject(slot.error);
  }

  // In-flight
  if (slot.promise) {
    return slot.promise;
  }

  // Start new load
  const opts: Required<LoadOptions> = {
    timeoutMs: options?.timeoutMs ?? 15000,
    maxAttempts: options?.maxAttempts ?? 3,
    baseBackoffMs: options?.baseBackoffMs ?? 300,
    initArg: options?.initArg,
    onReady: options?.onReady ?? (() => {}),
  };

  slot.promise = actuallyLoadSDK(opts)
    .then((result) => {
      slot.value = result;
      slot.promise = undefined;
      slot.error = undefined;
      opts.onReady(result.sdk);
      return result;
    })
    .catch((err) => {
      slot.error = err;
      slot.promise = undefined;
      throw err;
    });

  return slot.promise;
}

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
 * const client = new sdk.RbeeClient('http://localhost:8500');
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
 *   const client = new sdk.RbeeClient('http://localhost:8500');
 *   return <div>Ready!</div>;
 * }
 * ```
 */
export function useRbeeSDKSuspense(options?: LoadOptions): RbeeSDK {
  const slot = getGlobalSlot();

  // Already loaded
  if (slot.value) {
    return slot.value.sdk;
  }

  // Previous error
  if (slot.error) {
    throw slot.error;
  }

  // Throw promise for Suspense
  throw loadSDKOnce(options).then((r) => r.sdk);
}
