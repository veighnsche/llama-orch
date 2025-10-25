// TEAM-291: Core SDK loader with retries and timeout

import type { RbeeSDK, LoadOptions } from './types';
import { withTimeout, sleep } from './utils';
import { getGlobalSlot } from './globalSlot';

/**
 * Core loader with retries and backoff
 */
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
        import('@rbee/queen-rbee-sdk'),
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

/**
 * Singleflight loader - ensures only one load happens at a time
 */
export function loadSDKOnce(options?: LoadOptions): Promise<{ sdk: RbeeSDK }> {
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
