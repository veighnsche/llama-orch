// TEAM-291: Global singleton slot (HMR-safe, singleflight)

import type { GlobalSlot } from './types';

declare global {
  // eslint-disable-next-line no-var
  var __rbeeSDKInit_v1__: GlobalSlot | undefined;
}

/**
 * Get or create the global singleton slot
 * Survives HMR by using globalThis
 */
export function getGlobalSlot(): GlobalSlot {
  if (!globalThis.__rbeeSDKInit_v1__) {
    globalThis.__rbeeSDKInit_v1__ = {};
  }
  return globalThis.__rbeeSDKInit_v1__;
}
