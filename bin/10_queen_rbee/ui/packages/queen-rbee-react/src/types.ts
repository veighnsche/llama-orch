// TEAM-291: Type definitions for rbee SDK
// TEAM-295: Fixed import to use @rbee/queen-rbee-sdk instead of @rbee/sdk

import type { RbeeClient, HeartbeatMonitor, OperationBuilder } from '@rbee/queen-rbee-sdk';

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

export type GlobalSlot = {
  promise?: Promise<{ sdk: RbeeSDK }>;
  value?: { sdk: RbeeSDK };
  error?: Error;
};
