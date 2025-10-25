// TEAM-291: Type definitions for rbee SDK

import type { RbeeClient, HeartbeatMonitor, OperationBuilder } from '@rbee/sdk';

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
