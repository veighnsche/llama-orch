// TEAM-352: Migrated to use @rbee/react-hooks
// Old implementation: ~94 LOC of manual SSE connection + health check
// New implementation: ~40 LOC using shared hooks
// Reduction: 54 LOC (57%)

"use client";

import { useSSEWithHealthCheck } from "@rbee/react-hooks";
import { useRbeeSDK } from "./useRbeeSDK";

export interface HeartbeatData {
  workers_online: number;
  hives_online: number;
  timestamp: string;
  workers: Array<{
    id: string;
    model_id: string;
    device: number;
    port: number;
    status: string;
  }>;
}

export interface UseHeartbeatResult {
  data: HeartbeatData | null;
  connected: boolean;
  loading: boolean;
  error: Error | null;
}

/**
 * Hook for monitoring Queen heartbeat
 *
 * TEAM-352: Now uses @rbee/react-hooks for connection management
 *
 * @param baseUrl - Queen API URL (default: http://localhost:7833)
 * @returns Heartbeat data and connection status
 */
export function useHeartbeat(
  baseUrl: string = "http://localhost:7833",
): UseHeartbeatResult {
  const { sdk, loading: sdkLoading, error: sdkError } = useRbeeSDK();

  const {
    data,
    connected,
    loading: sseLoading,
    error: sseError,
  } = useSSEWithHealthCheck<HeartbeatData>(
    (url) => {
      if (!sdk) {
        throw new Error("SDK not loaded");
      }
      return new sdk.HeartbeatMonitor(url);
    },
    baseUrl,
    {
      autoRetry: true,
      retryDelayMs: 5000,
      maxRetries: 3,
    },
  );

  return {
    data,
    connected,
    loading: sdkLoading || sseLoading,
    error: sdkError || sseError,
  };
}
