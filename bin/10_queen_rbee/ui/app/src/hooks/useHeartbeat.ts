// TEAM-291: Heartbeat hook for real-time system status
// Ported from web-ui.old - Connects rbee SDK directly to zustand store
// TEAM-294: Updated to use @rbee/queen-rbee-react

import { useEffect } from 'react';
import { useRbeeSDK } from '@rbee/queen-rbee-react';
import { useRbeeStore } from '../stores/rbeeStore';

/**
 * Initialize heartbeat monitoring by connecting SDK to zustand store
 * 
 * @param baseUrl - Queen rbee base URL (default: http://localhost:8500)
 * @returns SDK loading state and connection error if any
 */
export function useHeartbeat(baseUrl: string = 'http://localhost:8500') {
  const { sdk, loading, error } = useRbeeSDK();
  const { startMonitoring, stopMonitoring, setQueenError, queen } = useRbeeStore();

  useEffect(() => {
    if (!sdk) return;

    try {
      const monitor = new sdk.HeartbeatMonitor(baseUrl);
      startMonitoring(monitor, baseUrl);
    } catch (err) {
      setQueenError((err as Error).message);
    }

    return () => {
      stopMonitoring();
    };
  }, [sdk, baseUrl, startMonitoring, stopMonitoring, setQueenError]);

  return {
    connected: queen.connected,
    loading,
    error: error || (queen.error ? new Error(queen.error) : null),
  };
}
