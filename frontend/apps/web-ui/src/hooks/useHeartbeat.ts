// TEAM-288: React hook for heartbeat monitoring
// TEAM-291: Updated to use zustand store and @rbee/react package

'use client';

import { useEffect, useRef } from 'react';
import { useRbeeSDK } from '@rbee/react';
import { useRbeeStore, type HeartbeatSnapshot } from '@/src/stores/rbeeStore';

export function useHeartbeat(baseUrl: string = 'http://localhost:8500') {
  const { sdk, loading: sdkLoading, error: sdkError } = useRbeeSDK();
  
  // TEAM-291: Get store actions and state
  const { 
    updateFromHeartbeat, 
    setQueenConnected, 
    setQueenError,
    queen,
  } = useRbeeStore();
  
  // TEAM-288: Using any for WASM HeartbeatMonitor instance (no TypeScript types available)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const monitorRef = useRef<any>(null);

  useEffect(() => {
    console.log('🐝 [useHeartbeat] Effect running, sdk:', sdk ? 'LOADED' : 'NULL');
    
    if (!sdk) {
      console.log('🐝 [useHeartbeat] SDK not ready yet, waiting...');
      return;
    }

    console.log('🐝 [useHeartbeat] Creating HeartbeatMonitor with baseUrl:', baseUrl);
    
    // Create monitor
    const monitor = new sdk.HeartbeatMonitor(baseUrl);
    monitorRef.current = monitor;

    console.log('🐝 [useHeartbeat] HeartbeatMonitor created:', monitor);

    try {
      // Start monitoring
      console.log('🐝 [useHeartbeat] Starting monitor...');
      monitor.start((snapshot: HeartbeatSnapshot) => {
        console.log('🐝 [useHeartbeat] CALLBACK FIRED! Received snapshot:', snapshot);
        // TEAM-291: Update zustand store instead of local state
        updateFromHeartbeat(snapshot);
      });

      console.log('🐝 [useHeartbeat] Monitor.start() called');

      // Check connection after a moment
      setTimeout(() => {
        const isConn = monitor.isConnected();
        console.log('🐝 [useHeartbeat] Connection check after 1s, isConnected:', isConn);
        if (isConn) {
          // TEAM-291: Update store
          setQueenConnected(true);
        }
      }, 1000);
    } catch (err) {
      console.error('🐝 [useHeartbeat] ERROR starting monitor:', err);
      // TEAM-291: Update store with error
      setQueenError((err as Error).message);
    }

    // Cleanup
    return () => {
      console.log('🐝 [useHeartbeat] Cleanup: stopping monitor');
      if (monitorRef.current) {
        monitorRef.current.stop();
      }
    };
  }, [sdk, baseUrl, updateFromHeartbeat, setQueenConnected, setQueenError]);

  // TEAM-291: Return store state instead of local state
  return {
    connected: queen.connected,
    loading: sdkLoading,
    error: sdkError || (queen.error ? new Error(queen.error) : null),
  };
}
