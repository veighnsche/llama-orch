// TEAM-288: React hook for heartbeat monitoring

'use client';

import { useState, useEffect, useRef } from 'react';
import { useRbeeSDK } from './useRbeeSDK';

interface HeartbeatSnapshot {
  timestamp: string;
  workers_online: number;
  workers_available: number;
  hives_online: number;
  hives_available: number;
  worker_ids: string[];
  hive_ids: string[];
}

export function useHeartbeat(baseUrl: string = 'http://localhost:8500') {
  const { sdk, loading: sdkLoading } = useRbeeSDK();
  const [heartbeat, setHeartbeat] = useState<HeartbeatSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
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
        setHeartbeat(snapshot);
        setConnected(true);
        setError(null);
      });

      console.log('🐝 [useHeartbeat] Monitor.start() called');

      // Check connection after a moment
      setTimeout(() => {
        const isConn = monitor.isConnected();
        console.log('🐝 [useHeartbeat] Connection check after 1s, isConnected:', isConn);
        if (isConn) {
          setConnected(true);
        }
      }, 1000);
    } catch (err) {
      console.error('🐝 [useHeartbeat] ERROR starting monitor:', err);
      setError(err as Error);
      setConnected(false);
    }

    // Cleanup
    return () => {
      console.log('🐝 [useHeartbeat] Cleanup: stopping monitor');
      if (monitorRef.current) {
        monitorRef.current.stop();
      }
    };
  }, [sdk, baseUrl]);

  return {
    heartbeat,
    connected,
    loading: sdkLoading,
    error,
  };
}
