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
  const monitorRef = useRef<any>(null);

  useEffect(() => {
    if (!sdk) return;

    // Create monitor
    const monitor = new sdk.HeartbeatMonitor(baseUrl);
    monitorRef.current = monitor;

    try {
      // Start monitoring
      monitor.start((snapshot: HeartbeatSnapshot) => {
        setHeartbeat(snapshot);
        setConnected(true);
        setError(null);
      });

      // Check connection after a moment
      setTimeout(() => {
        if (monitor.isConnected()) {
          setConnected(true);
        }
      }, 1000);
    } catch (err) {
      setError(err as Error);
      setConnected(false);
    }

    // Cleanup
    return () => {
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
