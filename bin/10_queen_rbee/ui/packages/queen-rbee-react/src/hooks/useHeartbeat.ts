// TEAM-352: Migrated to use @rbee/react-hooks
// Old implementation: ~94 LOC of manual SSE connection + health check
// New implementation: ~40 LOC using shared hooks
// Reduction: 54 LOC (57%)

"use client";

import * as React from "react";

// TEAM-364: Updated to match backend HeartbeatEvent structure
export interface ProcessStats {
  pid: number;
  group: string;
  instance: string;
  cpu_pct: number;
  rss_mb: number;
  io_r_mb_s: number;
  io_w_mb_s: number;
  uptime_s: number;
  gpu_util_pct: number;
  vram_mb: number;
  total_vram_mb: number;
  model: string | null;
}

export interface HiveData {
  hive_id: string;
  workers: ProcessStats[];
  last_update: string;
}

export interface HeartbeatData {
  workers_online: number;
  workers_available: number;
  hives_online: number;
  hives_available: number;
  hives: HiveData[];
  timestamp: string;
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
 * TEAM-364: Updated to handle HeartbeatEvent stream (hive_telemetry + queen)
 * Aggregates hive telemetry events into a unified view
 *
 * @param baseUrl - Queen API URL (default: http://localhost:7833)
 * @returns Heartbeat data and connection status
 */
export function useHeartbeat(
  baseUrl: string = "http://localhost:7833",
): UseHeartbeatResult {
  const [data, setData] = React.useState<HeartbeatData | null>(null);
  const [connected, setConnected] = React.useState(false);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<Error | null>(null);

  // TEAM-364: Store hive telemetry by hive_id
  const hivesRef = React.useRef<Map<string, HiveData>>(new Map());
  const queenDataRef = React.useRef<{
    workers_online: number;
    workers_available: number;
    hives_online: number;
    hives_available: number;
    timestamp: string;
  } | null>(null);

  React.useEffect(() => {
    const eventSource = new EventSource(`${baseUrl}/v1/heartbeats/stream`);

    eventSource.onopen = () => {
      setConnected(true);
      setLoading(false);
      setError(null);
    };

    eventSource.addEventListener('heartbeat', (event) => {
      try {
        const heartbeatEvent = JSON.parse(event.data);

        if (heartbeatEvent.type === 'hive_telemetry') {
          // TEAM-364: Update hive telemetry
          hivesRef.current.set(heartbeatEvent.hive_id, {
            hive_id: heartbeatEvent.hive_id,
            workers: heartbeatEvent.workers,
            last_update: heartbeatEvent.timestamp,
          });
        } else if (heartbeatEvent.type === 'queen') {
          // TEAM-364: Update queen stats
          queenDataRef.current = {
            workers_online: heartbeatEvent.workers_online,
            workers_available: heartbeatEvent.workers_available,
            hives_online: heartbeatEvent.hives_online,
            hives_available: heartbeatEvent.hives_available,
            timestamp: heartbeatEvent.timestamp,
          };
        }

        // TEAM-364: Aggregate data
        const aggregated: HeartbeatData = {
          workers_online: queenDataRef.current?.workers_online ?? 0,
          workers_available: queenDataRef.current?.workers_available ?? 0,
          hives_online: queenDataRef.current?.hives_online ?? 0,
          hives_available: queenDataRef.current?.hives_available ?? 0,
          hives: Array.from(hivesRef.current.values()),
          timestamp: queenDataRef.current?.timestamp ?? new Date().toISOString(),
        };

        setData(aggregated);
      } catch (err) {
        console.error('Failed to parse heartbeat event:', err);
      }
    });

    eventSource.onerror = () => {
      setConnected(false);
      setError(new Error('SSE connection failed'));
    };

    return () => {
      eventSource.close();
    };
  }, [baseUrl]);

  return {
    data,
    connected,
    loading,
    error,
  };
}
