// TEAM-291: Zustand store for Queen status and hives

import { create } from 'zustand';

// TEAM-291: Types from heartbeat snapshot
export interface HeartbeatSnapshot {
  timestamp: string;
  workers_online: number;
  workers_available: number;
  hives_online: number;
  hives_available: number;
  worker_ids: string[];
  hive_ids: string[];
}

export interface QueenStatus {
  connected: boolean;
  lastUpdate: string | null;
  error: string | null;
}

export interface HiveInfo {
  id: string;
  status: 'online' | 'offline';
  lastSeen: string;
}

// TEAM-291: Store state interface
interface RbeeState {
  // Queen status
  queen: QueenStatus;
  
  // Hives
  hives: HiveInfo[];
  hivesOnline: number;
  hivesAvailable: number;
  
  // Workers (for completeness)
  workersOnline: number;
  workersAvailable: number;
  workerIds: string[];
  
  // Raw heartbeat
  lastHeartbeat: HeartbeatSnapshot | null;
  
  // Actions
  setQueenConnected: (connected: boolean) => void;
  setQueenError: (error: string | null) => void;
  updateFromHeartbeat: (heartbeat: HeartbeatSnapshot) => void;
  resetState: () => void;
}

// TEAM-291: Initial state
const initialState = {
  queen: {
    connected: false,
    lastUpdate: null,
    error: null,
  },
  hives: [],
  hivesOnline: 0,
  hivesAvailable: 0,
  workersOnline: 0,
  workersAvailable: 0,
  workerIds: [],
  lastHeartbeat: null,
};

// TEAM-291: Create zustand store
export const useRbeeStore = create<RbeeState>((set) => ({
  ...initialState,

  // TEAM-291: Set queen connection status
  setQueenConnected: (connected: boolean) =>
    set((state) => ({
      queen: {
        ...state.queen,
        connected,
        lastUpdate: connected ? new Date().toISOString() : state.queen.lastUpdate,
      },
    })),

  // TEAM-291: Set queen error
  setQueenError: (error: string | null) =>
    set((state) => ({
      queen: {
        ...state.queen,
        error,
        connected: error ? false : state.queen.connected,
      },
    })),

  // TEAM-291: Update from heartbeat snapshot
  updateFromHeartbeat: (heartbeat: HeartbeatSnapshot) =>
    set(() => {
      // TEAM-291: Convert hive_ids to HiveInfo objects
      const hives: HiveInfo[] = heartbeat.hive_ids.map((id) => ({
        id,
        status: 'online' as const,
        lastSeen: heartbeat.timestamp,
      }));

      return {
        queen: {
          connected: true,
          lastUpdate: heartbeat.timestamp,
          error: null,
        },
        hives,
        hivesOnline: heartbeat.hives_online,
        hivesAvailable: heartbeat.hives_available,
        workersOnline: heartbeat.workers_online,
        workersAvailable: heartbeat.workers_available,
        workerIds: heartbeat.worker_ids,
        lastHeartbeat: heartbeat,
      };
    }),

  // TEAM-291: Reset to initial state
  resetState: () => set(initialState),
}));
