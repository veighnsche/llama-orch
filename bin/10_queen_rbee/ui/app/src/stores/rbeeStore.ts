// TEAM-292: Zustand store for Queen status and hives
// Ported from web-ui.old - Connected directly to rbee SDK
// TEAM-294: Updated to use @rbee/queen-rbee-react

import type { HeartbeatMonitor } from '@rbee/queen-rbee-react'
import { create } from 'zustand'

// TEAM-292: Types from heartbeat snapshot
export interface HeartbeatSnapshot {
  timestamp: string
  workers_online: number
  workers_available: number
  hives_online: number
  hives_available: number
  worker_ids: string[]
  hive_ids: string[]
}

export interface QueenStatus {
  connected: boolean
  lastUpdate: string | null
  error: string | null
}

export interface HiveInfo {
  id: string
  status: 'online' | 'offline'
  lastSeen: string
}

// TEAM-292: Store state interface
interface RbeeState {
  // Queen status
  queen: QueenStatus

  // Hives
  hives: HiveInfo[]
  hivesOnline: number
  hivesAvailable: number

  // Workers (for completeness)
  workersOnline: number
  workersAvailable: number
  workerIds: string[]

  // Raw heartbeat
  lastHeartbeat: HeartbeatSnapshot | null

  // SDK connection
  monitor: HeartbeatMonitor | null

  // Actions
  setQueenError: (error: string | null) => void
  startMonitoring: (monitor: HeartbeatMonitor, baseUrl: string) => void
  stopMonitoring: () => void
  resetState: () => void
}

// TEAM-292: Initial state
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
  monitor: null,
}

// TEAM-292: Create zustand store
export const useRbeeStore = create<RbeeState>((set) => ({
  ...initialState,

  // TEAM-292: Set queen error
  setQueenError: (error: string | null) =>
    set((state) => ({
      queen: {
        ...state.queen,
        error,
        connected: error ? false : state.queen.connected,
      },
    })),

  // TEAM-292: Start monitoring heartbeats
  // This callback fires every ~5 seconds when queen sends heartbeat
  startMonitoring: (monitorInstance: HeartbeatMonitor, _baseUrl: string) =>
    set((state) => {
      // Stop existing monitor if any
      if (state.monitor) {
        state.monitor.stop()
      }

      // Start new monitor - callback fires on EVERY heartbeat from queen
      monitorInstance.start((snapshot: HeartbeatSnapshot) => {
        // REAL-TIME UPDATE: This runs every time queen sends a heartbeat
        const hives: HiveInfo[] = snapshot.hive_ids.map((id) => ({
          id,
          status: 'online' as const,
          lastSeen: snapshot.timestamp,
        }))

        // Update store immediately with latest data
        set({
          queen: {
            connected: true,
            lastUpdate: snapshot.timestamp,
            error: null,
          },
          hives,
          hivesOnline: snapshot.hives_online,
          hivesAvailable: snapshot.hives_available,
          workersOnline: snapshot.workers_online,
          workersAvailable: snapshot.workers_available,
          workerIds: snapshot.worker_ids,
          lastHeartbeat: snapshot,
        })
      })

      return { monitor: monitorInstance }
    }),

  // TEAM-292: Stop monitoring
  stopMonitoring: () =>
    set((state) => {
      if (state.monitor) {
        state.monitor.stop()
      }
      return { monitor: null }
    }),

  // TEAM-292: Reset to initial state
  resetState: () =>
    set((state) => {
      if (state.monitor) {
        state.monitor.stop()
      }
      return initialState
    }),
}))
