// TEAM-338: Zustand store for Queen service state and commands
// TEAM-338: Added persist and immer middleware

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import { commands } from '@/generated/bindings'
import { withCommandExecution } from './commandUtils'

// TEAM-338: Define QueenStatus with camelCase (Tauri bindings use snake_case)
export interface QueenStatus {
  isRunning: boolean
  isInstalled: boolean
}

interface QueenState {
  status: QueenStatus | null
  isLoading: boolean
  error: string | null
  _fetchPromise: Promise<void> | null // TEAM-341: Cache in-flight fetch to prevent race conditions

  // Actions
  fetchStatus: () => Promise<void>
  start: () => Promise<void>
  stop: () => Promise<void>
  install: () => Promise<void>
  rebuild: () => Promise<void>
  uninstall: () => Promise<void>
  reset: () => void
}

export const useQueenStore = create<QueenState>()(
  persist(
    immer((set, get) => ({
      status: null,
      isLoading: false,
      error: null,
      _fetchPromise: null,

      fetchStatus: async () => {
        // TEAM-341: Return existing promise if fetch is already in progress
        // This prevents race conditions when multiple components mount simultaneously
        const existing = get()._fetchPromise
        if (existing) return existing

        const promise = (async () => {
          set((state) => {
            state.isLoading = true
            state.error = null
          })
          try {
            // TEAM-338: Call Tauri command to get queen status
            const result = await commands.queenStatus()

            if (result.status === 'ok') {
              // Convert snake_case to camelCase
              const status: QueenStatus = {
                isRunning: result.data.is_running,
                isInstalled: result.data.is_installed,
              }
              set((state) => {
                state.status = status
                state.isLoading = false
              })
            } else {
              set((state) => {
                state.error = result.error
                state.isLoading = false
              })
              throw new Error(result.error || 'Failed to fetch Queen status')
            }
          } catch (error) {
            set((state) => {
              state.error = error instanceof Error ? error.message : 'Failed to fetch Queen status'
              state.isLoading = false
            })
            throw error
          } finally {
            // TEAM-341: Clear promise cache after completion (success or error)
            set((state) => {
              state._fetchPromise = null
            })
          }
        })()

        set((state) => {
          state._fetchPromise = promise
        })
        return promise
      },

      start: async () => {
        await withCommandExecution(() => commands.queenStart(), get().fetchStatus, 'Queen start')
      },

      stop: async () => {
        await withCommandExecution(() => commands.queenStop(), get().fetchStatus, 'Queen stop')
      },

      install: async () => {
        await withCommandExecution(() => commands.queenInstall(null), get().fetchStatus, 'Queen install')
      },

      rebuild: async () => {
        await withCommandExecution(() => commands.queenRebuild(false), get().fetchStatus, 'Queen rebuild')
      },

      uninstall: async () => {
        await withCommandExecution(() => commands.queenUninstall(), get().fetchStatus, 'Queen uninstall')
      },

      reset: () => {
        set((state) => {
          state.status = null
          state.isLoading = false
          state.error = null
          state._fetchPromise = null
        })
      },
    })),
    {
      name: 'queen-store',
      partialize: (state) => ({
        status: state.status,
      }),
    },
  ),
)
