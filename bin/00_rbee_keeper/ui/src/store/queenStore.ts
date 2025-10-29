// TEAM-338: Zustand store for Queen service state and commands
// TEAM-351: Rewritten with query-based pattern (no promise caching)

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import { commands } from '@/generated/bindings'
import { withCommandExecution } from './commandUtils'
import { useEffect } from 'react'

// TEAM-338: Define QueenStatus with camelCase (Tauri bindings use snake_case)
export interface QueenStatus {
  isRunning: boolean
  isInstalled: boolean
}

// TEAM-351: Query state for Queen
interface QueenQuery {
  data: QueenStatus | null
  isLoading: boolean
  error: string | null
  lastFetch: number
}

interface QueenState {
  // TEAM-351: Query cache
  query: QueenQuery

  // TEAM-351: Query actions
  fetchQueen: (force?: boolean) => Promise<void>
  invalidate: () => Promise<void>
  
  // TEAM-351: Mutation actions
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
      query: {
        data: null,
        isLoading: false,
        error: null,
        lastFetch: 0,
      },

      // TEAM-351: Fetch Queen status with deduplication
      fetchQueen: async (force = false) => {
        const now = Date.now()
        const existing = get().query
        
        // Skip if fresh (< 5s old) and not forced
        if (!force && !existing.isLoading && (now - existing.lastFetch < 5000)) {
          return
        }
        
        // Skip if already loading
        if (existing.isLoading) {
          return
        }
        
        set((state) => {
          state.query = {
            data: existing.data,
            isLoading: true,
            error: null,
            lastFetch: now,
          }
        })
        
        try {
          const result = await commands.queenStatus()
          
          if (result.status === 'ok') {
            const status: QueenStatus = {
              isRunning: result.data.is_running,
              isInstalled: result.data.is_installed,
            }
            set((state) => {
              state.query = {
                data: status,
                isLoading: false,
                error: null,
                lastFetch: now,
              }
            })
          } else {
            throw new Error(result.error || 'Failed to fetch Queen status')
          }
        } catch (error) {
          set((state) => {
            state.query = {
              data: existing.data,
              isLoading: false,
              error: error instanceof Error ? error.message : 'Failed',
              lastFetch: now,
            }
          })
        }
      },
      
      // TEAM-351: Invalidate query
      invalidate: async () => {
        set((state) => {
          state.query.lastFetch = 0
        })
        await get().fetchQueen(true)
      },

      start: async () => {
        await withCommandExecution(() => commands.queenStart(), get().invalidate, 'Queen start')
      },

      stop: async () => {
        await withCommandExecution(() => commands.queenStop(), get().invalidate, 'Queen stop')
      },

      install: async () => {
        await withCommandExecution(() => commands.queenInstall(null), get().invalidate, 'Queen install')
      },

      rebuild: async () => {
        await withCommandExecution(() => commands.queenRebuild(false), get().invalidate, 'Queen rebuild')
      },

      uninstall: async () => {
        await withCommandExecution(() => commands.queenUninstall(), get().invalidate, 'Queen uninstall')
      },

      reset: () => {
        set((state) => {
          state.query = {
            data: null,
            isLoading: false,
            error: null,
            lastFetch: 0,
          }
        })
      },
    })),
    {
      name: 'queen-store',
      partialize: () => ({}), // Don't persist query cache
    },
  ),
)

// TEAM-351: Query hook for components
// TEAM-355: Fixed - extract fetchQueen to avoid store in deps (prevents infinite loop)
export function useQueen() {
  const store = useQueenStore()
  const query = store.query
  const fetchQueen = store.fetchQueen
  
  useEffect(() => {
    fetchQueen()
  }, [fetchQueen])
  
  return {
    queen: query.data,
    isLoading: query.isLoading,
    error: query.error,
    refetch: () => store.fetchQueen(true),
  }
}

// TEAM-351: Action hooks for mutations
export function useQueenActions() {
  const store = useQueenStore()
  
  return {
    start: store.start,
    stop: store.stop,
    install: store.install,
    rebuild: store.rebuild,
    uninstall: store.uninstall,
  }
}
