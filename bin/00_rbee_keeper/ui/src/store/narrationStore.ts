// TEAM-339: Zustand store for narration events
// Persists narration events even when panel is closed
// Listens to Tauri events at app level
// TEAM-340: Added showNarration state for layout awareness

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { immer } from 'zustand/middleware/immer'
import type { NarrationEvent } from '../generated/bindings'

export interface NarrationEntry extends NarrationEvent {
  id: number
}

interface NarrationState {
  entries: NarrationEntry[]
  idCounter: number
  showNarration: boolean

  // Actions
  addEntry: (event: NarrationEvent) => void
  clearEntries: () => void
  setShowNarration: (show: boolean) => void
}

export const useNarrationStore = create<NarrationState>()(
  persist(
    immer((set) => ({
      entries: [],
      idCounter: 0,
      showNarration: true,

      addEntry: (event: NarrationEvent) => {
        set((state) => {
          // Prepend new entry to top (newest first)
          state.entries.unshift({
            ...event,
            id: state.idCounter++,
          })
        })
      },

      clearEntries: () => {
        set((state) => {
          state.entries = []
          state.idCounter = 0
        })
      },

      setShowNarration: (show: boolean) => {
        set((state) => {
          state.showNarration = show
        })
      },
    })),
    {
      name: 'narration-store',
      partialize: (state) => ({
        entries: state.entries.slice(0, 100), // Keep last 100 entries
        idCounter: state.idCounter,
        showNarration: state.showNarration, // Persist panel state
      }),
    },
  ),
)
