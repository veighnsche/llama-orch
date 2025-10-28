// TEAM-294: Zustand store for command execution state
import { create } from 'zustand'

interface CommandState {
  activeCommand: string | undefined
  isExecuting: boolean
  setActiveCommand: (command: string | undefined) => void
  setIsExecuting: (isExecuting: boolean) => void
  resetCommand: () => void
}

export const useCommandStore = create<CommandState>((set) => ({
  activeCommand: undefined,
  isExecuting: false,
  setActiveCommand: (command) => set({ activeCommand: command }),
  setIsExecuting: (isExecuting) => set({ isExecuting }),
  resetCommand: () => set({ activeCommand: undefined, isExecuting: false }),
}))
