// TEAM-294: Zustand store for command execution and narration streaming
import { create } from 'zustand';

interface CommandState {
  // Current command execution
  activeCommand: string | undefined;
  isExecuting: boolean;
  
  // Output lines (streaming narration)
  outputLines: string[];
  
  // Actions
  setActiveCommand: (command: string | undefined) => void;
  setIsExecuting: (isExecuting: boolean) => void;
  appendOutput: (line: string) => void;
  clearOutput: () => void;
  resetCommand: () => void;
}

export const useCommandStore = create<CommandState>((set) => ({
  // Initial state
  activeCommand: undefined,
  isExecuting: false,
  outputLines: ['Click a command to execute...'],
  
  // Actions
  setActiveCommand: (command) => set({ activeCommand: command }),
  
  setIsExecuting: (isExecuting) => set({ isExecuting }),
  
  appendOutput: (line) => set((state) => ({
    outputLines: [...state.outputLines, line]
  })),
  
  clearOutput: () => set({ outputLines: [] }),
  
  resetCommand: () => set({
    activeCommand: undefined,
    isExecuting: false,
  }),
}));
