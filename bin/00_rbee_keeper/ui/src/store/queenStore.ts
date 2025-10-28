// TEAM-338: Zustand store for Queen service state and commands
// Imports commandStore internally to manage global isExecuting state
import { create } from "zustand";
import { commands } from "@/generated/bindings";
import { useCommandStore } from "./commandStore";

export interface QueenStatus {
  isRunning: boolean;
  isInstalled: boolean;
}

interface QueenState {
  status: QueenStatus | null;
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchStatus: () => Promise<void>;
  start: () => Promise<void>;
  stop: () => Promise<void>;
  install: () => Promise<void>;
  rebuild: () => Promise<void>;
  uninstall: () => Promise<void>;
  reset: () => void;
}

// Helper to wrap commands with global isExecuting state
const withCommandExecution = async (
  commandFn: () => Promise<unknown>,
  refreshFn: () => Promise<void>,
) => {
  const { setIsExecuting } = useCommandStore.getState();
  setIsExecuting(true);
  try {
    await commandFn();
    await refreshFn();
  } catch (error) {
    console.error("Queen command failed:", error);
    throw error;
  } finally {
    setIsExecuting(false);
  }
};

export const useQueenStore = create<QueenState>((set, get) => ({
  status: null,
  isLoading: false,
  error: null,

  fetchStatus: async () => {
    set({ isLoading: true, error: null });
    try {
      // TEAM-338: For now, return a mock status until we have a status command
      // TODO: Replace with actual queen_status command when available
      const status: QueenStatus = {
        isRunning: false,
        isInstalled: false,
      };
      set({ status, isLoading: false });
    } catch (error) {
      set({
        error:
          error instanceof Error
            ? error.message
            : "Failed to fetch Queen status",
        isLoading: false,
      });
    }
  },

  start: async () => {
    await withCommandExecution(() => commands.queenStart(), get().fetchStatus);
  },

  stop: async () => {
    await withCommandExecution(() => commands.queenStop(), get().fetchStatus);
  },

  install: async () => {
    await withCommandExecution(
      () => commands.queenInstall(null),
      get().fetchStatus,
    );
  },

  rebuild: async () => {
    await withCommandExecution(
      () => commands.queenRebuild(false),
      get().fetchStatus,
    );
  },

  uninstall: async () => {
    await withCommandExecution(
      () => commands.queenUninstall(),
      get().fetchStatus,
    );
  },

  reset: () => {
    set({
      status: null,
      isLoading: false,
      error: null,
    });
  },
}));
