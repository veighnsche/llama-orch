// TEAM-338: Zustand store for Queen service state and commands
import { create } from "zustand";
import { commands } from "@/generated/bindings";
import { withCommandExecution } from "./commandUtils";

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
    await withCommandExecution(
      () => commands.queenStart(),
      get().fetchStatus,
      "Queen start",
    );
  },

  stop: async () => {
    await withCommandExecution(
      () => commands.queenStop(),
      get().fetchStatus,
      "Queen stop",
    );
  },

  install: async () => {
    await withCommandExecution(
      () => commands.queenInstall(null),
      get().fetchStatus,
      "Queen install",
    );
  },

  rebuild: async () => {
    await withCommandExecution(
      () => commands.queenRebuild(false),
      get().fetchStatus,
      "Queen rebuild",
    );
  },

  uninstall: async () => {
    await withCommandExecution(
      () => commands.queenUninstall(),
      get().fetchStatus,
      "Queen uninstall",
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
