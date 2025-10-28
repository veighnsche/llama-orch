// TEAM-338: Zustand store for SSH Hives state
// Replaces SshHivesContainer with idiomatic Zustand pattern
// Imports commandStore internally to manage global isExecuting state
import { create } from "zustand";
import { commands } from "@/generated/bindings";
import type { SshTarget } from "@/generated/bindings";
import { useCommandStore } from "./commandStore";

export interface SshHive {
  host: string;
  host_subtitle?: string;
  hostname: string;
  user: string;
  port: number;
  status: "online" | "offline" | "unknown";
}

interface SshHivesState {
  hives: SshHive[];
  installedHives: string[]; // List of installed hive aliases
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchHives: () => Promise<void>;
  install: (targetId: string) => Promise<void>;
  refresh: () => Promise<void>;
  reset: () => void;
}

// Convert tauri-specta SshTarget to SshHive
function convertToSshHive(target: SshTarget): SshHive {
  return {
    host: target.host,
    host_subtitle: target.host_subtitle ?? undefined,
    hostname: target.hostname,
    user: target.user,
    port: target.port,
    status: target.status,
  };
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
    console.error("Hive command failed:", error);
    throw error;
  } finally {
    setIsExecuting(false);
  }
};

export const useSshHivesStore = create<SshHivesState>((set, get) => ({
  hives: [],
  installedHives: [],
  isLoading: false,
  error: null,

  fetchHives: async () => {
    set({ isLoading: true, error: null });
    const result = await commands.sshList();
    if (result.status === "ok") {
      const hives = result.data.map(convertToSshHive);
      set({ hives, isLoading: false });
    } else {
      set({ isLoading: false });
      // Throw error so ErrorBoundary can catch it
      throw new Error(result.error || "Failed to load SSH hives");
    }
  },

  install: async (targetId: string) => {
    await withCommandExecution(
      async () => {
        await commands.hiveInstall(targetId);
        // Add to installed hives list
        set((state) => ({
          installedHives: [...state.installedHives, targetId],
        }));
      },
      get().fetchHives,
    );
  },

  refresh: async () => {
    await get().fetchHives();
  },

  reset: () => {
    set({
      hives: [],
      installedHives: [],
      isLoading: false,
      error: null,
    });
  },
}));
