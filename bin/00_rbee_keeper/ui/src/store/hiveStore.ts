// TEAM-338: Zustand store for SSH Hives state
// Replaces SshHivesContainer with idiomatic Zustand pattern
// TEAM-338: Added persist and immer middleware
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { commands } from "@/generated/bindings";
import type { SshTarget } from "@/generated/bindings";
import { withCommandExecution } from "./commandUtils";

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
  start: (hiveId: string) => Promise<void>;
  stop: (hiveId: string) => Promise<void>;
  uninstall: (hiveId: string) => Promise<void>;
  refreshCapabilities: (hiveId: string) => Promise<void>;
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

export const useSshHivesStore = create<SshHivesState>()(
  persist(
    immer((set, get) => ({
      hives: [],
      installedHives: [],
      isLoading: false,
      error: null,

      fetchHives: async () => {
        set((state) => {
          state.isLoading = true;
          state.error = null;
        });
        const result = await commands.sshList();
        if (result.status === "ok") {
          const hives = result.data.map(convertToSshHive);
          set((state) => {
            state.hives = hives;
            state.isLoading = false;
          });
        } else {
          set((state) => {
            state.isLoading = false;
          });
          // Throw error so ErrorBoundary can catch it
          throw new Error(result.error || "Failed to load SSH hives");
        }
      },

      install: async (targetId: string) => {
        await withCommandExecution(
          async () => {
            await commands.hiveInstall(targetId);
            // Add to installed hives list
            set((state) => {
              state.installedHives.push(targetId);
            });
          },
          get().fetchHives,
          "Hive install",
        );
      },

      start: async (hiveId: string) => {
        await withCommandExecution(
          () => commands.hiveStart(hiveId),
          get().fetchHives,
          "Hive start",
        );
      },

      stop: async (hiveId: string) => {
        await withCommandExecution(
          () => commands.hiveStop(hiveId),
          get().fetchHives,
          "Hive stop",
        );
      },

      uninstall: async (hiveId: string) => {
        await withCommandExecution(
          async () => {
            await commands.hiveUninstall(hiveId);
            // Remove from installed hives list
            set((state) => {
              state.installedHives = state.installedHives.filter(
                (id) => id !== hiveId,
              );
            });
          },
          get().fetchHives,
          "Hive uninstall",
        );
      },

      refreshCapabilities: async (hiveId: string) => {
        await withCommandExecution(
          () => commands.hiveRefreshCapabilities(hiveId),
          get().fetchHives,
          "Hive refresh capabilities",
        );
      },

      refresh: async () => {
        await get().fetchHives();
      },

      reset: () => {
        set((state) => {
          state.hives = [];
          state.installedHives = [];
          state.isLoading = false;
          state.error = null;
        });
      },
    })),
    {
      name: "hive-store",
      partialize: (state) => ({
        hives: state.hives,
        installedHives: state.installedHives,
      }),
    },
  ),
);
