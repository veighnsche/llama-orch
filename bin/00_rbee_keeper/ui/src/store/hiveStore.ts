// TEAM-338: Zustand store for SSH Hives state
// Replaces SshHivesContainer with idiomatic Zustand pattern
// TEAM-338: Added persist and immer middleware
// TEAM-341: Added enableMapSet for Map support in Immer
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import { enableMapSet } from "immer";
import type { SshTarget } from "@/generated/bindings";
import { commands } from "@/generated/bindings";
import { withCommandExecution } from "./commandUtils";

// TEAM-341: Enable Map/Set support in Immer (required for _fetchPromises)
enableMapSet();

export interface SshHive {
  host: string;
  host_subtitle?: string;
  hostname: string;
  user: string;
  port: number;
  status: "online" | "offline" | "unknown";
  isInstalled?: boolean; // TEAM-338: Track installation status
}

interface SshHivesState {
  hives: SshHive[];
  installedHives: string[]; // List of installed hive aliases
  isLoading: boolean;
  error: string | null;
  _fetchHivesPromise: Promise<void> | null; // TEAM-341: Cache in-flight fetchHives to prevent race conditions
  _fetchPromises: Map<string, Promise<void>>; // TEAM-341: Cache in-flight fetchHiveStatus per hive

  // Actions
  fetchHives: () => Promise<void>;
  fetchHiveStatus: (hiveId: string) => Promise<void>; // TEAM-338: Fetch individual hive status
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
      _fetchHivesPromise: null,
      _fetchPromises: new Map(),

      fetchHives: async () => {
        // TEAM-341: Return existing promise if fetch is already in progress
        const existing = get()._fetchHivesPromise;
        if (existing) return existing;

        const promise = (async () => {
          set((state) => {
            state.isLoading = true;
            state.error = null;
          });
          try {
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
          } finally {
            // TEAM-341: Clear promise cache after completion
            set((state) => {
              state._fetchHivesPromise = null;
            });
          }
        })();

        // TEAM-342: Defer promise caching to avoid render-during-render
        queueMicrotask(() => {
          set((state) => {
            state._fetchHivesPromise = promise;
          });
        });
        return promise;
      },

      // TEAM-338: Fetch individual hive status (running + installed)
      // TEAM-341: Deduplicated to prevent race conditions when multiple HiveCards mount
      fetchHiveStatus: async (hiveId: string) => {
        // TEAM-341: Return existing promise if fetch is already in progress for this hive
        const existing = get()._fetchPromises.get(hiveId);
        if (existing) return existing;

        const promise = (async () => {
          try {
            const result = await commands.hiveStatus(hiveId);
            if (result.status === "ok") {
              const { is_running, is_installed } = result.data;
              set((state) => {
                const hive = state.hives.find((h) => h.host === hiveId);
                if (hive) {
                  hive.status = is_running ? "online" : "offline";
                  hive.isInstalled = is_installed;
                }
                // Update installedHives list
                if (is_installed && !state.installedHives.includes(hiveId)) {
                  state.installedHives.push(hiveId);
                } else if (!is_installed) {
                  state.installedHives = state.installedHives.filter(
                    (id) => id !== hiveId,
                  );
                }
              });
            } else {
              throw new Error(
                result.error || `Failed to fetch status for hive ${hiveId}`,
              );
            }
          } finally {
            // TEAM-341: Clear promise cache after completion
            set((state) => {
              state._fetchPromises.delete(hiveId);
            });
          }
        })();

        // TEAM-342: Defer promise caching to avoid render-during-render
        queueMicrotask(() => {
          set((state) => {
            state._fetchPromises.set(hiveId, promise);
          });
        });
        return promise;
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
          () => get().fetchHiveStatus(hiveId), // TEAM-339: Fetch individual hive status after start
          "Hive start",
        );
      },

      stop: async (hiveId: string) => {
        await withCommandExecution(
          () => commands.hiveStop(hiveId),
          () => get().fetchHiveStatus(hiveId), // TEAM-339: Fetch individual hive status after stop
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
          state._fetchHivesPromise = null;
          state._fetchPromises.clear();
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
