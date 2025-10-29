// TEAM-338: Zustand store for SSH Hives state
// TEAM-351: Rewritten with query-based pattern (no promise caching)
import { create } from "zustand";
import { persist } from "zustand/middleware";
import { immer } from "zustand/middleware/immer";
import type { SshTarget } from "@/generated/bindings";
import { commands } from "@/generated/bindings";
import { withCommandExecution } from "./commandUtils";
import { useEffect } from "react";

export interface SshHive {
  host: string;
  host_subtitle?: string;
  hostname: string;
  user: string;
  port: number;
  status: "online" | "offline" | "unknown";
  isInstalled?: boolean;
}

// TEAM-351: Query state for a single hive
interface HiveQuery {
  data: SshHive | null;
  isLoading: boolean;
  error: string | null;
  lastFetch: number; // Timestamp for stale detection
}

// TEAM-351: Query state for SSH hives list
interface HivesListQuery {
  data: SshHive[];
  isLoading: boolean;
  error: string | null;
  lastFetch: number;
}

interface SshHivesState {
  // TEAM-355: Query cache - Record instead of Map (no enableMapSet needed)
  queries: Record<string, HiveQuery>;
  // TEAM-351: Separate query for SSH hives list
  hivesListQuery: HivesListQuery;
  installedHives: string[]; // Persisted list

  // TEAM-351: Query actions
  fetchHive: (hiveId: string, force?: boolean) => Promise<void>;
  fetchHivesList: (force?: boolean) => Promise<void>;
  invalidate: (hiveId: string) => Promise<void>;
  invalidateAll: () => Promise<void>;
  
  // TEAM-351: Mutation actions
  install: (targetId: string) => Promise<void>;
  start: (hiveId: string) => Promise<void>;
  stop: (hiveId: string) => Promise<void>;
  uninstall: (hiveId: string) => Promise<void>;
  rebuild: (hiveId: string) => Promise<void>;
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
      queries: {},
      hivesListQuery: {
        data: [],
        isLoading: false,
        error: null,
        lastFetch: 0,
      },
      installedHives: [],

      // TEAM-351: Fetch SSH hives list with deduplication
      fetchHivesList: async (force = false) => {
        const now = Date.now();
        const existing = get().hivesListQuery;
        
        // Skip if fresh (< 5s old) and not forced
        if (!force && !existing.isLoading && (now - existing.lastFetch < 5000)) {
          return;
        }
        
        // Skip if already loading
        if (existing.isLoading) {
          return;
        }
        
        set((state) => {
          state.hivesListQuery = {
            data: existing.data,
            isLoading: true,
            error: null,
            lastFetch: now,
          };
        });
        
        try {
          const result = await commands.sshList();
          if (result.status === "ok") {
            const hives = result.data.map(convertToSshHive);
            set((state) => {
              state.hivesListQuery = {
                data: hives,
                isLoading: false,
                error: null,
                lastFetch: now,
              };
            });
          } else {
            throw new Error(result.error || "Failed to load SSH hives");
          }
        } catch (error) {
          set((state) => {
            state.hivesListQuery = {
              data: existing.data,
              isLoading: false,
              error: error instanceof Error ? error.message : "Failed to load SSH hives",
              lastFetch: now,
            };
          });
        }
      },

      // TEAM-351: Fetch individual hive status with deduplication
      fetchHive: async (hiveId: string, force = false) => {
        const now = Date.now();
        const existing = get().queries[hiveId];
        
        // Skip if fresh (< 5s old) and not forced
        if (!force && existing && !existing.isLoading && (now - existing.lastFetch < 5000)) {
          return;
        }
        
        // Skip if already loading
        if (existing?.isLoading) {
          return;
        }
        
        set((state) => {
          state.queries[hiveId] = {
            data: existing?.data ?? null,
            isLoading: true,
            error: null,
            lastFetch: now,
          };
        });
        
        try {
          const result = await commands.hiveStatus(hiveId);
          if (result.status === "ok") {
            const { is_running, is_installed } = result.data;
            
            // Get hive details from list
            const hiveDetails = get().hivesListQuery.data.find((h) => h.host === hiveId);
            
            const hiveData: SshHive = hiveDetails ? {
              ...hiveDetails,
              status: is_running ? "online" : "offline",
              isInstalled: is_installed,
            } : {
              host: hiveId,
              hostname: hiveId,
              user: "unknown",
              port: 22,
              status: is_running ? "online" : "offline",
              isInstalled: is_installed,
            };
            
            set((state) => {
              state.queries[hiveId] = {
                data: hiveData,
                isLoading: false,
                error: null,
                lastFetch: now,
              };
              
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
            throw new Error(result.error || `Failed to fetch status for hive ${hiveId}`);
          }
        } catch (error) {
          set((state) => {
            state.queries[hiveId] = {
              data: existing?.data ?? null,
              isLoading: false,
              error: error instanceof Error ? error.message : "Failed",
              lastFetch: now,
            };
          });
        }
      },
      
      // TEAM-351: Invalidate single hive query
      invalidate: (hiveId: string) => {
        set((state) => {
          delete state.queries[hiveId];
        });
        // Trigger refetch
        return get().fetchHive(hiveId, true);
      },
      
      // TEAM-351: Invalidate all queries
      invalidateAll: async () => {
        set((state) => {
          state.queries = {};
          state.hivesListQuery.lastFetch = 0;
        });
        // Trigger refetch of list
        await get().fetchHivesList(true);
      },

      install: async (targetId: string) => {
        await withCommandExecution(
          async () => {
            await commands.hiveInstall(targetId);
            set((state) => {
              if (!state.installedHives.includes(targetId)) {
                state.installedHives.push(targetId);
              }
            });
          },
          () => get().invalidateAll(),
          "Hive install",
        );
      },

      start: async (hiveId: string) => {
        await withCommandExecution(
          () => commands.hiveStart(hiveId),
          () => get().invalidate(hiveId),
          "Hive start",
        );
      },

      stop: async (hiveId: string) => {
        await withCommandExecution(
          () => commands.hiveStop(hiveId),
          () => get().invalidate(hiveId),
          "Hive stop",
        );
      },

      uninstall: async (hiveId: string) => {
        await withCommandExecution(
          async () => {
            await commands.hiveUninstall(hiveId);
            set((state) => {
              state.installedHives = state.installedHives.filter(
                (id) => id !== hiveId,
              );
            });
          },
          () => get().invalidateAll(),
          "Hive uninstall",
        );
      },

      rebuild: async (hiveId: string) => {
        await withCommandExecution(
          () => commands.hiveRebuild(hiveId),
          () => get().invalidate(hiveId),
          "Hive rebuild",
        );
      },

      reset: () => {
        set((state) => {
          state.queries = {};
          state.hivesListQuery = {
            data: [],
            isLoading: false,
            error: null,
            lastFetch: 0,
          };
          state.installedHives = [];
        });
      },
    })),
    {
      name: "hive-store",
      partialize: (state) => ({
        installedHives: state.installedHives,
      }),
    },
  ),
);

// TEAM-351: Query hooks for components
// TEAM-355: Fixed - extract fetchHive to avoid store in deps (prevents infinite loop)
export function useHive(hiveId: string) {
  const store = useSshHivesStore();
  const query = store.queries[hiveId];
  const fetchHive = store.fetchHive;
  
  useEffect(() => {
    fetchHive(hiveId);
  }, [hiveId, fetchHive]);
  
  return {
    hive: query?.data ?? null,
    isLoading: query?.isLoading ?? true,
    error: query?.error ?? null,
    refetch: () => store.fetchHive(hiveId, true),
  };
}

export function useSshHives() {
  const store = useSshHivesStore();
  const query = store.hivesListQuery;
  const fetchHivesList = store.fetchHivesList;
  
  useEffect(() => {
    fetchHivesList();
  }, [fetchHivesList]);
  
  return {
    hives: query.data,
    isLoading: query.isLoading,
    error: query.error,
    refetch: () => store.fetchHivesList(true),
  };
}

// TEAM-351: Action hooks for mutations
export function useHiveActions() {
  const store = useSshHivesStore();
  
  return {
    start: store.start,
    stop: store.stop,
    install: store.install,
    uninstall: store.uninstall,
    rebuild: store.rebuild,
  };
}
