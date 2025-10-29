// TEAM-367: React Query hooks for HIVE data fetching ONLY
// Queen is in queenQueries.ts (separate file!)
// NO MORE useEffect - proper data fetching with TanStack Query
// NO MORE Zustand persistence - get actual state from backend!

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { commands } from '@/generated/bindings';
import { withCommandExecution } from './commandUtils';
import type { SshTarget } from '@/generated/bindings';

export interface SshHive {
  host: string;
  host_subtitle?: string;
  hostname: string;
  user: string;
  port: number;
  status: "online" | "offline" | "unknown";
  isInstalled?: boolean;
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

// Query keys for React Query
export const hiveKeys = {
  all: ['hives'] as const,
  lists: () => [...hiveKeys.all, 'list'] as const,
  list: () => [...hiveKeys.lists()] as const,
  details: () => [...hiveKeys.all, 'detail'] as const,
  detail: (id: string) => [...hiveKeys.details(), id] as const,
};


// Fetch SSH hives list
async function fetchSshHivesList(): Promise<SshHive[]> {
  const result = await commands.sshList();
  if (result.status === 'ok') {
    return result.data.map(convertToSshHive);
  }
  throw new Error(result.error || 'Failed to load SSH hives');
}

// Fetch individual hive status
async function fetchHiveStatus(hiveId: string): Promise<SshHive> {
  const result = await commands.hiveStatus(hiveId);
  if (result.status === 'ok') {
    const { is_running, is_installed } = result.data;
    
    // Try to get hive details from the list query cache
    const status = is_running ? 'online' : 'offline';
    
    return {
      host: hiveId,
      hostname: hiveId,
      user: 'unknown',
      port: 22,
      status: status as 'online' | 'offline',
      isInstalled: is_installed,
    };
  }
  throw new Error(result.error || `Failed to fetch status for hive ${hiveId}`);
}

// Hook: Fetch SSH hives list
export function useSshHives() {
  return useQuery({
    queryKey: hiveKeys.list(),
    queryFn: fetchSshHivesList,
    staleTime: 5 * 60 * 1000, // 5 minutes - don't refetch unless stale
    gcTime: 10 * 60 * 1000, // 10 minutes cache
  });
}

// Hook: Get list of installed hives from backend (TEAM-367)
export function useInstalledHives() {
  return useQuery({
    queryKey: [...hiveKeys.all, 'installed'] as const,
    queryFn: async () => {
      const result = await commands.getInstalledHives();
      if (result.status === 'ok') {
        return result.data;
      }
      throw new Error(result.error || 'Failed to get installed hives');
    },
    staleTime: 5 * 1000, // 5 seconds
    gcTime: 60 * 1000, // 1 minute cache
  });
}

// Hook: Fetch individual hive status
export function useHive(hiveId: string) {
  return useQuery({
    queryKey: hiveKeys.detail(hiveId),
    queryFn: () => fetchHiveStatus(hiveId),
    staleTime: 5 * 1000, // 5 seconds
    gcTime: 60 * 1000, // 1 minute cache
    // TEAM-368: Don't use initialData from SSH list - it doesn't have daemon status!
  });
}

// Hook: Hive mutations (install, start, stop, etc.)
export function useHiveActions() {
  const queryClient = useQueryClient();
  
  const install = useMutation({
    mutationFn: async (targetId: string) => {
      await withCommandExecution(
        () => commands.hiveInstall(targetId),
        async () => {},
        'Hive install',
      );
      return targetId;
    },
    onSuccess: () => {
      // TEAM-368: Invalidate ALL queries - hives list AND installed hives list
      queryClient.invalidateQueries({ queryKey: hiveKeys.all });
      queryClient.invalidateQueries({ queryKey: ['hives', 'installed'] });
    },
  });
  
  const start = useMutation({
    mutationFn: async (hiveId: string) => {
      await withCommandExecution(
        () => commands.hiveStart(hiveId),
        async () => {},
        'Hive start',
      );
    },
    onSuccess: (_, hiveId) => {
      queryClient.invalidateQueries({ queryKey: hiveKeys.detail(hiveId) });
    },
  });
  
  const stop = useMutation({
    mutationFn: async (hiveId: string) => {
      await withCommandExecution(
        () => commands.hiveStop(hiveId),
        async () => {},
        'Hive stop',
      );
    },
    onSuccess: (_, hiveId) => {
      queryClient.invalidateQueries({ queryKey: hiveKeys.detail(hiveId) });
    },
  });
  
  const rebuild = useMutation({
    mutationFn: async (hiveId: string) => {
      await withCommandExecution(
        () => commands.hiveRebuild(hiveId),
        async () => {},
        'Hive rebuild',
      );
    },
    onSuccess: (_, hiveId) => {
      queryClient.invalidateQueries({ queryKey: hiveKeys.detail(hiveId) });
    },
  });
  
  const uninstall = useMutation({
    mutationFn: async (hiveId: string) => {
      await withCommandExecution(
        () => commands.hiveUninstall(hiveId),
        async () => {},
        'Hive uninstall',
      );
      return hiveId;
    },
    onSuccess: () => {
      // TEAM-368: Invalidate ALL queries - hives list AND installed hives list
      queryClient.invalidateQueries({ queryKey: hiveKeys.all });
      queryClient.invalidateQueries({ queryKey: ['hives', 'installed'] });
    },
  });
  
  return {
    install: async (id: string) => { await install.mutateAsync(id); },
    start: async (id: string) => { await start.mutateAsync(id); },
    stop: async (id: string) => { await stop.mutateAsync(id); },
    rebuild: async (id: string) => { await rebuild.mutateAsync(id); },
    uninstall: async (id: string) => { await uninstall.mutateAsync(id); },
  };
}

