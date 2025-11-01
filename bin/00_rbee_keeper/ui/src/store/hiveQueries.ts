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
  buildMode: string | null; // TEAM-379: "debug", "release", or null
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
    buildMode: null, // TEAM-379: Not available from SSH list, only from status check
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
// TEAM-378: RULE ZERO - Backend now returns complete info (hostname, user, port)
async function fetchHiveStatus(hiveId: string, sshHives?: SshHive[]): Promise<SshHive> {
  const result = await commands.hiveStatus(hiveId);
  if (result.status === 'ok') {
    const { is_running, is_installed, build_mode, hostname, user, port } = result.data;
    
    // TEAM-378: Get host_subtitle from SSH list (only field not in DaemonStatus)
    const sshConfig = sshHives?.find(h => h.host === hiveId);
    
    const status = is_running ? 'online' : 'offline';
    
    return {
      host: hiveId,
      hostname, // TEAM-378: Now comes directly from backend!
      user,     // TEAM-378: Now comes directly from backend!
      port,     // TEAM-378: Now comes directly from backend!
      status: status as 'online' | 'offline',
      isInstalled: is_installed,
      buildMode: build_mode,
      host_subtitle: sshConfig?.host_subtitle,
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

// Hook: Get list of installed hives from backend (TEAM-370)
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
    staleTime: 30 * 1000, // 30 seconds - don't check too often
    gcTime: 5 * 60 * 1000, // 5 minutes cache
  });
}

// Hook: Fetch individual hive status
// TEAM-378: Merges SSH config (hostname/IP) with daemon status (running/installed)
export function useHive(hiveId: string) {
  const queryClient = useQueryClient();
  
  return useQuery({
    queryKey: hiveKeys.detail(hiveId),
    queryFn: () => {
      // TEAM-378: Get SSH config from cache to merge hostname/IP
      const sshHives = queryClient.getQueryData<SshHive[]>(hiveKeys.list());
      return fetchHiveStatus(hiveId, sshHives);
    },
    staleTime: 0, // TEAM-379: Always fetch fresh - cards only show for installed hives
    gcTime: 30 * 1000, // 30 seconds - keep in memory briefly for quick navigation
    // TEAM-368: Don't use initialData from SSH list - it doesn't have daemon status!
  });
}

// Hook: Hive mutations (install, start, stop, etc.)
export function useHiveActions() {
  const queryClient = useQueryClient();
  
  const install = useMutation({
    mutationFn: async ({ targetId, buildMode = "dev" }: { targetId: string; buildMode?: "dev" | "prod" }) => {
      await withCommandExecution(
        () => commands.hiveInstall(targetId, buildMode === "prod" ? "release" : null),
        async () => {},
        `Hive install (${buildMode})`,
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
    install: async (targetId: string, buildMode: "dev" | "prod" = "dev") => { 
      await install.mutateAsync({ targetId, buildMode }); 
    },
    start: async (id: string) => { await start.mutateAsync(id); },
    stop: async (id: string) => { await stop.mutateAsync(id); },
    rebuild: async (id: string) => { await rebuild.mutateAsync(id); },
    uninstall: async (id: string) => { await uninstall.mutateAsync(id); },
  };
}

