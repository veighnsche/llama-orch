// TEAM-365: Queen React Query hooks (separate file - not mixed with hives!)
// NO MORE useEffect - proper data fetching with TanStack Query

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { commands } from '@/generated/bindings';
import { withCommandExecution } from './commandUtils';

export interface QueenStatus {
  isRunning: boolean;
  isInstalled: boolean;
}

// Query keys for React Query
export const queenKeys = {
  all: ['queen'] as const,
  status: () => [...queenKeys.all, 'status'] as const,
};

// Fetch Queen status
async function fetchQueenStatus(): Promise<QueenStatus> {
  const result = await commands.queenStatus();
  if (result.status === 'ok') {
    return {
      isRunning: result.data.is_running,
      isInstalled: result.data.is_installed,
    };
  }
  throw new Error(result.error || 'Failed to fetch Queen status');
}

// Hook: Fetch Queen status
export function useQueen() {
  return useQuery({
    queryKey: queenKeys.status(),
    queryFn: fetchQueenStatus,
    staleTime: 10 * 1000, // 10 seconds
    gcTime: 2 * 60 * 1000, // 2 minutes cache
  });
}

// Hook: Queen mutations
export function useQueenActions() {
  const queryClient = useQueryClient();
  
  const start = useMutation({
    mutationFn: async () => {
      await withCommandExecution(
        () => commands.queenStart(),
        async () => {},
        'Queen start',
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queenKeys.all });
    },
  });
  
  const stop = useMutation({
    mutationFn: async () => {
      await withCommandExecution(
        () => commands.queenStop(),
        async () => {},
        'Queen stop',
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queenKeys.all });
    },
  });
  
  const install = useMutation({
    mutationFn: async () => {
      await withCommandExecution(
        () => commands.queenInstall(null),
        async () => {},
        'Queen install',
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queenKeys.all });
    },
  });
  
  const rebuild = useMutation({
    mutationFn: async () => {
      await withCommandExecution(
        () => commands.queenRebuild(false),
        async () => {},
        'Queen rebuild',
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queenKeys.all });
    },
  });
  
  const uninstall = useMutation({
    mutationFn: async () => {
      await withCommandExecution(
        () => commands.queenUninstall(),
        async () => {},
        'Queen uninstall',
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queenKeys.all });
    },
  });
  
  return {
    start: async () => { await start.mutateAsync(); },
    stop: async () => { await stop.mutateAsync(); },
    install: async () => { await install.mutateAsync(); },
    installProd: async () => { 
      await withCommandExecution(
        () => commands.queenInstall("release"),
        async () => {},
        'Queen install (production)',
      );
      queryClient.invalidateQueries({ queryKey: queenKeys.all });
    },
    rebuild: async () => { await rebuild.mutateAsync(); },
    uninstall: async () => { await uninstall.mutateAsync(); },
  };
}
