export { QueryClient, QueryClientProvider } from '@tanstack/react-query';
export type { QueryClientConfig } from '@tanstack/react-query';
export interface Model {
    id: string;
    name: string;
    size_bytes: number;
}
export interface Worker {
    pid: number;
    model: string;
    device: string;
}
/**
 * Hook for fetching model list from Hive
 *
 * TEAM-353: Migrated to TanStack Query + WASM SDK (job-based architecture)
 * - Automatic caching
 * - Automatic error handling
 * - Automatic retry
 * - Stale data management
 */
export declare function useModels(): {
    models: any;
    loading: boolean;
    error: Error | null;
    refetch: (options?: import("@tanstack/react-query").RefetchOptions) => Promise<import("@tanstack/react-query").QueryObserverResult<any, Error>>;
};
/**
 * Hook for fetching worker list from Hive
 *
 * TEAM-353: Migrated to TanStack Query + WASM SDK (job-based architecture)
 * - Automatic polling (refetchInterval)
 * - Automatic caching
 * - Automatic error handling
 * - Automatic retry
 */
export declare function useWorkers(): {
    workers: any;
    loading: boolean;
    error: Error | null;
    refetch: (options?: import("@tanstack/react-query").RefetchOptions) => Promise<import("@tanstack/react-query").QueryObserverResult<any, Error>>;
};
export { useHiveOperations } from './hooks/useHiveOperations';
export type { UseHiveOperationsResult } from './hooks/useHiveOperations';
