export type { ModelInfo, HFModel } from '@rbee/rbee-hive-sdk';
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
export { useHiveOperations, WORKER_TYPE_OPTIONS, WORKER_TYPES } from './hooks/useHiveOperations';
export type { UseHiveOperationsResult, WorkerType, WorkerTypeOption, SpawnWorkerParams } from './hooks/useHiveOperations';
export { useModelOperations } from './hooks/useModelOperations';
export type { UseModelOperationsResult, LoadModelParams, UnloadModelParams, DeleteModelParams } from './hooks/useModelOperations';
