// TEAM-353: Migrated to use TanStack Query (no manual state management)
// TEAM-353: Uses WASM SDK (job-based architecture)
// TEAM-377: React Query removed - use @rbee/ui/providers instead
// TEAM-377: React Query REMOVED
// DO NOT re-export React Query - import from @rbee/ui/providers:
//   import { QueryProvider } from '@rbee/ui/providers'
// This ensures consistent configuration across all apps
import { useQuery } from '@tanstack/react-query';
// TEAM-381: Lazy WASM SDK initialization (avoid module load issues)
let sdkModule = null;
async function ensureWasmInit() {
    if (!sdkModule) {
        sdkModule = await import('@rbee/rbee-hive-sdk');
        sdkModule.init(); // Initialize WASM module
    }
    return sdkModule;
}
// TEAM-381: Lazy client initialization (avoid window access at module load time)
let client = null;
async function getClient() {
    if (!client) {
        const sdk = await ensureWasmInit();
        // Get hive address from window.location (Hive UI is served BY the Hive)
        const hiveAddress = window.location.hostname;
        const hivePort = '7835'; // TODO: Get from config
        client = new sdk.HiveClient(`http://${hiveAddress}:${hivePort}`, hiveAddress);
    }
    return client;
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
export function useModels() {
    const { data: models, isLoading: loading, error, refetch } = useQuery({
        queryKey: ['hive-models'],
        queryFn: async () => {
            const sdk = await ensureWasmInit();
            const client = await getClient(); // TEAM-381: Lazy client initialization
            const hiveId = client.hiveId; // TEAM-353: Get hive_id from client
            const op = sdk.OperationBuilder.modelList(hiveId);
            const lines = [];
            // TEAM-381: Add timeout to prevent infinite hanging if backend doesn't respond
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Request timeout: Backend did not respond within 10 seconds. Is rbee-hive running?')), 10000);
            });
            const streamPromise = client.submitAndStream(op, (line) => {
                if (line !== '[DONE]') {
                    lines.push(line);
                }
            });
            await Promise.race([streamPromise, timeoutPromise]);
            // Find the JSON line (starts with '[' or '{')
            // Backend emits narration lines first, then JSON on last line
            const jsonLine = lines.reverse().find(line => {
                const trimmed = line.trim();
                return trimmed.startsWith('[') || trimmed.startsWith('{');
            });
            return jsonLine ? JSON.parse(jsonLine) : [];
        },
        staleTime: 30000, // Models change less frequently (30 seconds)
        retry: 2, // TEAM-381: Reduced from 3 to fail faster
        retryDelay: 1000, // TEAM-381: Fixed 1s delay instead of exponential
    });
    return {
        models: models || [],
        loading,
        error: error,
        refetch
    };
}
/**
 * Hook for fetching worker list from Hive
 *
 * TEAM-353: Migrated to TanStack Query + WASM SDK (job-based architecture)
 * - Automatic polling (refetchInterval)
 * - Automatic caching
 * - Automatic error handling
 * - Automatic retry
 */
export function useWorkers() {
    const { data: workers, isLoading: loading, error, refetch } = useQuery({
        queryKey: ['hive-workers'],
        queryFn: async () => {
            const sdk = await ensureWasmInit();
            const client = await getClient(); // TEAM-381: Lazy client initialization
            const hiveId = client.hiveId; // TEAM-353: Get hive_id from client
            const op = sdk.OperationBuilder.workerList(hiveId);
            const lines = [];
            await client.submitAndStream(op, (line) => {
                if (line !== '[DONE]') {
                    lines.push(line);
                }
            });
            // Parse JSON response from last line
            const lastLine = lines[lines.length - 1];
            return lastLine ? JSON.parse(lastLine) : [];
        },
        staleTime: 5000, // Workers change frequently (5 seconds)
        refetchInterval: 2000, // Auto-refetch every 2 seconds
        retry: 3,
        retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    });
    return {
        workers: workers || [],
        loading,
        error: error,
        refetch
    };
}
// Export operation hooks
export { useHiveOperations, WORKER_TYPE_OPTIONS, WORKER_TYPES } from './hooks/useHiveOperations';
export { useModelOperations } from './hooks/useModelOperations';
// TEAM-378: Worker operations (install + spawn)
export { useWorkerOperations } from './hooks/useWorkerOperations';
