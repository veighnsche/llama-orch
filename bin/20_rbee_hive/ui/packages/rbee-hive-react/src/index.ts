// TEAM-353: Migrated to use TanStack Query (no manual state management)
// TEAM-353: Uses WASM SDK (job-based architecture)
import { useQuery } from '@tanstack/react-query'
import { init, HiveClient, OperationBuilder } from '@rbee/rbee-hive-sdk'

// TEAM-353: Initialize WASM module once
let wasmInitialized = false
async function ensureWasmInit() {
  if (!wasmInitialized) {
    init() // TEAM-353: init() is synchronous in WASM
    wasmInitialized = true
  }
}

// TEAM-353: Create client instance
// Get hive address from window.location (Hive UI is served BY the Hive)
const hiveAddress = window.location.hostname
const hivePort = '7835' // TODO: Get from config
const client = new HiveClient(`http://${hiveAddress}:${hivePort}`, hiveAddress)

// TEAM-353: Type definitions (matching backend)
export interface Model {
  id: string
  name: string
  size_bytes: number
}

export interface Worker {
  pid: number
  model: string
  device: string
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
  const {
    data: models,
    isLoading: loading,
    error,
    refetch
  } = useQuery({
    queryKey: ['hive-models'],
    queryFn: async () => {
      await ensureWasmInit()
      const hiveId = client.hiveId // TEAM-353: Get hive_id from client
      const op = OperationBuilder.modelList(hiveId)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
        }
      })
      
      // Parse JSON response from last line
      const lastLine = lines[lines.length - 1]
      return lastLine ? JSON.parse(lastLine) : []
    },
    staleTime: 30000, // Models change less frequently (30 seconds)
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  return {
    models: models || [],
    loading,
    error: error as Error | null,
    refetch
  }
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
  const { 
    data: workers, 
    isLoading: loading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['hive-workers'],
    queryFn: async () => {
      await ensureWasmInit()
      const hiveId = client.hiveId // TEAM-353: Get hive_id from client
      const op = OperationBuilder.workerList(hiveId)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
        }
      })
      
      // Parse JSON response from last line
      const lastLine = lines[lines.length - 1]
      return lastLine ? JSON.parse(lastLine) : []
    },
    staleTime: 5000, // Workers change frequently (5 seconds)
    refetchInterval: 2000, // Auto-refetch every 2 seconds
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  })

  return { 
    workers: workers || [], 
    loading,
    error: error as Error | null,
    refetch
  }
}

// Export operation hooks
export { useHiveOperations } from './hooks/useHiveOperations'
export type { UseHiveOperationsResult } from './hooks/useHiveOperations'
