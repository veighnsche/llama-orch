// TEAM-377: Hive operations using TanStack Query mutations
// Consistent with useModels/useWorkers pattern

'use client'

import { useMutation } from '@tanstack/react-query'
import { init, HiveClient, OperationBuilder } from '@rbee/rbee-hive-sdk'

// TEAM-377: Initialize WASM (same pattern as index.ts)
let wasmInitialized = false
async function ensureWasmInit() {
  if (!wasmInitialized) {
    init()
    wasmInitialized = true
  }
}

// TEAM-377: Create client instance (same pattern as index.ts)
const hiveAddress = window.location.hostname
const hivePort = '7835'
const client = new HiveClient(`http://${hiveAddress}:${hivePort}`, hiveAddress)

// TEAM-377: Worker types (matches Rust enum in worker-catalog/src/types.rs)
export type WorkerType = 'cpu' | 'cuda' | 'metal'

export const WORKER_TYPES: readonly WorkerType[] = ['cpu', 'cuda', 'metal'] as const

export interface WorkerTypeOption {
  value: WorkerType
  label: string
  description: string
}

export const WORKER_TYPE_OPTIONS: readonly WorkerTypeOption[] = [
  { value: 'cpu', label: 'CPU', description: 'CPU-based LLM worker' },
  { value: 'cuda', label: 'CUDA', description: 'NVIDIA GPU-based LLM worker' },
  { value: 'metal', label: 'Metal', description: 'Apple Metal GPU-based LLM worker (macOS)' },
] as const

export interface SpawnWorkerParams {
  modelId: string
  workerType?: WorkerType
  deviceId?: number
}

export interface UseHiveOperationsResult {
  spawnWorker: (params: SpawnWorkerParams) => void
  installWorker: (workerId: string) => void
  isPending: boolean
  isSuccess: boolean
  isError: boolean
  error: Error | null
  reset: () => void
}

/**
 * Hook for Hive operations using TanStack Query mutations
 * 
 * TEAM-377: Refactored to use useMutation (consistent with useModels/useWorkers)
 * - Automatic loading states
 * - Automatic error handling
 * - Automatic retry logic
 * - Consistent with other hooks in this package
 * 
 * @returns Mutation functions and state
 * 
 * @example
 * ```tsx
 * const { spawnWorker, isPending, error } = useHiveOperations()
 * 
 * <button 
 *   onClick={() => spawnWorker({ 
 *     modelId: 'llama-3.2-1b',
 *     workerType: 'cuda',
 *     deviceId: 0
 *   })}
 *   disabled={isPending}
 * >
 *   {isPending ? 'Spawning...' : 'Spawn Worker'}
 * </button>
 * ```
 */
export function useHiveOperations(): UseHiveOperationsResult {
  const spawnMutation = useMutation<any, Error, SpawnWorkerParams>({
    mutationFn: async ({ modelId, workerType = 'cuda', deviceId = 0 }) => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      // TEAM-377: workerSpawn(hive_id, model, worker_type, device_id)
      // Worker types: 'cpu', 'cuda', 'metal' (matches Rust WorkerType enum)
      const op = OperationBuilder.workerSpawn(hiveId, modelId, workerType, deviceId)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
          console.log('[Hive] Worker spawn:', line)
        }
      })
      
      // Parse response from last line
      const lastLine = lines[lines.length - 1]
      return lastLine ? JSON.parse(lastLine) : null
    },
    retry: 1, // Only retry once for mutations
    retryDelay: 1000,
  })

  // TEAM-378: Worker installation mutation
  const installMutation = useMutation<any, Error, string>({
    mutationFn: async (workerId: string) => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      // TEAM-378: workerInstall(hive_id, worker_id)
      const op = OperationBuilder.workerInstall(hiveId, workerId)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
          console.log('[Hive] Worker install:', line)
        }
      })
      
      return { success: true, workerId }
    },
    retry: 1,
    retryDelay: 1000,
  })

  return {
    spawnWorker: spawnMutation.mutate,
    installWorker: installMutation.mutate,
    isPending: spawnMutation.isPending || installMutation.isPending,
    isSuccess: spawnMutation.isSuccess || installMutation.isSuccess,
    isError: spawnMutation.isError || installMutation.isError,
    error: spawnMutation.error || installMutation.error,
    reset: () => {
      spawnMutation.reset()
      installMutation.reset()
    },
  }
}
