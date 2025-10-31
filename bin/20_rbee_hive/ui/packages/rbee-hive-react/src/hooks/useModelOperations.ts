// Model operations using TanStack Query mutations
// Consistent with useHiveOperations pattern

'use client'

import { useMutation } from '@tanstack/react-query'
import { init, HiveClient, OperationBuilder } from '@rbee/rbee-hive-sdk'

// Initialize WASM (same pattern as useHiveOperations)
let wasmInitialized = false
async function ensureWasmInit() {
  if (!wasmInitialized) {
    init()
    wasmInitialized = true
  }
}

// Create client instance (same pattern as useHiveOperations)
const hiveAddress = window.location.hostname
const hivePort = '7835'
const client = new HiveClient(`http://${hiveAddress}:${hivePort}`, hiveAddress)

export interface LoadModelParams {
  modelId: string
  device?: string
}

export interface UnloadModelParams {
  modelId: string
}

export interface DeleteModelParams {
  modelId: string
}

export interface UseModelOperationsResult {
  loadModel: (params: LoadModelParams) => void
  unloadModel: (params: UnloadModelParams) => void
  deleteModel: (params: DeleteModelParams) => void
  isPending: boolean
  isSuccess: boolean
  isError: boolean
  error: Error | null
  reset: () => void
}

/**
 * Hook for model operations using TanStack Query mutations
 * 
 * Follows the same pattern as useHiveOperations:
 * - Uses HiveClient from rbee-hive-sdk
 * - Uses OperationBuilder for type-safe operations
 * - Streams narration through submitAndStream
 * - Automatic loading states and error handling
 * 
 * @returns Mutation functions and state
 * 
 * @example
 * ```tsx
 * const { loadModel, unloadModel, deleteModel, isPending } = useModelOperations()
 * 
 * <button 
 *   onClick={() => loadModel({ 
 *     modelId: 'meta-llama-Llama-2-7b',
 *     device: 'cuda:0'
 *   })}
 *   disabled={isPending}
 * >
 *   {isPending ? 'Loading...' : 'Load to RAM'}
 * </button>
 * ```
 */
export function useModelOperations(): UseModelOperationsResult {
  const loadMutation = useMutation<any, Error, LoadModelParams>({
    mutationFn: async ({ modelId, device = 'cuda:0' }) => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      const op = OperationBuilder.modelLoad(hiveId, modelId, device)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
          console.log('[Hive] Model load:', line)
        }
      })
      
      return { modelId, device }
    },
    retry: 1,
    retryDelay: 1000,
  })

  const unloadMutation = useMutation<any, Error, UnloadModelParams>({
    mutationFn: async ({ modelId }) => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      const op = OperationBuilder.modelUnload(hiveId, modelId)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
          console.log('[Hive] Model unload:', line)
        }
      })
      
      return { modelId }
    },
    retry: 1,
    retryDelay: 1000,
  })

  const deleteMutation = useMutation<any, Error, DeleteModelParams>({
    mutationFn: async ({ modelId }) => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      const op = OperationBuilder.modelDelete(hiveId, modelId)
      const lines: string[] = []
      
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
          console.log('[Hive] Model delete:', line)
        }
      })
      
      return { modelId }
    },
    retry: 1,
    retryDelay: 1000,
  })

  // Return combined state (use first mutation for shared state)
  return {
    loadModel: loadMutation.mutate,
    unloadModel: unloadMutation.mutate,
    deleteModel: deleteMutation.mutate,
    isPending: loadMutation.isPending || unloadMutation.isPending || deleteMutation.isPending,
    isSuccess: loadMutation.isSuccess || unloadMutation.isSuccess || deleteMutation.isSuccess,
    isError: loadMutation.isError || unloadMutation.isError || deleteMutation.isError,
    error: loadMutation.error || unloadMutation.error || deleteMutation.error,
    reset: () => {
      loadMutation.reset()
      unloadMutation.reset()
      deleteMutation.reset()
    },
  }
}
