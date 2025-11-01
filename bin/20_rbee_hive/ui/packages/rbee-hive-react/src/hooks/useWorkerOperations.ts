// TEAM-378: Worker operations using TanStack Query mutations
// Handles worker installation and spawning

'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { init, HiveClient, OperationBuilder } from '@rbee/rbee-hive-sdk'

// TEAM-378: Initialize WASM (same pattern as other hooks)
let wasmInitialized = false
async function ensureWasmInit() {
  if (!wasmInitialized) {
    init()
    wasmInitialized = true
  }
}

// TEAM-378: Create client instance
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

export interface UseWorkerOperationsResult {
  installWorker: (workerId: string) => void
  spawnWorker: (params: SpawnWorkerParams) => void
  isPending: boolean
  isSuccess: boolean
  isError: boolean
  error: Error | null
  installProgress: string[]
  reset: () => void
}

/**
 * Hook for Worker operations using TanStack Query mutations
 * 
 * TEAM-378: Handles worker installation and spawning
 * - installWorker: Download PKGBUILD, build, and install worker binary
 * - spawnWorker: Start a worker process with a model
 * 
 * @returns Mutation functions and state
 * 
 * @example
 * ```tsx
 * const { installWorker, spawnWorker, isPending } = useWorkerOperations()
 * 
 * // Install a worker binary
 * <button onClick={() => installWorker('llm-worker-rbee-cpu')}>
 *   Install Worker
 * </button>
 * 
 * // Spawn a worker process
 * <button onClick={() => spawnWorker({ 
 *   modelId: 'llama-3.2-1b',
 *   workerType: 'cuda',
 *   deviceId: 0
 * })}>
 *   Spawn Worker
 * </button>
 * ```
 */
export function useWorkerOperations(): UseWorkerOperationsResult {
  // TEAM-378: Track installation progress messages
  const [progressMessages, setProgressMessages] = useState<string[]>([])
  
  // TEAM-378: Worker installation mutation
  const installMutation = useMutation<any, Error, string>({
    mutationFn: async (workerId: string) => {
      console.log('[useWorkerOperations] 🎬 Starting installation mutation for:', workerId)
      setProgressMessages([]) // Clear previous messages
      
      console.log('[useWorkerOperations] 🔧 Initializing WASM...')
      await ensureWasmInit()
      console.log('[useWorkerOperations] ✓ WASM initialized')
      
      const hiveId = client.hiveId
      console.log('[useWorkerOperations] 🏠 Hive ID:', hiveId)
      
      // TEAM-378: workerInstall(hive_id, worker_id)
      console.log('[useWorkerOperations] 🔨 Building WorkerInstall operation...')
      const op = OperationBuilder.workerInstall(hiveId, workerId)
      console.log('[useWorkerOperations] ✓ Operation built:', op)
      
      const lines: string[] = []
      
      console.log('[useWorkerOperations] 📡 Submitting operation and streaming SSE...')
      await client.submitAndStream(op, (line: string) => {
        if (line !== '[DONE]') {
          lines.push(line)
          console.log('[useWorkerOperations] 📨 SSE message:', line)
          // TEAM-378: Update progress in real-time
          setProgressMessages(prev => [...prev, line])
        } else {
          console.log('[useWorkerOperations] 🏁 SSE stream complete ([DONE] received)')
        }
      })
      
      console.log('[useWorkerOperations] ✅ Installation complete! Total messages:', lines.length)
      return { success: true, workerId }
    },
    retry: 1,
    retryDelay: 1000,
  })

  // TEAM-377: Worker spawn mutation
  const spawnMutation = useMutation<any, Error, SpawnWorkerParams>({
    mutationFn: async ({ modelId, workerType = 'cuda', deviceId = 0 }) => {
      await ensureWasmInit()
      const hiveId = client.hiveId
      // TEAM-377: workerSpawn(hive_id, model, worker_type, device_id)
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
    retry: 1,
    retryDelay: 1000,
  })

  return {
    installWorker: installMutation.mutate,
    spawnWorker: spawnMutation.mutate,
    isPending: installMutation.isPending || spawnMutation.isPending,
    isSuccess: installMutation.isSuccess || spawnMutation.isSuccess,
    isError: installMutation.isError || spawnMutation.isError,
    error: installMutation.error || spawnMutation.error,
    installProgress: progressMessages,
    reset: () => {
      installMutation.reset()
      spawnMutation.reset()
      setProgressMessages([])
    },
  }
}
