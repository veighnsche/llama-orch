// TEAM-353: Hive operations with narration support
// Uses @rbee/narration-client (no custom implementation)

'use client'

import { useState } from 'react'
import { createStreamHandler, SERVICES } from '@rbee/narration-client'
import { getServiceUrl } from '@rbee/shared-config'

export interface UseHiveOperationsResult {
  spawnWorker: (modelId: string) => Promise<void>
  spawning: boolean
  error: Error | null
}

/**
 * Hook for Hive operations with narration
 * 
 * TEAM-353: Uses @rbee/narration-client for narration
 * 
 * @param baseUrl - Optional Hive API URL (defaults to shared config)
 * @returns Hive operation functions
 */
export function useHiveOperations(
  baseUrl?: string
): UseHiveOperationsResult {
  const [spawning, setSpawning] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  // TEAM-353: Use shared config for default URL
  const isDev = (import.meta as any).env?.DEV ?? false
  const defaultUrl = getServiceUrl('hive', isDev ? 'dev' : 'prod')
  const url = baseUrl || defaultUrl

  const spawnWorker = async (modelId: string) => {
    setSpawning(true)
    setError(null)

    try {
      // TEAM-353: Create narration handler using shared package
      const narrationHandler = createStreamHandler(SERVICES.hive, (event) => {
        console.log('[Hive] Narration event:', event)
      }, {
        debug: true,
        silent: false,
        validate: true,
      })

      // Submit operation with narration
      const operation = {
        operation: 'worker_spawn',
        model_id: modelId,
      }

      // TODO: Use Hive SDK to submit operation with narration
      // This depends on how rbee-hive-sdk exposes operations
      // For now, this is a placeholder showing the pattern

      console.log('[Hive] Spawning worker with model:', modelId)
      console.log('[Hive] Using URL:', url)
      // await hiveClient.submitAndStream(operation, narrationHandler)

    } catch (err) {
      console.error('[Hive] Worker spawn error:', err)
      setError(err as Error)
      throw err
    } finally {
      setSpawning(false)
    }
  }

  return {
    spawnWorker,
    spawning,
    error,
  }
}
