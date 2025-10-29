// Heartbeat hook for Queen UI
// Connects to HeartbeatMonitor from SDK

'use client'

import { useEffect, useState } from 'react'
import { useRbeeSDK } from './useRbeeSDK'

export interface HeartbeatData {
  workers_online: number
  hives_online: number
  timestamp: string
  workers: Array<{
    id: string
    model_id: string
    device: number
    port: number
    status: string
  }>
}

export interface UseHeartbeatResult {
  data: HeartbeatData | null
  connected: boolean
  loading: boolean
  error: Error | null
}

/**
 * Hook for monitoring Queen heartbeat
 * 
 * @param baseUrl - Queen API URL (default: http://localhost:7833)
 * @returns Heartbeat data and connection status
 */
export function useHeartbeat(baseUrl: string = 'http://localhost:7833'): UseHeartbeatResult {
  const { sdk, loading: sdkLoading, error: sdkError } = useRbeeSDK()
  const [data, setData] = useState<HeartbeatData | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!sdk) return

    let monitor: any = null
    let mounted = true

    // TEAM-XXX: Check health before starting SSE to avoid CORS errors when queen is offline
    const startMonitoring = async () => {
      try {
        monitor = new sdk.HeartbeatMonitor(baseUrl)
        
        // Check if queen is reachable
        const isHealthy = await monitor.checkHealth()
        
        if (!mounted) return
        
        if (!isHealthy) {
          setError(new Error('Queen is offline'))
          setConnected(false)
          return
        }
        
        // Queen is healthy, start SSE
        monitor.start((snapshot: any) => {
          if (!mounted) return
          setData(snapshot)
          setConnected(true)
          setError(null)
        })
      } catch (err) {
        if (!mounted) return
        setError(err as Error)
        setConnected(false)
      }
    }

    startMonitoring()

    return () => {
      mounted = false
      if (monitor) {
        monitor.stop()
      }
    }
  }, [sdk, baseUrl])

  return {
    data,
    connected,
    loading: sdkLoading,
    error: error || sdkError,
  }
}
