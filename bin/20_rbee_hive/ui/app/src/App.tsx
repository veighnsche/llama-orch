// TEAM-353: Hive UI - Worker & Model Management
// TEAM-374: Added HeartbeatMonitor for real-time worker updates
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { logStartupMode } from '@rbee/dev-utils'
import { useState, useEffect } from 'react'
import './App.css'

// TEAM-353: Use shared startup logging
logStartupMode("HIVE UI", import.meta.env.DEV, 7836)

// TEAM-353: Create QueryClient for TanStack Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
  },
})

// TEAM-374: Heartbeat status component
function HeartbeatStatus() {
  const [connected, setConnected] = useState(false)
  const [workerCount, setWorkerCount] = useState(0)
  const [lastUpdate, setLastUpdate] = useState<string>('')

  useEffect(() => {
    let monitor: any = null

    const initMonitor = async () => {
      try {
        // TEAM-374: Import WASM SDK
        const { HeartbeatMonitor } = await import('@rbee/rbee-hive-sdk')
        
        const baseUrl = import.meta.env.DEV 
          ? 'http://localhost:7835'  // Dev: direct to hive
          : window.location.origin    // Prod: same origin

        monitor = new HeartbeatMonitor(baseUrl)
        
        monitor.start((event: any) => {
          setConnected(true)
          setWorkerCount(event.workers?.length || 0)
          setLastUpdate(new Date().toLocaleTimeString())
        })
      } catch (err) {
        console.error('Failed to start heartbeat monitor:', err)
      }
    }

    initMonitor()

    return () => {
      if (monitor) {
        monitor.stop()
      }
    }
  }, [])

  return (
    <div style={{ 
      padding: '1rem', 
      border: '1px solid #646cff', 
      borderRadius: '8px',
      marginBottom: '2rem'
    }}>
      <h2>🐝 Hive Heartbeat</h2>
      <div style={{ display: 'flex', gap: '2rem', marginTop: '1rem' }}>
        <div>
          <strong>Status:</strong>{' '}
          <span style={{ color: connected ? '#4ade80' : '#f87171' }}>
            {connected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
        </div>
        <div>
          <strong>Workers:</strong> {workerCount}
        </div>
        {lastUpdate && (
          <div>
            <strong>Last Update:</strong> {lastUpdate}
          </div>
        )}
      </div>
    </div>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div>
        <h1>🐝 Hive UI</h1>
        <HeartbeatStatus />
        <p className="read-the-docs">
          Worker & Model Management Dashboard
        </p>
      </div>
    </QueryClientProvider>
  )
}

export default App
