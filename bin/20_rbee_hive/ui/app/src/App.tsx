// TEAM-374: Hive UI - Rebuilt with shared components
// Uses @rbee/ui for consistent styling across all rbee applications
import { QueryProvider } from '@rbee/ui/providers'
import { logStartupMode } from '@rbee/dev-utils'
import { receiveThemeChanges } from '@rbee/iframe-bridge'
import { useState, useEffect } from 'react'
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle,
  Badge
} from '@rbee/ui/atoms'
import { 
  StatusKPI,
  MetricCard
} from '@rbee/ui/molecules'
import { Activity, Cpu, HardDrive } from 'lucide-react'
import { ModelManagement } from './components/ModelManagement'

// TEAM-374: Use shared startup logging
logStartupMode("HIVE UI", import.meta.env.DEV, 7836)

// TEAM-374: Device type from capabilities endpoint
interface HiveDevice {
  id: string
  device_type: 'GPU' | 'CPU'
  name: string
  memory_mb?: number
  compute_capability?: string
}

// TEAM-374: Fetch device capabilities
function useDevices() {
  const [devices, setDevices] = useState<HiveDevice[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchDevices = async () => {
      try {
        // TEAM-378: In dev mode, derive backend URL from current origin
        // If loaded from http://192.168.178.241:7836, backend is at http://192.168.178.241:7835
        const baseUrl = import.meta.env.DEV 
          ? `${window.location.protocol}//${window.location.hostname}:7835`
          : window.location.origin
        
        // TEAM-381: Capabilities endpoint moved to /v1/capabilities
        const response = await fetch(`${baseUrl}/v1/capabilities`)
        const data = await response.json()
        setDevices(data.devices || [])
      } catch (err) {
        console.error('Failed to fetch devices:', err)
      } finally {
        setLoading(false)
      }
    }

    fetchDevices()
  }, [])

  return { devices, loading }
}

// TEAM-374: Heartbeat status component with shared UI components
// TEAM-378: Added hive metadata extraction for self-awareness
function HeartbeatStatus() {
  const [connected, setConnected] = useState(false)
  const [workerCount, setWorkerCount] = useState(0)
  const [lastUpdate, setLastUpdate] = useState<string>('')
  const [hiveInfo, setHiveInfo] = useState<any>(null)

  useEffect(() => {
    let monitor: any = null

    const initMonitor = async () => {
      try {
        const { HeartbeatMonitor } = await import('@rbee/rbee-hive-sdk')
        
        // TEAM-378: In dev mode, derive backend URL from current origin
        const baseUrl = import.meta.env.DEV 
          ? `${window.location.protocol}//${window.location.hostname}:7835`
          : window.location.origin

        monitor = new HeartbeatMonitor(baseUrl)
        
        monitor.start((event: any) => {
          setConnected(true)
          setWorkerCount(event.workers?.length || 0)
          setLastUpdate(new Date().toLocaleTimeString())
          // TEAM-378: Extract hive metadata for self-awareness
          if (event.hive_info) {
            setHiveInfo(event.hive_info)
          }
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
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Hive Status
            </CardTitle>
            <CardDescription>
              {/* TEAM-378: Display hive metadata for self-awareness */}
              {hiveInfo ? (
                <span>
                  {hiveInfo.id} • {hiveInfo.hostname}:{hiveInfo.port} • v{hiveInfo.version}
                </span>
              ) : (
                'Real-time worker telemetry'
              )}
            </CardDescription>
          </div>
          <div className="flex gap-2">
            {/* TEAM-378: Show operational status */}
            {hiveInfo && (
              <Badge variant="outline">
                {hiveInfo.operational_status}
              </Badge>
            )}
            <Badge variant={connected ? "default" : "destructive"}>
              {connected ? 'Connected' : 'Disconnected'}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <StatusKPI
            label="Connection"
            value={connected ? 'Online' : 'Offline'}
            color={connected ? 'success' : 'warning'}
            icon={<Activity className="h-4 w-4" />}
          />
          <StatusKPI
            label="Active Workers"
            value={workerCount.toString()}
            color="primary"
            icon={<Cpu className="h-4 w-4" />}
          />
          <StatusKPI
            label="Last Update"
            value={lastUpdate || 'Never'}
            color="muted"
            icon={<HardDrive className="h-4 w-4" />}
          />
        </div>
      </CardContent>
    </Card>
  )
}

function App() {
  const { devices, loading: devicesLoading } = useDevices()
  const [hiveInfo, setHiveInfo] = useState<any>(null)

  // TEAM-375: Listen for theme changes from parent (Keeper)
  useEffect(() => {
    const cleanup = receiveThemeChanges()
    return cleanup
  }, [])

  // TEAM-378: Subscribe to heartbeat for hive metadata
  useEffect(() => {
    let monitor: any = null

    const initMonitor = async () => {
      try {
        const { HeartbeatMonitor } = await import('@rbee/rbee-hive-sdk')
        
        // TEAM-378: In dev mode, derive backend URL from current origin
        const baseUrl = import.meta.env.DEV 
          ? `${window.location.protocol}//${window.location.hostname}:7835`
          : window.location.origin

        monitor = new HeartbeatMonitor(baseUrl)
        
        monitor.start((event: any) => {
          if (event.hive_info) {
            setHiveInfo(event.hive_info)
          }
        })
      } catch (err) {
        console.error('Failed to start heartbeat monitor for header:', err)
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
    <QueryProvider>
      <div className="min-h-screen bg-background text-foreground font-sans p-6 space-y-6">
          {/* Header - TEAM-378: Self-aware with hive metadata */}
          <div>
            <h1 className="text-3xl font-bold">
              {hiveInfo ? `Hive: ${hiveInfo.id}` : 'Hive'}
            </h1>
            <p className="text-muted-foreground">
              {hiveInfo ? (
                <span>
                  {hiveInfo.hostname}:{hiveInfo.port} • Version {hiveInfo.version} • {hiveInfo.operational_status}
                </span>
              ) : (
                'Worker & Model Management Dashboard'
              )}
            </p>
          </div>

          {/* Status Row */}
          <HeartbeatStatus />

          {/* Management Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Model Management */}
            <ModelManagement />

            {/* Worker Management */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    <Cpu className="h-5 w-5" />
                    Worker Management
                  </CardTitle>
                  <Badge variant="secondary">0 Workers</Badge>
                </div>
                <CardDescription>Build and manage worker processes</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="text-sm text-muted-foreground text-center py-8">
                  No workers available
                </div>
                <div className="flex gap-2">
                  <button className="flex-1 px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors">
                    Build Worker
                  </button>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Spawn Worker */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Spawn Worker
              </CardTitle>
              <CardDescription>Start a worker on an available device</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {devicesLoading ? (
                <div className="text-sm text-muted-foreground text-center py-4">
                  Loading devices...
                </div>
              ) : devices.length === 0 ? (
                <div className="text-sm text-muted-foreground text-center py-4">
                  No devices detected
                </div>
              ) : (
                <div className="space-y-2">
                  <div className="text-sm font-medium text-foreground mb-3">Available Devices:</div>
                  {devices.map((device) => (
                    <div
                      key={device.id}
                      className="flex items-center justify-between p-3 border border-border rounded-md hover:bg-accent/50 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        {device.device_type === 'GPU' ? (
                          <Cpu className="h-5 w-5 text-primary" />
                        ) : (
                          <Cpu className="h-5 w-5 text-muted-foreground" />
                        )}
                        <div>
                          <div className="text-sm font-medium text-foreground">{device.name}</div>
                          <div className="text-xs text-muted-foreground">
                            {device.device_type} • {device.id}
                            {device.memory_mb && ` • ${Math.round(device.memory_mb / 1024)} GB`}
                          </div>
                        </div>
                      </div>
                      <button className="px-3 py-1 text-sm bg-primary text-primary-foreground rounded hover:bg-primary/90 transition-colors">
                        Spawn
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* GPU Utilization */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                GPU Utilization
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricCard
                  label="Average Usage"
                  value="0%"
                  description="Across all GPUs"
                />
                <MetricCard
                  label="VRAM Used"
                  value="0 MB"
                  description="Total allocated"
                />
                <MetricCard
                  label="Active GPUs"
                  value="0"
                  description="Currently in use"
                />
              </div>
            </CardContent>
          </Card>
        </div>
    </QueryProvider>
  )
}

export default App
