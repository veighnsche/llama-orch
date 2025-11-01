// TEAM-382: Individual worker card component

import { Cpu, Zap, HardDrive, Activity, X } from 'lucide-react'
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Badge,
  Button,
  Progress,
} from '@rbee/ui/atoms'
import type { ProcessStats } from './types'

interface WorkerCardProps {
  worker: ProcessStats
  onTerminate?: (pid: number) => void
  isTerminating?: boolean
}

export function WorkerCard({ worker, onTerminate, isTerminating }: WorkerCardProps) {
  const isIdle = worker.gpu_util_pct === 0.0
  const vramUsagePercent = worker.total_vram_mb > 0 
    ? (worker.vram_mb / worker.total_vram_mb) * 100 
    : 0

  return (
    <Card className="relative">
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            <Activity className={`h-4 w-4 ${isIdle ? 'text-muted-foreground' : 'text-green-500'}`} />
            <CardTitle className="text-base">
              {worker.model || 'Unknown Model'}
            </CardTitle>
          </div>
          {onTerminate && (
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 -mr-2 -mt-1"
              onClick={() => onTerminate(worker.pid)}
              disabled={isTerminating}
            >
              <X className="h-4 w-4" />
            </Button>
          )}
        </div>
        <div className="flex gap-2 mt-2">
          <Badge variant={isIdle ? 'secondary' : 'default'} className="text-xs">
            {isIdle ? 'Idle' : 'Active'}
          </Badge>
          <Badge variant="outline" className="text-xs">
            {worker.group}:{worker.instance}
          </Badge>
          <Badge variant="outline" className="text-xs">
            PID {worker.pid}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-3">
        {/* GPU Utilization */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-1.5">
              <Zap className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-muted-foreground">GPU</span>
            </div>
            <span className="font-medium">{worker.gpu_util_pct.toFixed(1)}%</span>
          </div>
          <Progress value={worker.gpu_util_pct} className="h-1.5" />
        </div>

        {/* VRAM Usage */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-1.5">
              <HardDrive className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-muted-foreground">VRAM</span>
            </div>
            <span className="font-medium">
              {worker.vram_mb} / {worker.total_vram_mb} MB
            </span>
          </div>
          <Progress value={vramUsagePercent} className="h-1.5" />
        </div>

        {/* CPU & RAM */}
        <div className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-1.5">
              <Cpu className="h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-muted-foreground">CPU</span>
            </div>
            <span className="font-medium">{worker.cpu_pct.toFixed(1)}%</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">RAM</span>
            <span className="font-medium">{worker.rss_mb} MB</span>
          </div>
        </div>

        {/* Uptime */}
        <div className="pt-2 border-t">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Uptime</span>
            <span>{formatUptime(worker.uptime_s)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function formatUptime(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`
  } else {
    return `${secs}s`
  }
}
