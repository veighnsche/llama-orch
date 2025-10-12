import { cn } from '@/lib/utils'

export interface GPUListItemProps {
  name: string
  subtitle?: string
  value: string | number
  label?: string
  status?: 'active' | 'idle' | 'offline'
  statusColor?: string
  className?: string
}

export function GPUListItem({
  name,
  subtitle,
  value,
  label,
  status = 'active',
  statusColor,
  className,
}: GPUListItemProps) {
  const statusColors = {
    active: statusColor || 'bg-chart-3',
    idle: 'bg-chart-4',
    offline: 'bg-muted-foreground/30',
  }

  return (
    <div
      className={cn(
        'flex items-center justify-between rounded-lg border border-border bg-background/50 p-3',
        className
      )}
    >
      <div className="flex items-center gap-3">
        <div className={cn('h-2 w-2 rounded-full', statusColors[status])} />
        <div>
          <div className="text-sm font-medium text-foreground">{name}</div>
          {subtitle && (
            <div className="text-xs text-muted-foreground">{subtitle}</div>
          )}
        </div>
      </div>
      <div className="text-right">
        <div className="text-sm font-medium text-foreground">{value}</div>
        {label && (
          <div className="text-xs text-muted-foreground">{label}</div>
        )}
      </div>
    </div>
  )
}
