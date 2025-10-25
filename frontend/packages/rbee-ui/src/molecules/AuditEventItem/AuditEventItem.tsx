import { Badge } from '@rbee/ui/atoms/Badge'
import { cn } from '@rbee/ui/utils'

export interface AuditEventItemProps {
  /** Event name/type (e.g., "auth.success", "data.access") */
  event: string
  /** User who triggered the event */
  user: string
  /** ISO 8601 timestamp for the time element */
  time: string
  /** Human-readable display time */
  displayTime: string
  /** Event status */
  status: string
  /** Optional className for custom styling */
  className?: string
}

export function AuditEventItem({ event, user, time, displayTime, status, className }: AuditEventItemProps) {
  return (
    <li
      className={cn('rounded border border-border bg-background p-3', className)}
      aria-label={`${event} by ${user} at ${displayTime} – ${status}`}
    >
      <div className="mb-1 flex items-center justify-between">
        <span className="font-mono text-sm text-primary">{event}</span>
        <Badge variant="secondary" className="bg-chart-3/20 px-2 py-0.5 text-xs text-chart-3">
          {status}
        </Badge>
      </div>
      <div className="text-xs text-muted-foreground">
        <div>{user}</div>
        <time dateTime={time} className="text-muted-foreground/70">
          {displayTime}
        </time>
      </div>
    </li>
  )
}
