import { cn } from '@rbee/ui/utils'

export interface ProgressBarProps {
  /** Progress label */
  label: string
  /** Progress percentage (0-100) */
  percentage: number
  /** Progress bar color (Tailwind class) */
  color?: string
  /** Size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Show label */
  showLabel?: boolean
  /** Show percentage */
  showPercentage?: boolean
  /** Additional CSS classes */
  className?: string
}

export function ProgressBar({
  label,
  percentage,
  color = 'primary',
  size = 'md',
  showLabel = true,
  showPercentage = true,
  className,
}: ProgressBarProps) {
  const heightClasses = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  }

  const labelWidthClasses = {
    sm: 'w-20',
    md: 'w-24',
    lg: 'w-28',
  }

  const colorClasses = {
    primary: 'bg-primary',
    'chart-1': 'bg-chart-1',
    'chart-2': 'bg-chart-2',
    'chart-3': 'bg-chart-3',
    'chart-4': 'bg-chart-4',
    'chart-5': 'bg-chart-5',
  }

  const bgColor = colorClasses[color as keyof typeof colorClasses] || colorClasses.primary

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {showLabel && <span className={cn('text-muted-foreground text-xs', labelWidthClasses[size])}>{label}</span>}
      <div className={cn('flex-1 bg-muted rounded-full overflow-hidden', heightClasses[size])}>
        <div className={cn('h-full', bgColor)} style={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}></div>
      </div>
      {showPercentage && <span className="text-muted-foreground text-xs">{percentage}%</span>}
    </div>
  )
}
