import { cn } from '@/lib/utils'

export interface PulseBadgeProps {
  /** Badge text */
  text: string
  /** Color variant */
  variant?: 'primary' | 'success' | 'warning' | 'info'
  /** Size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Enable pulse animation */
  animated?: boolean
  /** Additional CSS classes */
  className?: string
}

export function PulseBadge({
  text,
  variant = 'primary',
  size = 'md',
  animated = true,
  className,
}: PulseBadgeProps) {
  const variantClasses = {
    primary: 'bg-primary/10 border-primary/20 text-primary',
    success: 'bg-chart-3/10 border-chart-3/20 text-chart-3',
    warning: 'bg-chart-4/10 border-chart-4/20 text-chart-4',
    info: 'bg-chart-2/10 border-chart-2/20 text-chart-2',
  }

  const sizeClasses = {
    sm: 'px-3 py-1 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-5 py-2.5 text-base',
  }

  const dotSizeClasses = {
    sm: 'h-1.5 w-1.5',
    md: 'h-2 w-2',
    lg: 'h-2.5 w-2.5',
  }

  const variantDotClasses = {
    primary: 'bg-primary',
    success: 'bg-chart-3',
    warning: 'bg-chart-4',
    info: 'bg-chart-2',
  }

  return (
    <div
      className={cn(
        'inline-flex items-center gap-2 border rounded-full',
        variantClasses[variant],
        sizeClasses[size],
        className
      )}
    >
      <span className={cn('relative flex', dotSizeClasses[size])}>
        {animated && (
          <span
            className={cn(
              'animate-ping absolute inline-flex h-full w-full rounded-full opacity-75',
              variantDotClasses[variant]
            )}
          ></span>
        )}
        <span
          className={cn(
            'relative inline-flex rounded-full',
            dotSizeClasses[size],
            variantDotClasses[variant]
          )}
        ></span>
      </span>
      {text}
    </div>
  )
}
