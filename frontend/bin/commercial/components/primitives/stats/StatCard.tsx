import { cn } from '@/lib/utils'

export interface StatCardProps {
  /** Stat value */
  value: string | number
  /** Stat label */
  label: string
  /** Color variant */
  variant?: 'primary' | 'success' | 'warning'
  /** Size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Additional CSS classes */
  className?: string
}

export function StatCard({
  value,
  label,
  variant = 'primary',
  size = 'md',
  className,
}: StatCardProps) {
  const variantClasses = {
    primary: 'text-primary',
    success: 'text-chart-3',
    warning: 'text-amber-500',
  }

  const valueSizeClasses = {
    sm: 'text-2xl',
    md: 'text-4xl',
    lg: 'text-5xl',
  }

  const labelSizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  }

  const spacingClasses = {
    sm: 'mb-1',
    md: 'mb-2',
    lg: 'mb-3',
  }

  return (
    <div className={cn('text-center', className)}>
      <div
        className={cn(
          'font-bold',
          valueSizeClasses[size],
          spacingClasses[size],
          variantClasses[variant]
        )}
      >
        {value}
      </div>
      <div className={cn('text-muted-foreground', labelSizeClasses[size])}>
        {label}
      </div>
    </div>
  )
}
