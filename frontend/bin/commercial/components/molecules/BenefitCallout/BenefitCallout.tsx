import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

export interface BenefitCalloutProps {
  /** Callout text */
  text: string
  /** Color variant */
  variant?: 'success' | 'primary' | 'info' | 'warning'
  /** Optional icon */
  icon?: ReactNode
  /** Additional CSS classes */
  className?: string
}

export function BenefitCallout({
  text,
  variant = 'success',
  icon,
  className,
}: BenefitCalloutProps) {
  const variantClasses = {
    success: 'bg-chart-3/10 border-chart-3/20 text-chart-3',
    primary: 'bg-primary/10 border-primary/20 text-primary',
    info: 'bg-chart-2/10 border-chart-2/20 text-chart-2',
    warning: 'bg-chart-4/10 border-chart-4/20 text-chart-4',
  }

  return (
    <div
      className={cn(
        'border rounded-lg p-4',
        variantClasses[variant],
        className
      )}
    >
      <p className="font-medium flex items-center gap-2">
        {icon || '✓'} {text}
      </p>
    </div>
  )
}
