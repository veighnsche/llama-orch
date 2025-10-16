import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface TrustIndicatorProps {
  /** Lucide icon component */
  icon: ReactNode
  /** Indicator text */
  text: string
  /** Color variant */
  variant?: 'default' | 'primary' | 'success'
  /** Additional CSS classes */
  className?: string
}

export function TrustIndicator({ icon: Icon, text, variant = 'default', className }: TrustIndicatorProps) {
  const variantClasses = {
    default: 'text-muted-foreground',
    primary: 'text-primary',
    success: 'text-chart-3',
  }

  return (
    <div className={cn('flex items-center gap-2', variantClasses[variant], className)}>
      <div className="h-5 w-5">{Icon}</div>
      <span className="text-sm">{text}</span>
    </div>
  )
}
