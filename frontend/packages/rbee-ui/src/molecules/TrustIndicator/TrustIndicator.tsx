import { cn } from '@rbee/ui/utils'
import type { LucideIcon } from 'lucide-react'

export interface TrustIndicatorProps {
  /** Lucide icon component */
  icon: LucideIcon
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
      <Icon className="h-5 w-5" />
      <span className="text-sm">{text}</span>
    </div>
  )
}
