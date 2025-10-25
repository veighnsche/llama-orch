import { cn } from '@rbee/ui/utils'
import { Shield } from 'lucide-react'

export interface RibbonBannerProps {
  /** Banner text content */
  text: string
  /** Additional CSS classes */
  className?: string
}

/**
 * RibbonBanner organism - Emerald-themed banner for insurance/guarantee messaging
 */
export function RibbonBanner({ text, className }: RibbonBannerProps) {
  return (
    <div className={cn('rounded border border-emerald-400/30 bg-emerald-400/10 p-5 text-center', className)}>
      <p className="flex items-center justify-center gap-2 text-balance text-base font-medium text-emerald-400 lg:text-lg">
        <Shield className="h-4 w-4" aria-hidden="true" />
        <span className="tabular-nums">{text}</span>
      </p>
    </div>
  )
}
