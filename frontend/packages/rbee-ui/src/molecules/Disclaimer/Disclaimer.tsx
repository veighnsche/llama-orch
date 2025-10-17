import { Card, CardContent } from '@rbee/ui/atoms/Card'
import { cn } from '@rbee/ui/utils'
import { AlertCircle, Info } from 'lucide-react'
import type { ReactNode } from 'react'

export interface DisclaimerProps {
  /** Disclaimer text content */
  children: ReactNode
  /** Visual variant */
  variant?: 'default' | 'info' | 'warning' | 'muted' | 'subtle' | 'card'
  /** Show icon */
  showIcon?: boolean
  /** Additional CSS classes */
  className?: string
}

const variantStyles = {
  default: {
    card: 'border-primary/20 bg-primary/5',
    text: 'text-foreground/90',
    icon: 'text-primary/70',
  },
  info: {
    card: 'border-blue-500/20 bg-blue-500/5',
    text: 'text-foreground/90',
    icon: 'text-blue-500/70',
  },
  warning: {
    card: 'border-amber-500/20 bg-amber-500/5',
    text: 'text-foreground/90',
    icon: 'text-amber-500/70',
  },
  muted: {
    card: 'border-border/40 bg-muted/30',
    text: 'text-muted-foreground',
    icon: 'text-muted-foreground/60',
  },
  subtle: {
    card: 'border-0 bg-transparent',
    text: 'text-muted-foreground/80',
    icon: 'text-muted-foreground/60',
  },
  card: {
    card: 'border-border/20 bg-primary/20',
    text: 'text-foreground/80',
    icon: 'text-muted-foreground/60',
  },
} as const

/**
 * Disclaimer molecule - displays important disclaimers, legal text, or informational notices
 *
 * @example
 * ```tsx
 * <Disclaimer variant="default">
 *   Earnings are estimates based on current market rates and may vary.
 * </Disclaimer>
 *
 * <Disclaimer variant="warning" showIcon>
 *   This feature is currently in beta testing.
 * </Disclaimer>
 *
 * <Disclaimer variant="subtle">
 *   Consult your legal team for certification.
 * </Disclaimer>
 * ```
 */
export function Disclaimer({ children, variant = 'default', showIcon = false, className }: DisclaimerProps) {
  const styles = variantStyles[variant]
  const Icon = variant === 'warning' ? AlertCircle : Info

  return (
    <Card className={cn(styles.card, className)}>
      <CardContent className="p-4">
        <div className="flex gap-3">
          {showIcon && <Icon className={cn('h-4 w-4 shrink-0 mt-0.5', styles.icon)} aria-hidden="true" />}
          <p className={cn('text-sm leading-relaxed font-sans', styles.text)}>{children}</p>
        </div>
      </CardContent>
    </Card>
  )
}

export { Disclaimer as default }
