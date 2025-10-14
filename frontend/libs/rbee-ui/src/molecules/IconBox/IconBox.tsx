import type { LucideIcon } from 'lucide-react'
import { cn } from '@rbee/ui/utils'

export interface IconBoxProps {
  /** Lucide icon component */
  icon: LucideIcon
  /** Icon color (Tailwind class) */
  color?: string
  /** Size variant */
  size?: 'sm' | 'md' | 'lg' | 'xl'
  /** Shape variant */
  variant?: 'rounded' | 'circle' | 'square'
  /** Additional CSS classes */
  className?: string
}

export function IconBox({ icon: Icon, color = 'primary', size = 'md', variant = 'rounded', className }: IconBoxProps) {
  const sizeClasses = {
    sm: 'h-8 w-8',
    md: 'h-10 w-10',
    lg: 'h-12 w-12',
    xl: 'h-14 w-14',
  }

  const iconSizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-5 w-5',
    lg: 'h-6 w-6',
    xl: 'h-7 w-7',
  }

  const variantClasses = {
    rounded: 'rounded-lg',
    circle: 'rounded-full',
    square: 'rounded-none',
  }

  const colorClasses = {
    primary: { bg: 'bg-primary/10', text: 'text-primary' },
    'chart-1': { bg: 'bg-chart-1/10', text: 'text-chart-1' },
    'chart-2': { bg: 'bg-chart-2/10', text: 'text-chart-2' },
    'chart-3': { bg: 'bg-chart-3/10', text: 'text-chart-3' },
    'chart-4': { bg: 'bg-chart-4/10', text: 'text-chart-4' },
    'chart-5': { bg: 'bg-chart-5/10', text: 'text-chart-5' },
  }

  const colors = colorClasses[color as keyof typeof colorClasses] || colorClasses.primary

  return (
    <div
      className={cn(
        'flex items-center justify-center',
        sizeClasses[size],
        variantClasses[variant],
        colors.bg,
        className,
      )}
    >
      <Icon className={cn(iconSizeClasses[size], colors.text)} />
    </div>
  )
}
