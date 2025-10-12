import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

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

export function IconBox({
  icon: Icon,
  color = 'primary',
  size = 'md',
  variant = 'rounded',
  className,
}: IconBoxProps) {
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

  return (
    <div
      className={cn(
        'flex items-center justify-center',
        sizeClasses[size],
        variantClasses[variant],
        `bg-${color}/10`,
        className
      )}
    >
      <Icon className={cn(iconSizeClasses[size], `text-${color}`)} />
    </div>
  )
}
