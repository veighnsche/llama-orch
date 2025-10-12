import { Check } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface CheckListItemProps {
  /** List item text */
  text: string
  /** Color variant */
  variant?: 'success' | 'primary' | 'muted'
  /** Size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Additional CSS classes */
  className?: string
}

export function CheckListItem({
  text,
  variant = 'success',
  size = 'md',
  className,
}: CheckListItemProps) {
  const variantClasses = {
    success: 'text-chart-3',
    primary: 'text-primary',
    muted: 'text-muted-foreground',
  }

  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-5 w-5',
    lg: 'h-6 w-6',
  }

  const textSizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  }

  return (
    <li className={cn('flex items-start gap-2', className)}>
      <Check
        className={cn(
          'flex-shrink-0 mt-0.5',
          sizeClasses[size],
          variantClasses[variant]
        )}
      />
      <span className={cn('text-muted-foreground', textSizeClasses[size])}>
        {text}
      </span>
    </li>
  )
}
