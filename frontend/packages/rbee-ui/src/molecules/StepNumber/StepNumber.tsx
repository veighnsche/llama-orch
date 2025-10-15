import { cn } from '@rbee/ui/utils'

export interface StepNumberProps {
  /** Step number */
  number: number
  /** Size variant */
  size?: 'sm' | 'md' | 'lg' | 'xl'
  /** Color variant */
  variant?: 'primary' | 'secondary' | 'outline'
  /** Additional CSS classes */
  className?: string
}

export function StepNumber({ number, size = 'md', variant = 'primary', className }: StepNumberProps) {
  const sizeClasses = {
    sm: 'h-8 w-8 text-sm',
    md: 'h-12 w-12 text-xl',
    lg: 'h-16 w-16 text-2xl',
    xl: 'h-20 w-20 text-3xl',
  }

  const variantClasses = {
    primary: 'bg-primary text-primary-foreground',
    secondary: 'bg-secondary text-secondary-foreground',
    outline: 'bg-transparent border-2 border-primary text-primary',
  }

  return (
    <div
      className={cn(
        'inline-flex items-center justify-center rounded-full font-bold',
        sizeClasses[size],
        variantClasses[variant],
        className,
      )}
    >
      {number}
    </div>
  )
}
