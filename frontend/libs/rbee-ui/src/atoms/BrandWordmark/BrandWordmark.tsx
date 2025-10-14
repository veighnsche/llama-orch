import { cn } from '@rbee/ui/utils'

export interface BrandWordmarkProps {
  /** Size variant */
  size?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '3xl' | '4xl' | '5xl'
  /** Additional CSS classes */
  className?: string
  /** Inline display (for flowing with text) */
  inline?: boolean
}

const sizeClasses = {
  sm: 'text-sm',
  md: 'text-base',
  lg: 'text-xl',
  xl: 'text-2xl',
  '2xl': 'text-3xl',
  '3xl': 'text-4xl',
  '4xl': 'text-5xl',
  '5xl': 'text-6xl',
}

export function BrandWordmark({ size = 'md', className, inline = false }: BrandWordmarkProps) {
  return (
    <span
      className={cn(
        'font-bold tracking-tight text-foreground',
        sizeClasses[size],
        inline && 'inline',
        className
      )}
      style={{ fontFamily: 'var(--font-geist-mono)' }}
    >
      rbee
    </span>
  )
}
