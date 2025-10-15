import { BeeMark } from '@rbee/ui/icons'
import { cn } from '@rbee/ui/utils'

export interface BrandMarkProps {
  /** Size variant */
  size?: 'sm' | 'md' | 'lg' | 'xl'
  /** Additional CSS classes */
  className?: string
  /** Next.js Image priority loading */
  priority?: boolean
}

const sizeMap = {
  sm: 20,
  md: 24,
  lg: 32,
  xl: 48,
}

export function BrandMark({ size = 'md', className, priority = false, ...rest }: BrandMarkProps) {
  const pixels = sizeMap[size]
  const alt = 'rbee orchestration platform'

  return <BeeMark size={pixels} aria-label={alt} className={cn('rounded-sm', className)} {...rest} />
}
