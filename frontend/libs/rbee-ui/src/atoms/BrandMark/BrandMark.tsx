import Image from 'next/image'
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

export function BrandMark({ size = 'md', className, priority = false }: BrandMarkProps) {
  const pixels = sizeMap[size]

  return (
    <Image
      src="/illustrations/bee-mark.svg"
      width={pixels}
      height={pixels}
      priority={priority}
      alt="rbee orchestration platform"
      className={cn('rounded-sm', className)}
      style={{ width: pixels, height: pixels }}
    />
  )
}
