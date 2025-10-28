import { BrandMark } from '@rbee/ui/atoms/BrandMark'
import { BrandWordmark } from '@rbee/ui/atoms/BrandWordmark'
import { cn } from '@rbee/ui/utils'

export interface BrandLogoProps {
  /** Additional CSS classes */
  className?: string
  /** Next.js Image priority loading (passed to BrandMark) */
  priority?: boolean
  /** Size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Wrapper element type - use 'div' for static logo, or wrap with your own Link component */
  as?: 'div' | 'span'
}

const sizeClasses = {
  sm: 'gap-2',
  md: 'gap-2.5',
  lg: 'gap-3',
}

/**
 * BrandLogo - Framework-agnostic brand logo component
 *
 * @example
 * // Static logo
 * <BrandLogo />
 *
 * @example
 * // With Next.js Link
 * import Link from 'next/link'
 * <Link href="/">
 *   <BrandLogo />
 * </Link>
 *
 * @example
 * // With React Router Link
 * import { Link } from 'react-router-dom'
 * <Link to="/">
 *   <BrandLogo />
 * </Link>
 */
export function BrandLogo({ className, priority = false, size = 'md', as: Component = 'div' }: BrandLogoProps) {
  return (
    <Component className={cn('flex items-center', sizeClasses[size], className)}>
      <BrandMark size={size} priority={priority} />
      <BrandWordmark size={size} />
    </Component>
  )
}
