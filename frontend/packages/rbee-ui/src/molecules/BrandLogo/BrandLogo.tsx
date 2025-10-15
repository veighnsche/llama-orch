import { BrandMark } from '@rbee/ui/atoms/BrandMark'
import { BrandWordmark } from '@rbee/ui/atoms/BrandWordmark'
import { cn } from '@rbee/ui/utils'
import Link from 'next/link'

export interface BrandLogoProps {
  className?: string
  href?: string
  priority?: boolean
  size?: 'sm' | 'md' | 'lg'
}

const sizeClasses = {
  sm: 'gap-2',
  md: 'gap-2.5',
  lg: 'gap-3',
}

export function BrandLogo({ className, href = '/', priority = false, size = 'md' }: BrandLogoProps) {
  const content = (
    <>
      <BrandMark size={size} priority={priority} />
      <BrandWordmark size={size} />
    </>
  )

  if (href) {
    return (
      <Link href={href} aria-label="rbee home" className={cn('flex items-center', sizeClasses[size], className)}>
        {content}
      </Link>
    )
  }

  return <div className={cn('flex items-center', sizeClasses[size], className)}>{content}</div>
}
