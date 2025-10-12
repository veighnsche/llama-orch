import Link from 'next/link'
import { cn } from '@/lib/utils'
import { BrandMark } from '@/components/atoms/BrandMark/BrandMark'
import { BrandWordmark } from '@/components/atoms/BrandWordmark/BrandWordmark'

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

export function BrandLogo({
  className,
  href = '/',
  priority = false,
  size = 'md',
}: BrandLogoProps) {
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
