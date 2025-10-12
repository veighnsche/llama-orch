import Link from 'next/link'
import Image from 'next/image'
import { cn } from '@/lib/utils'

export interface BrandLogoProps {
  className?: string
  href?: string
  showWordmark?: boolean
  priority?: boolean
  size?: 'sm' | 'md' | 'lg'
}

const sizeClasses = {
  sm: {
    icon: 'size-5',
    gap: 'gap-2',
    text: 'text-sm',
  },
  md: {
    icon: 'size-6',
    gap: 'gap-2.5',
    text: 'text-base',
  },
  lg: {
    icon: 'size-8',
    gap: 'gap-3',
    text: 'text-xl',
  },
}

export function BrandLogo({
  className,
  href = '/',
  showWordmark = true,
  priority = false,
  size = 'md',
}: BrandLogoProps) {
  const sizes = sizeClasses[size]
  
  const content = (
    <>
      <Image
        src="/brand/bee-mark.svg"
        width={size === 'sm' ? 20 : size === 'lg' ? 32 : 24}
        height={size === 'sm' ? 20 : size === 'lg' ? 32 : 24}
        priority={priority}
        alt="rbee orchestration platform - distributed AI infrastructure"
        className={cn('rounded-sm', sizes.icon)}
      />
      {showWordmark && (
        <span
          className={cn('font-bold tracking-tight text-foreground', sizes.text)}
          style={{ fontFamily: 'var(--font-geist-mono)' }}
        >
          rbee
        </span>
      )}
    </>
  )

  if (href) {
    return (
      <Link href={href} aria-label="rbee home" className={cn('flex items-center', sizes.gap, className)}>
        {content}
      </Link>
    )
  }

  return <div className={cn('flex items-center', sizes.gap, className)}>{content}</div>
}
