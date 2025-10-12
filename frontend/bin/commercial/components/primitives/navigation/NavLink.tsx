import { cn } from '@/lib/utils'
import Link from 'next/link'
import type { ReactNode } from 'react'

export interface NavLinkProps {
  href: string
  children: ReactNode
  variant?: 'default' | 'mobile'
  onClick?: () => void
  className?: string
}

export function NavLink({
  href,
  children,
  variant = 'default',
  onClick,
  className,
}: NavLinkProps) {
  const variantClasses = {
    default: 'text-muted-foreground hover:text-foreground transition-colors',
    mobile: 'text-foreground hover:text-primary transition-colors text-lg',
  }

  return (
    <Link
      href={href}
      onClick={onClick}
      className={cn(variantClasses[variant], className)}
    >
      {children}
    </Link>
  )
}
