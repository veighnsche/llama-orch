'use client'

import { cn } from '@/lib/utils'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import type { ReactNode } from 'react'

export interface NavLinkProps {
  href: string
  children: ReactNode
  variant?: 'default' | 'mobile'
  onClick?: () => void
  className?: string
  target?: string
  rel?: string
}

export function NavLink({ href, children, variant = 'default', onClick, className, target, rel }: NavLinkProps) {
  const pathname = usePathname()
  const isActive = pathname === href || (href !== '/' && pathname.startsWith(href))
  const isExternal = href.startsWith('http')

  const variantClasses = {
    default: cn(
      'relative text-muted-foreground hover:text-foreground transition-colors',
      'after:absolute after:left-0 after:right-0 after:-bottom-2 after:h-0.5 after:rounded-full after:bg-primary/80',
      'after:transition-opacity after:duration-200',
      isActive ? 'text-foreground after:opacity-100' : 'after:opacity-0'
    ),
    mobile: cn(
      'block text-foreground hover:text-primary transition-colors text-lg',
      isActive && 'text-primary font-medium'
    ),
  }

  return (
    <Link
      href={href}
      onClick={onClick}
      className={cn(variantClasses[variant], className)}
      target={target}
      rel={rel}
    >
      {children}
    </Link>
  )
}
