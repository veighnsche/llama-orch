'use client'

import { Shield } from 'lucide-react'
import { cn } from '@/lib/utils'
import Link from 'next/link'

export interface PledgeCalloutProps {
  /** Additional CSS classes */
  className?: string
}

export function PledgeCallout({ className }: PledgeCalloutProps) {
  return (
    <div className={cn('rounded-2xl border bg-card p-6 md:p-7 shadow-sm', className)}>
      <div className="flex gap-4 items-start">
        {/* Icon */}
        <div className="h-9 w-9 rounded-full bg-accent/20 flex items-center justify-center shrink-0" aria-hidden="true">
          <Shield className="size-5 text-chart-2" aria-hidden="true" />
        </div>
        
        {/* Content */}
        <div className="flex-1 space-y-1">
          <p className="text-sm md:text-base font-semibold text-card-foreground">
            Your models. Your rules.
          </p>
          <p className="text-sm text-muted-foreground leading-6">
            rbee enforces zero-trust auth, immutable audit trails, and strict bind policies—so your code stays yours.{' '}
            <Link href="/security" className="text-primary hover:underline underline-offset-4">
              Security details
            </Link>
          </p>
        </div>
      </div>
    </div>
  )
}
