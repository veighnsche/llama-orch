'use client'

import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import { Shield } from 'lucide-react'
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
        <IconPlate icon={<Shield className="w-6 h-6" />} size="md" shape="circle" tone="chart-2" className="shrink-0" />

        {/* Content */}
        <div className="flex-1 space-y-1">
          <p className="text-sm md:text-base font-semibold text-card-foreground">Your models. Your rules.</p>
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
