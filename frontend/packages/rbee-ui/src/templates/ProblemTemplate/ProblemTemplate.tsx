'use client'

import { FeatureInfoCard } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type * as React from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/**
 * ProblemItem represents a single problem card in the section.
 * Supports both icon types for backward compatibility.
 */
export type ProblemItem = {
  title: string
  body: string
  icon: React.ComponentType<{ className?: string }> | React.ReactNode
  tag?: string
  tone?: 'primary' | 'destructive' | 'muted'
}

/**
 * ProblemSection template displays a grid of problem cards with improved visual hierarchy.
 * Based on the ProvidersProblem design with CTA banner, loss tags, and staggered animations.
 *
 * @example
 * ```tsx
 * <ProblemSection
 *   items={[
 *     { title: 'Problem A', body: '…', icon: AlertTriangle, tag: 'Loss €50/mo', tone: 'destructive' },
 *     { title: 'Problem B', body: '…', icon: DollarSign, tone: 'primary' },
 *     { title: 'Problem C', body: '…', icon: Lock, tone: 'destructive' },
 *   ]}
 *   ctaPrimary={{ label: 'Start Earning', href: '/signup' }}
 *   ctaSecondary={{ label: 'Learn More', href: '#details' }}
 *   ctaCopy="Turn this problem into opportunity."
 * />
 * ```
 */
export type ProblemTemplateProps = {
  /** Array of problem items to display */
  items: ProblemItem[]
  /** Custom class name for the root element */
  className?: string
  /** Custom class name for the grid */
  gridClassName?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function ProblemTemplate({ items, className, gridClassName }: ProblemTemplateProps) {
  return (
    <div className={className}>
      {/* Grid of Problem Cards */}
      <div className={cn('grid gap-6 sm:gap-7 md:grid-cols-3', gridClassName)}>
        {items.map((item, idx) => (
          <FeatureInfoCard
            key={item.title}
            icon={item.icon}
            title={item.title}
            body={item.body}
            tag={item.tag}
            tone={item.tone || 'destructive'}
            size="base"
            delay={['delay-75', 'delay-150', 'delay-200'][idx]}
            className="min-h-[220px]"
          />
        ))}
      </div>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Export Types for Reuse
// ──────────────────────────────────────────────────────────────────────────────

export type { ProblemTemplateProps as ProvidersProblemProps }
