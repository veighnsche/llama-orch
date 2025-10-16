'use client'

import { Button } from '@rbee/ui/atoms/Button'
import { FeatureInfoCard } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import Link from 'next/link'
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
export type ProblemSectionProps = {
  /** Array of problem items to display */
  items: ProblemItem[]
  /** Primary CTA button configuration */
  ctaPrimary?: { label: string; href: string }
  /** Secondary CTA button configuration */
  ctaSecondary?: { label: string; href: string }
  /** Copy text above the CTA buttons */
  ctaCopy?: string
  /** Custom class name for the root element */
  className?: string
  /** Custom class name for the grid */
  gridClassName?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function ProblemSection({
  items,
  ctaPrimary,
  ctaSecondary,
  ctaCopy,
  className,
  gridClassName,
}: ProblemSectionProps) {
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

      {/* CTA Banner */}
      {(ctaCopy || ctaPrimary || ctaSecondary) && (
        <div className="mt-10 rounded-2xl border border-border bg-card/60 p-6 text-center sm:mt-12 sm:p-7 animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none delay-300">
          {ctaCopy && <p className="text-balance text-lg font-medium text-foreground">{ctaCopy}</p>}
          <div className="mt-4 flex flex-col items-center gap-3 sm:mt-5 sm:flex-row sm:justify-center">
            {ctaPrimary && (
              <Button asChild size="lg" className="animate-in fade-in motion-reduce:animate-none delay-150">
                <Link href={ctaPrimary.href}>{ctaPrimary.label}</Link>
              </Button>
            )}
            {ctaSecondary && (
              <Button
                asChild
                variant="outline"
                size="lg"
                className="animate-in fade-in motion-reduce:animate-none delay-150"
              >
                <Link href={ctaSecondary.href}>{ctaSecondary.label}</Link>
              </Button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Export Types for Reuse
// ──────────────────────────────────────────────────────────────────────────────

export type { ProblemSectionProps as ProvidersProblemProps }
