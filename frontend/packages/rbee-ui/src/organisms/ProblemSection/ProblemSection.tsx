'use client'

import { Button } from '@rbee/ui/atoms/Button'
import { FeatureInfoCard, SectionContainer } from '@rbee/ui/molecules'
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
 * ProblemSection displays a grid of problem cards with improved visual hierarchy.
 * Based on the ProvidersProblem design with CTA banner, loss tags, and staggered animations.
 *
 * @example
 * ```tsx
 * <ProblemSection
 *   kicker="Why this matters"
 *   title="The hidden risk of AI-assisted development"
 *   subtitle="You're building complex codebases with AI assistance..."
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
  kicker?: string
  title: string
  subtitle?: string
  items: ProblemItem[]
  ctaPrimary?: { label: string; href: string }
  ctaSecondary?: { label: string; href: string }
  ctaCopy?: string
  id?: string
  className?: string
  gridClassName?: string
  // Legacy support
  eyebrow?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function ProblemSection({
  kicker,
  eyebrow, // Legacy support - maps to kicker
  title,
  subtitle,
  items,
  ctaPrimary,
  ctaSecondary,
  ctaCopy,
  id,
  className,
  gridClassName,
}: ProblemSectionProps) {
  // Use eyebrow as fallback for kicker (backward compatibility)
  const displayKicker = kicker || eyebrow

  return (
    <SectionContainer
      title={title}
      description={subtitle}
      kicker={displayKicker}
      kickerVariant="destructive"
      bgVariant="destructive-gradient"
      paddingY="xl"
      maxWidth="7xl"
      align="center"
      headingId={id}
      className={className}
    >
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
    </SectionContainer>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Export Types for Reuse
// ──────────────────────────────────────────────────────────────────────────────

export type { ProblemSectionProps as ProvidersProblemProps }
