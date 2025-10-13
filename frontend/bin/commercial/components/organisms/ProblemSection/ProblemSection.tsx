'use client'

import * as React from 'react'
import Link from 'next/link'
import { AlertTriangle, DollarSign, Lock } from 'lucide-react'
import { Button } from '@/components/atoms/Button/Button'
import { cn } from '@/lib/utils'

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
  title?: string
  subtitle?: string
  items?: ProblemItem[]
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
// Tone Mapping
// ──────────────────────────────────────────────────────────────────────────────

const toneMap = {
  primary: {
    border: 'border-primary/40',
    bg: 'from-primary/15 to-background',
    iconBg: 'bg-primary/10',
    iconText: 'text-primary',
    tagBg: 'bg-primary/10',
    tagText: 'text-primary',
  },
  destructive: {
    border: 'border-destructive/40',
    bg: 'from-destructive/15 to-background',
    iconBg: 'bg-destructive/10',
    iconText: 'text-destructive',
    tagBg: 'bg-destructive/10',
    tagText: 'text-destructive',
  },
  muted: {
    border: 'border-border',
    bg: 'from-muted/50 to-background',
    iconBg: 'bg-muted',
    iconText: 'text-muted-foreground',
    tagBg: 'bg-muted',
    tagText: 'text-muted-foreground',
  },
}

// ──────────────────────────────────────────────────────────────────────────────
// ProblemCard Molecule
// ──────────────────────────────────────────────────────────────────────────────

function ProblemCard({
  icon,
  title,
  body,
  tag,
  tone = 'destructive',
  delay,
}: ProblemItem & { delay?: string }) {
  const styles = toneMap[tone]
  
  // Handle both icon types (Component or ReactNode)
  let iconElement: React.ReactNode
  
  if (typeof icon === 'function') {
    iconElement = React.createElement(icon as React.ComponentType<{ className?: string }>, { 
      className: `h-6 w-6 ${styles.iconText}` 
    })
  } else if (React.isValidElement(icon)) {
    iconElement = React.cloneElement(icon, {
      // @ts-ignore - icon className merging
      className: cn(icon.props.className, styles.iconText)
    } as any)
  } else {
    iconElement = icon
  }

  return (
    <div
      className={cn(
        'min-h-[220px] rounded-2xl border bg-gradient-to-b p-6 backdrop-blur supports-[backdrop-filter]:bg-background/60 sm:p-7',
        'animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none',
        styles.border,
        styles.bg,
        delay,
      )}
    >
      <div 
        className={cn('mb-4 flex h-11 w-11 items-center justify-center rounded-xl', styles.iconBg)} 
        aria-hidden="true"
      >
        {iconElement}
      </div>
      <h3 className="text-lg font-semibold text-foreground">{title}</h3>
      <p className="text-pretty leading-relaxed text-muted-foreground">{body}</p>
      {tag && (
        <span 
          className={cn(
            'mt-3 inline-flex rounded-full px-2.5 py-1 text-xs tabular-nums',
            styles.tagBg,
            styles.tagText
          )}
        >
          {tag}
        </span>
      )}
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function ProblemSection({
  kicker,
  eyebrow, // Legacy support - maps to kicker
  title = 'The hidden risk of AI-assisted development',
  subtitle = "You're building complex codebases with AI assistance. What happens when the provider changes the rules?",
  items = [
    {
      title: 'The model changes',
      body: 'Your assistant updates overnight. Code generation breaks; workflows stall; your team is blocked.',
      icon: <AlertTriangle className="h-6 w-6" />,
      tone: 'destructive' as const,
    },
    {
      title: 'The price increases',
      body: '$20/month becomes $200/month—multiplied by your team. Infrastructure costs spiral.',
      icon: <DollarSign className="h-6 w-6" />,
      tone: 'primary' as const,
    },
    {
      title: 'The provider shuts down',
      body: 'APIs get deprecated. Your AI-built code becomes unmaintainable overnight.',
      icon: <Lock className="h-6 w-6" />,
      tone: 'destructive' as const,
    },
  ],
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
    <section
      id={id}
      className={cn(
        'border-b border-border bg-gradient-to-b from-background via-destructive/8 to-background px-6 py-20 lg:py-28',
        className,
      )}
    >
      <div className="mx-auto max-w-7xl">
        {/* Headline Stack */}
        <div className="mb-12 text-center animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none duration-500">
          {displayKicker && (
            <p className="mb-2 text-sm font-medium text-destructive/80">{displayKicker}</p>
          )}
          <h2 className="text-balance text-4xl font-extrabold tracking-tight text-foreground lg:text-5xl">
            {title}
          </h2>
          {subtitle && (
            <p className="mx-auto mt-4 max-w-2xl text-pretty text-lg leading-snug text-muted-foreground lg:text-xl">
              {subtitle}
            </p>
          )}
        </div>

        {/* Grid of Problem Cards */}
        <div className={cn('grid gap-6 sm:gap-7 md:grid-cols-3', gridClassName)}>
          {items.map((item, idx) => (
            <ProblemCard
              key={item.title}
              {...item}
              delay={['delay-75', 'delay-150', 'delay-200'][idx]}
            />
          ))}
        </div>

        {/* CTA Banner */}
        {(ctaCopy || ctaPrimary || ctaSecondary) && (
          <div className="mt-10 rounded-2xl border border-border bg-card/60 p-6 text-center sm:mt-12 sm:p-7 animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none delay-300">
            {ctaCopy && (
              <p className="text-balance text-lg font-medium text-foreground">
                {ctaCopy}
              </p>
            )}
            <div className="mt-4 flex flex-col items-center gap-3 sm:mt-5 sm:flex-row sm:justify-center">
              {ctaPrimary && (
                <Button
                  asChild
                  size="lg"
                  className="animate-in fade-in motion-reduce:animate-none delay-150"
                >
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
    </section>
  )
}

// ──────────────────────────────────────────────────────────────────────────────
// Export Types for Reuse
// ──────────────────────────────────────────────────────────────────────────────

export type { ProblemSectionProps as ProvidersProblemProps }
