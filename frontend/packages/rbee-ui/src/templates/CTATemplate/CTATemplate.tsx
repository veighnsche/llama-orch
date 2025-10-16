import { Button } from '@rbee/ui/atoms/Button'
import { cn } from '@rbee/ui/utils'
import Link from 'next/link'
import type React from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type CTAAction = {
  label: string
  href: string
  iconLeft?: React.ComponentType<{ className?: string }>
  iconRight?: React.ComponentType<{ className?: string }>
  variant?: 'default' | 'outline' | 'ghost' | 'secondary' | 'link' | 'destructive'
}

/**
 * CTATemplate displays a call-to-action section with title, subtitle, and action buttons.
 *
 * @example
 * ```tsx
 * <CTATemplate
 *   eyebrow="Limited Time"
 *   title="Start building today"
 *   subtitle="Join 500+ developers"
 *   primary={{ label: 'Get Started', href: '/start' }}
 *   secondary={{ label: 'Learn More', href: '/docs' }}
 *   note="No credit card required"
 *   emphasis="gradient"
 * />
 * ```
 */
export type CTATemplateProps = {
  /** Optional eyebrow text above title */
  eyebrow?: string
  /** Main heading text */
  title: string
  /** Optional subtitle below title */
  subtitle?: string
  /** Primary action button */
  primary: CTAAction
  /** Optional secondary action button */
  secondary?: CTAAction
  /** Optional trust note below buttons */
  note?: string
  /** Text alignment */
  align?: 'center' | 'left'
  /** Background emphasis style */
  emphasis?: 'none' | 'gradient'
  /** Section ID for anchor links */
  id?: string
  /** Custom class name for the root element */
  className?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function CTATemplate({
  eyebrow,
  title,
  subtitle,
  primary,
  secondary,
  note,
  align = 'center',
  emphasis = 'none',
  id,
  className,
}: CTATemplateProps) {
  const alignClasses = align === 'center' ? 'text-center' : 'text-left'
  const justifyClasses = align === 'center' ? 'justify-center' : 'justify-start'

  const renderAction = (action: CTAAction, isPrimary: boolean) => {
    const IconLeft = action.iconLeft
    const IconRight = action.iconRight
    const variant = action.variant || (isPrimary ? undefined : 'outline')

    return (
      <Button
        key={action.label}
        asChild
        size="lg"
        variant={variant}
        className={cn(
          isPrimary
            ? 'group bg-primary text-primary-foreground hover:bg-primary/90'
            : 'bg-transparent border-border text-foreground hover:bg-secondary',
        )}
      >
        <Link href={action.href}>
          {IconLeft && <IconLeft className="mr-2 h-4 w-4" aria-hidden="true" />}
          {action.label}
          {IconRight && (
            <IconRight
              className={cn('ml-2 h-4 w-4', isPrimary && 'transition-transform group-hover:translate-x-1')}
              aria-hidden="true"
            />
          )}
        </Link>
      </Button>
    )
  }

  return (
    <section
      id={id}
      className={cn(
        'border-b border-border bg-background py-24',
        emphasis === 'gradient' &&
          'relative isolate before:absolute before:inset-0 before:bg-[radial-gradient(70%_50%_at_50%_0%,theme(colors.primary/10),transparent_60%)] before:pointer-events-none',
        className,
      )}
    >
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className={cn('mx-auto max-w-3xl', alignClasses)}>
          {/* Eyebrow */}
          {eyebrow && (
            <div
              className={cn(
                'mb-3 inline-flex items-center rounded-full border px-3 py-1 text-xs font-medium text-muted-foreground',
                'motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-300',
              )}
            >
              {eyebrow}
            </div>
          )}

          {/* Title */}
          <h2
            className={cn(
              'text-4xl font-bold tracking-tight text-foreground sm:text-5xl',
              'motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-bottom-2 motion-safe:duration-400 motion-safe:delay-100',
            )}
          >
            {title}
          </h2>

          {/* Subtitle */}
          {subtitle && (
            <p
              className={cn(
                'mt-3 text-lg text-muted-foreground',
                'motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-400 motion-safe:delay-150',
              )}
            >
              {subtitle}
            </p>
          )}

          {/* Actions */}
          <div
            className={cn(
              'mt-8 flex flex-col items-center gap-3 sm:flex-row',
              justifyClasses,
              'motion-safe:animate-in motion-safe:zoom-in-50 motion-safe:duration-300 motion-safe:delay-200',
            )}
          >
            {renderAction(primary, true)}
            {secondary && renderAction(secondary, false)}
          </div>

          {/* Trust note */}
          {note && (
            <p
              className={cn(
                'mt-6 text-sm text-muted-foreground font-sans',
                'motion-safe:animate-in motion-safe:fade-in-50 motion-safe:duration-300 motion-safe:delay-300',
              )}
            >
              {note}
            </p>
          )}
        </div>
      </div>
    </section>
  )
}
