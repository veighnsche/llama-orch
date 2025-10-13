import { Button } from '@/components/atoms/Button/Button'
import { cn } from '@/lib/utils'
import Link from 'next/link'
import React from 'react'

export type CTAAction = {
  label: string
  href: string
  iconLeft?: React.ComponentType<{ className?: string }>
  iconRight?: React.ComponentType<{ className?: string }>
  variant?: 'default' | 'outline' | 'ghost' | 'secondary' | 'link' | 'destructive'
}

export type CTASectionProps = {
  eyebrow?: string
  title: string
  subtitle?: string
  primary: CTAAction
  secondary?: CTAAction
  note?: string // small trust line under buttons
  align?: 'center' | 'left' // default 'center'
  emphasis?: 'none' | 'gradient' // toggles subtle bg flourish
  id?: string
  className?: string
}

export function CTASection({
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
}: CTASectionProps) {
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
                'mb-3 inline-flex items-center rounded-full border border-border px-3 py-1 text-xs font-medium text-muted-foreground',
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
                'mt-6 text-sm text-muted-foreground',
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

// Legacy wrapper for homepage (complex variant with image)
export function CTASectionLegacy() {
  // Keep old implementation if needed, or remove after migration
  return null
}
