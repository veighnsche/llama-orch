import { CheckItem } from '@rbee/ui/atoms/CheckItem'
import { IconPlate } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { LucideIcon } from 'lucide-react'
import Link from 'next/link'

export interface SecurityCrateProps {
  /** Lucide icon component (e.g., Lock, Shield, Eye) */
  icon: LucideIcon
  /** Crate title (e.g., "auth-min: Zero-Trust Authentication") */
  title: string
  /** Optional subtitle (e.g., "The Trickster Guardians") */
  subtitle?: string
  /** Introduction paragraph */
  intro: string
  /** List of security features/capabilities */
  bullets: string[]
  /** Optional documentation link */
  docsHref?: string
  /** Visual tone (default: primary) */
  tone?: 'primary' | 'neutral'
  /** Additional CSS classes */
  className?: string
}

/**
 * SecurityCrate molecule for displaying security crate capabilities
 * with consistent structure, accessibility, and optional docs link
 */
export function SecurityCrate({
  icon,
  title,
  subtitle,
  intro,
  bullets,
  docsHref,
  tone = 'primary',
  className,
}: SecurityCrateProps) {
  const titleId = `security-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`

  return (
    <div
      className={cn(
        'flex h-full flex-col rounded-2xl border bg-card/60 p-6 md:p-8',
        'transition-shadow hover:shadow-lg',
        className,
      )}
      aria-labelledby={titleId}
    >
      {/* Header */}
      <div className="mb-4 flex items-center gap-3">
        <IconPlate icon={icon} size="lg" tone={tone === 'neutral' ? 'muted' : 'primary'} className="shrink-0" />
        <div className="flex-1">
          <h3 id={titleId} className="text-xl font-bold text-foreground">
            {title}
          </h3>
          {subtitle && <p className="text-sm text-muted-foreground">{subtitle}</p>}
        </div>
      </div>

      {/* Intro */}
      <p className="mb-4 text-sm leading-relaxed text-foreground/85">{intro}</p>

      {/* Bullets */}
      <ul className="mt-2 space-y-2" role="list">
        {bullets.map((bullet, idx) => (
          <CheckItem key={idx}>{bullet}</CheckItem>
        ))}
      </ul>

      {/* Footer with optional docs link */}
      {docsHref && (
        <div className="mt-auto pt-4">
          <Link
            href={docsHref}
            className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
            aria-label={`View documentation for ${title}`}
          >
            Docs →
          </Link>
        </div>
      )}
    </div>
  )
}
