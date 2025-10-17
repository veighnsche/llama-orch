import { Button } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'

export interface SectionCTAsProps {
  /** Optional label text above buttons */
  label?: string
  /** Primary CTA button */
  primary?: {
    label: string
    href: string
    ariaLabel?: string
  }
  /** Secondary CTA button */
  secondary?: {
    label: string
    href: string
    ariaLabel?: string
  }
  /** Optional caption text below buttons */
  caption?: string
  /** Additional CSS classes */
  className?: string
}

/**
 * SectionCTAs organism - Bottom section call-to-action buttons with optional label and caption
 * 
 * Centered layout with responsive button arrangement (stack on mobile, row on desktop).
 * Unlike CTABanner, this is NOT card-based - just buttons with optional text.
 * 
 * @example
 * <SectionCTAs
 *   label="Ready to get started?"
 *   primary={{ label: "Sign Up Free", href: "/signup" }}
 *   secondary={{ label: "View Pricing", href: "/pricing" }}
 *   caption="No credit card required"
 * />
 */
export function SectionCTAs({
  label,
  primary,
  secondary,
  caption,
  className,
}: SectionCTAsProps) {
  // Don't render if no content
  if (!label && !primary && !secondary && !caption) {
    return null
  }

  return (
    <div className={cn('text-center', className)}>
      {label && (
        <p className="mb-4 text-sm font-medium text-muted-foreground">
          {label}
        </p>
      )}

      {(primary || secondary) && (
        <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
          {primary && (
            <Button
              asChild
              size="lg"
              className="transition-transform active:scale-[0.98]"
            >
              <a
                href={primary.href}
                aria-label={primary.ariaLabel || primary.label}
              >
                {primary.label}
              </a>
            </Button>
          )}

          {secondary && (
            <Button
              asChild
              size="lg"
              variant="outline"
              className="transition-transform active:scale-[0.98]"
            >
              <a
                href={secondary.href}
                aria-label={secondary.ariaLabel || secondary.label}
              >
                {secondary.label}
              </a>
            </Button>
          )}
        </div>
      )}

      {caption && (
        <p
          className={cn(
            'text-sm text-muted-foreground font-sans text-center',
            (primary || secondary) && 'mt-4'
          )}
        >
          {caption}
        </p>
      )}
    </div>
  )
}
