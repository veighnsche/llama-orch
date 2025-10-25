import { Button, Card, CardContent } from '@rbee/ui/atoms'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface CTABannerProps {
  /** Copy text above buttons */
  copy?: string | ReactNode
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
  /** Additional CSS classes */
  className?: string
}

/**
 * CTABanner organism - Card-based call-to-action with optional copy and button(s)
 *
 * Follows standard card pattern:
 * - Card has padding (p-6 sm:p-8)
 * - CardContent has p-0
 * - Center-aligned content
 *
 * @example
 * <CTABanner
 *   copy="Ready to get started?"
 *   primary={{ label: "Sign Up", href: "/signup" }}
 *   secondary={{ label: "Learn More", href: "/docs" }}
 * />
 */
export function CTABanner({ copy, primary, secondary, className }: CTABannerProps) {
  // Don't render if no content
  if (!copy && !primary && !secondary) {
    return null
  }

  return (
    <Card
      className={cn(
        'rounded bg-card/60 p-6 text-center sm:p-8',
        'animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none delay-300',
        className,
      )}
    >
      <CardContent className="p-0 space-y-4 sm:space-y-5">
        {copy && <p className="text-balance text-lg font-medium text-foreground">{copy}</p>}

        {(primary || secondary) && (
          <div className="flex flex-col items-center gap-3 sm:flex-row sm:justify-center">
            {primary && (
              <Button asChild size="lg" className="animate-in fade-in motion-reduce:animate-none delay-150">
                <a href={primary.href} aria-label={primary.ariaLabel || primary.label}>
                  {primary.label}
                </a>
              </Button>
            )}

            {secondary && (
              <Button
                asChild
                variant="outline"
                size="lg"
                className="animate-in fade-in motion-reduce:animate-none delay-150"
              >
                <a href={secondary.href} aria-label={secondary.ariaLabel || secondary.label}>
                  {secondary.label}
                </a>
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
