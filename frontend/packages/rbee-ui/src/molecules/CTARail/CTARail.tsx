import { Button } from '@rbee/ui/atoms/Button'
import { Card, CardContent } from '@rbee/ui/atoms/Card'
import { parseInlineMarkdown } from '@rbee/ui/utils'
import Link from 'next/link'
import React from 'react'

export interface CTARailProps {
  /** Main heading text */
  heading: string
  /** Optional description text */
  description?: string
  /** CTA buttons */
  buttons: Array<{
    text: string
    href: string
    variant?: 'default' | 'outline'
    ariaLabel?: string
  }>
  /** Optional footer links with bullet separators */
  links?: Array<{
    text: string
    href: string
  }>
  /** Optional footnote text */
  footnote?: string
  /** Additional CSS classes */
  className?: string
}

/**
 * CTARail molecule - a centered CTA card with heading, buttons, and optional links
 * Uses Card atom for consistent structure
 *
 * @example
 * <CTARail
 *   heading="Ready to get started?"
 *   buttons={[
 *     { text: 'Get Started', href: '/signup' },
 *     { text: 'Learn More', href: '/docs', variant: 'outline' }
 *   ]}
 *   links={[
 *     { text: 'Documentation', href: '/docs' },
 *     { text: 'API Reference', href: '/api' }
 *   ]}
 * />
 */
export function CTARail({ heading, description, buttons, links, footnote }: CTARailProps) {
  return (
    <Card className="border-primary/20 bg-primary/5">
      <CardContent className="p-6 text-center">
        <p className="mb-6 text-lg font-semibold text-foreground">{heading}</p>
        {description && <p className="mb-6 text-foreground/85">{parseInlineMarkdown(description)}</p>}

        <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
          {buttons.map((button, idx) => (
            <Button
              key={idx}
              size="lg"
              variant={button.variant}
              asChild
              className="transition-transform active:scale-[0.98]"
            >
              <Link href={button.href} aria-label={button.ariaLabel}>
                {button.text}
              </Link>
            </Button>
          ))}
        </div>

        {links && links.length > 0 && (
          <div className="mt-4 flex flex-wrap justify-center gap-3 text-sm text-muted-foreground">
            {links.map((link, idx) => (
              <React.Fragment key={idx}>
                {idx > 0 && <span>•</span>}
                <Link href={link.href} className="hover:text-primary hover:underline">
                  {link.text}
                </Link>
              </React.Fragment>
            ))}
          </div>
        )}

        {footnote && <p className="mt-6 text-xs text-muted-foreground">{footnote}</p>}
      </CardContent>
    </Card>
  )
}
