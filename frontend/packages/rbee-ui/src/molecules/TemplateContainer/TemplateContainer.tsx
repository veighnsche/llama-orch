import { AuditReadinessCTA, type AuditReadinessCTAProps } from '@rbee/ui/molecules/AuditReadinessCTA'
import { CTARail } from '@rbee/ui/molecules/CTARail'
import { Disclaimer } from '@rbee/ui/molecules/Disclaimer'
import { FooterCTA } from '@rbee/ui/molecules/FooterCTA'
import { type HelperLink, HelperLinks } from '@rbee/ui/molecules/HelperLinks'
import { SecurityGuarantees } from '@rbee/ui/molecules/SecurityGuarantees'
import {
  CTABanner,
  type CTABannerProps,
  RibbonBanner,
  SectionCTAs,
  type SectionCTAsProps,
  TemplateBackground,
  type TemplateBackgroundProps,
} from '@rbee/ui/organisms'
import { cn, parseInlineMarkdown } from '@rbee/ui/utils'
import type { ReactNode } from 'react'

export interface TemplateContainerProps {
  /** Section title (null to skip rendering) */
  title: string | ReactNode | null
  /** Optional description */
  description?: string | ReactNode
  /** Small badge/label above title */
  eyebrow?: string | ReactNode
  /** Short lead-in sentence between eyebrow and title */
  kicker?: string | ReactNode
  /** Kicker color variant */
  kickerVariant?: 'default' | 'destructive'
  /** Right-aligned controls near the title (e.g., buttons) */
  actions?: ReactNode
  /** Background configuration */
  background?: {
    /** Background variant */
    variant?: TemplateBackgroundProps['variant']
    /** Optional decoration element */
    decoration?: ReactNode
    /** Overlay opacity (0-100) */
    overlayOpacity?: number
    /** Overlay color */
    overlayColor?: 'black' | 'white' | 'primary' | 'secondary'
    /** Enable blur effect */
    blur?: boolean
    /** Pattern size for pattern variants */
    patternSize?: 'small' | 'medium' | 'large'
    /** Pattern opacity (0-100) */
    patternOpacity?: number
  }
  /** Content alignment */
  align?: 'start' | 'center'
  /** Header layout: stack or split (two-column on md+) */
  layout?: 'stack' | 'split'
  /** Allow full-width background while constraining inner content */
  bleed?: boolean
  /** Vertical padding size */
  paddingY?: 'lg' | 'xl' | '2xl'
  /** Maximum width of content */
  maxWidth?: 'xl' | '2xl' | '3xl' | '4xl' | '5xl' | '6xl' | '7xl'
  /** Section content */
  children: ReactNode
  /** Additional CSS classes */
  className?: string
  /** Optional ID for the heading (for aria-labelledby) */
  headingId?: string
  /** Semantic heading level (1, 2, or 3) */
  headlineLevel?: 1 | 2 | 3
  /** Show a subtle separator under the header block */
  divider?: boolean
  /** Optional bottom CTAs */
  ctas?: SectionCTAsProps
  /** Optional disclaimer text */
  disclaimer?: {
    /** Disclaimer content */
    text: string | ReactNode
    /** Visual variant */
    variant?: 'default' | 'info' | 'warning' | 'muted'
    /** Show icon */
    showIcon?: boolean
  }
  /** Optional ribbon banner (e.g., insurance coverage) */
  ribbon?: {
    text: string
  }
  /** Optional CTA banner (appears after children, before bottom CTAs) */
  ctaBanner?: CTABannerProps
  /** Optional CTA rail (appears after children, uses CTARail molecule) */
  ctaRail?: {
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
  }
  /** Optional security guarantees (appears after children, uses SecurityGuarantees molecule) */
  securityGuarantees?: {
    /** Main heading text */
    heading: string
    /** Stats to display */
    stats: Array<{
      value: string
      label: string
      ariaLabel?: string
    }>
    /** Footnote text */
    footnote: string
  }
  /** Optional footer CTA (appears after children, uses FooterCTA molecule) */
  footerCTA?: {
    /** Optional message text */
    message?: string
    /** CTA buttons */
    ctas?: Array<{
      label: string
      href: string
      variant?: 'default' | 'outline' | 'ghost' | 'secondary' | 'destructive' | 'link'
    }>
  }
  /** Optional helper links (appears after children, uses HelperLinks molecule) */
  helperLinks?: HelperLink[]
  /** Optional audit readiness CTA (appears after children, uses AuditReadinessCTA molecule) */
  auditReadinessCTA?: AuditReadinessCTAProps
}

// Legacy bgVariant mapping to TemplateBackground variants (unused but kept for reference)
// const _legacyBgVariantMap = {
//   background: 'background',
//   secondary: 'secondary',
//   card: 'card',
//   default: 'background',
//   muted: 'muted',
//   subtle: 'subtle-border',
//   'destructive-gradient': 'gradient-destructive',
// } as const

const padY = {
  lg: 'py-16',
  xl: 'py-20',
  '2xl': 'py-24',
} as const

const maxWidthClasses = {
  xl: 'max-w-xl',
  '2xl': 'max-w-2xl',
  '3xl': 'max-w-3xl',
  '4xl': 'max-w-4xl',
  '5xl': 'max-w-5xl',
  '6xl': 'max-w-6xl',
  '7xl': 'max-w-7xl',
} as const

/** Semantic heading component that renders h1, h2, or h3 with same visual classes */
function HTag({
  as = 2,
  ...props
}: { as?: 1 | 2 | 3 } & React.HTMLAttributes<HTMLHeadingElement> & {
    id?: string
  }) {
  const Tag = (as === 1 ? 'h1' : as === 2 ? 'h2' : 'h3') as 'h1' | 'h2' | 'h3'
  return <Tag {...props} />
}

/** Slugify a string for use as an ID */
function slugify(value?: string): string | undefined {
  if (!value) return undefined
  return value
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^\w-]/g, '')
    .slice(0, 64)
}

export function TemplateContainer({
  title,
  description,
  eyebrow,
  kicker,
  kickerVariant = 'default',
  actions,
  background,
  align = 'center',
  layout = 'stack',
  bleed = false,
  paddingY = '2xl',
  maxWidth = '4xl',
  children,
  className,
  headingId,
  headlineLevel = 2,
  divider = false,
  ctas,
  disclaimer,
  ribbon,
  ctaBanner,
  ctaRail,
  securityGuarantees,
  footerCTA,
  helperLinks,
  auditReadinessCTA,
}: TemplateContainerProps) {
  // Resolve alignment
  const resolvedAlign = align

  // Generate heading ID if not provided
  const generatedId = headingId ?? (typeof title === 'string' ? slugify(title) : undefined)

  // Resolve background configuration
  const resolvedBackground = {
    variant: background?.variant ?? 'background',
    decoration: background?.decoration,
    overlayOpacity: background?.overlayOpacity,
    overlayColor: background?.overlayColor,
    blur: background?.blur,
    patternSize: background?.patternSize,
    patternOpacity: background?.patternOpacity,
  }

  return (
    <TemplateBackground
      variant={resolvedBackground.variant}
      decoration={resolvedBackground.decoration}
      overlayOpacity={resolvedBackground.overlayOpacity}
      overlayColor={resolvedBackground.overlayColor}
      blur={resolvedBackground.blur}
      patternSize={resolvedBackground.patternSize}
      patternOpacity={resolvedBackground.patternOpacity}
      className={className}
    >
      <section
        className={cn('relative', padY[paddingY], bleed && 'px-0')}
        aria-labelledby={title ? generatedId : undefined}
      >
        <div className={cn('container mx-auto', bleed ? 'px-4' : 'px-4')}>
          {title && (
            <div
              className={cn(
                maxWidthClasses[maxWidth],
                'mx-auto mb-14 md:mb-16',
                resolvedAlign === 'center' ? 'text-center' : 'text-left md:text-left',
                layout === 'split' && actions ? 'md:grid md:grid-cols-12 md:items-end md:gap-6' : '',
              )}
            >
              <div className={cn(layout === 'split' && actions ? 'md:col-span-8 space-y-3' : 'space-y-3')}>
                {eyebrow && (
                  <div className="text-xs font-medium text-primary uppercase tracking-wide animate-fade-in">
                    {eyebrow}
                  </div>
                )}

                {kicker && (
                  <div
                    className={cn(
                      'text-sm font-medium animate-fade-in',
                      kickerVariant === 'destructive' ? 'text-destructive/80' : 'text-muted-foreground',
                    )}
                  >
                    {kicker}
                  </div>
                )}

                <HTag
                  as={headlineLevel}
                  id={generatedId}
                  className="text-4xl md:text-5xl font-semibold tracking-tight text-foreground mb-2 text-balance leading-tight animate-fade-in-up"
                >
                  {title}
                </HTag>

                {description && (
                  <p className="text-lg text-muted-foreground leading-relaxed">
                    {typeof description === 'string' ? parseInlineMarkdown(description) : description}
                  </p>
                )}

                {divider && <div className="h-px bg-border/60 mt-6" />}
              </div>

              {actions && (
                <div
                  className={cn(
                    'mt-6 md:mt-0',
                    layout === 'split'
                      ? 'md:col-span-4 md:flex md:justify-end'
                      : resolvedAlign === 'center'
                        ? 'flex justify-center'
                        : '',
                  )}
                >
                  {actions}
                </div>
              )}
            </div>
          )}

          <div className={cn(maxWidthClasses[maxWidth], 'mx-auto')}>{children}</div>

          {/* Helper Links */}
          {helperLinks && helperLinks.length > 0 && (
            <div className="mt-12">
              <HelperLinks links={helperLinks} />
            </div>
          )}

          {/* Security Guarantees */}
          {securityGuarantees && (
            <div
              className={cn(
                'mt-10 animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none delay-300',
                maxWidthClasses[maxWidth],
                'mx-auto',
              )}
            >
              <SecurityGuarantees
                heading={securityGuarantees.heading}
                stats={securityGuarantees.stats}
                footnote={securityGuarantees.footnote}
              />
            </div>
          )}

          {/* Footer CTA */}
          {footerCTA && (
            <div
              className={cn(
                'mt-10 animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none delay-300',
                maxWidthClasses[maxWidth],
                'mx-auto',
              )}
            >
              <FooterCTA message={footerCTA.message} ctas={footerCTA.ctas} />
            </div>
          )}

          {/* CTA Rail */}
          {ctaRail && (
            <div
              className={cn(
                'mt-10 animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none delay-300',
                maxWidthClasses[maxWidth],
                'mx-auto',
              )}
            >
              <CTARail
                heading={ctaRail.heading}
                description={ctaRail.description}
                buttons={ctaRail.buttons}
                links={ctaRail.links}
                footnote={ctaRail.footnote}
              />
            </div>
          )}

          {/* Audit Readiness CTA */}
          {auditReadinessCTA && (
            <div
              className={cn(
                'mt-10 animate-in fade-in slide-in-from-bottom-2 motion-reduce:animate-none delay-300',
                maxWidthClasses[maxWidth],
                'mx-auto',
              )}
            >
              <AuditReadinessCTA {...auditReadinessCTA} />
            </div>
          )}

          {/* CTA Banner */}
          {ctaBanner && (
            <div className={cn('mt-10 sm:mt-12', maxWidthClasses[maxWidth], 'mx-auto')}>
              <CTABanner {...ctaBanner} />
            </div>
          )}

          {/* Ribbon Banner */}
          {ribbon && (
            <div className={cn('mt-10', maxWidthClasses[maxWidth], 'mx-auto')}>
              <RibbonBanner text={ribbon.text} />
            </div>
          )}

          {/* Bottom CTAs */}
          {ctas && (
            <div className={cn('mt-12', maxWidthClasses[maxWidth], 'mx-auto')}>
              <SectionCTAs {...ctas} />
            </div>
          )}

          {/* Disclaimer */}
          {disclaimer && (
            <div className={cn('mt-12', maxWidthClasses[maxWidth], 'mx-auto')}>
              <Disclaimer variant={disclaimer.variant ?? 'muted'} showIcon={disclaimer.showIcon}>
                {disclaimer.text}
              </Disclaimer>
            </div>
          )}
        </div>
      </section>
    </TemplateBackground>
  )
}
