import type { ReactNode } from 'react'

// ────────────────────────────────────────────────────────────────────────────
// Badge Types
// ────────────────────────────────────────────────────────────────────────────

export type HeroBadgeVariant = 'pulse' | 'icon' | 'simple' | 'none'

export type HeroBadge =
  | { variant: 'pulse'; text: string }
  | { variant: 'icon'; text: string; icon: ReactNode }
  | { variant: 'simple'; text: string }
  | { variant: 'none' }

// ────────────────────────────────────────────────────────────────────────────
// Headline Types
// ────────────────────────────────────────────────────────────────────────────

export type HeadlineVariant = 'two-line-highlight' | 'inline-highlight' | 'custom' | 'simple'

export type HeroHeadline =
  | { variant: 'two-line-highlight'; prefix: string; highlight: string }
  | { variant: 'inline-highlight'; content: string; highlight: string }
  | { variant: 'custom'; content: ReactNode }
  | { variant: 'simple'; content: string }

// ────────────────────────────────────────────────────────────────────────────
// Proof Elements Types
// ────────────────────────────────────────────────────────────────────────────

export type ProofElementVariant = 'bullets' | 'stats-tiles' | 'stats-pills' | 'badges' | 'indicators' | 'none'

export type BulletItem = {
  title: string
  variant?: 'check' | 'dot' | 'arrow'
  color?: 'primary' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5'
}

export type StatItem = {
  icon?: ReactNode
  value: string
  label: string
  helpText?: string
}

export type BadgeItem = {
  text: string
}

export type IndicatorItem = {
  text: string
  hasDot?: boolean
}

export type AssuranceItem = {
  text: string
  icon: ReactNode
}

export type HeroProofElements =
  | { variant: 'bullets'; items: BulletItem[] }
  | { variant: 'stats-tiles'; items: StatItem[]; columns?: 2 | 3 }
  | { variant: 'stats-pills'; items: StatItem[]; columns?: 2 | 3 }
  | { variant: 'badges'; items: BadgeItem[] }
  | { variant: 'indicators'; items: IndicatorItem[] }
  | { variant: 'assurance'; items: AssuranceItem[]; columns?: 2 | 3 }
  | { variant: 'none' }

// ────────────────────────────────────────────────────────────────────────────
// CTA Types
// ────────────────────────────────────────────────────────────────────────────

export type CTAButton = {
  label: string
  href?: string
  onClick?: () => void
  showIcon?: boolean
  variant?: 'default' | 'outline' | 'secondary'
  dataUmamiEvent?: string
  ariaLabel?: string
}

export type HeroCTAs = {
  primary: CTAButton
  secondary: CTAButton
  tertiary?: {
    label: string
    href: string
    mobileOnly?: boolean
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Trust Elements Types
// ────────────────────────────────────────────────────────────────────────────

export type TrustElementVariant = 'badges' | 'chips' | 'text' | 'none'

export type TrustBadgeItem = {
  type: 'github' | 'api' | 'cost'
  label: string
  href?: string
}

export type ComplianceChipItem = {
  icon: ReactNode
  label: string
  ariaLabel: string
}

export type HeroTrustElements =
  | { variant: 'badges'; items: TrustBadgeItem[] }
  | { variant: 'chips'; items: ComplianceChipItem[] }
  | { variant: 'text'; text: string }
  | { variant: 'none' }

// ────────────────────────────────────────────────────────────────────────────
// Layout Types
// ────────────────────────────────────────────────────────────────────────────

export type HeroLayout = {
  leftCols?: 6 | 7
  rightCols?: 5 | 6
  gap?: 8 | 10 | 12
  verticalSpacing?: 6 | 8 | 10
}

// ────────────────────────────────────────────────────────────────────────────
// Background Types
// ────────────────────────────────────────────────────────────────────────────

export type BackgroundVariant = 'gradient' | 'radial' | 'honeycomb' | 'custom'

export type HeroBackground =
  | { variant: 'gradient' }
  | { variant: 'radial' }
  | { variant: 'honeycomb'; size?: 'small' | 'large'; fadeDirection?: 'radial' | 'bottom' }
  | { variant: 'custom'; className: string }

// ────────────────────────────────────────────────────────────────────────────
// Main Props
// ────────────────────────────────────────────────────────────────────────────

export interface HeroTemplateProps {
  /** Badge configuration */
  badge?: HeroBadge

  /** Headline configuration */
  headline: HeroHeadline

  /** Subcopy text */
  subcopy: string | ReactNode

  /** Maximum width for subcopy */
  subcopyMaxWidth?: 'narrow' | 'medium' | 'wide'

  /** Proof elements configuration */
  proofElements?: HeroProofElements

  /** CTA buttons configuration */
  ctas: HeroCTAs

  /** Trust elements configuration */
  trustElements?: HeroTrustElements

  /** Helper text (appears between CTAs and trust elements) */
  helperText?: string

  /** Right-side aside content (page-specific visual) */
  aside: ReactNode

  /** Aria label for the aside */
  asideAriaLabel?: string

  /** Layout configuration */
  layout?: HeroLayout

  /** Background configuration */
  background?: HeroBackground

  /** Padding variant */
  padding?: 'default' | 'compact' | 'spacious'

  /** Animation configuration */
  animations?: {
    enabled?: boolean
    stagger?: boolean
    direction?: 'bottom' | 'left' | 'right'
  }

  /** Section aria-labelledby ID */
  headingId?: string
}
