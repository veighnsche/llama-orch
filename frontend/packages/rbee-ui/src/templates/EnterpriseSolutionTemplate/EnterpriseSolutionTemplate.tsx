import { SolutionSection } from '@rbee/ui/organisms'
import type * as React from 'react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type SolutionFeature = {
  icon: React.ReactNode
  title: string
  body: string
  badge?: string | React.ReactNode
}

export type SolutionStep = {
  title: string
  body: string
}

export type EarningRow = {
  model: string
  meta: string
  value: string
  note?: string
}

export type EnterpriseSolutionTemplateProps = {
  id?: string
  kicker?: string
  eyebrowIcon?: React.ReactNode
  title: string
  subtitle?: string
  features: SolutionFeature[]
  steps: SolutionStep[]
  earnings?: {
    title?: string
    rows: EarningRow[]
    disclaimer?: string
  }
  illustration?: React.ReactNode
  ctaPrimary?: { label: string; href: string }
  ctaSecondary?: { label: string; href: string }
  ctaCaption?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseSolutionTemplate(props: EnterpriseSolutionTemplateProps) {
  return <SolutionSection {...props} />
}
