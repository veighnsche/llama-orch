'use client'

import { Badge } from '@rbee/ui/atoms/Badge'
import { ComplianceShield, DevGrid, GpuMarket } from '@rbee/ui/icons'
import { AudienceCard } from '@rbee/ui/organisms'
import { ChevronRight, Code2, Server, Shield } from 'lucide-react'
import Link from 'next/link'
import type { ReactNode } from 'react'

export interface AudienceCardData {
  icon: ReactNode
  category: string
  title: string
  description: string
  features: string[]
  href: string
  ctaText: string
  color: 'chart-2' | 'chart-3' | 'primary'
  imageSlot: React.ReactNode
  badgeSlot?: React.ReactNode
  decisionLabel: string
}

export interface HelperLink {
  label: string
  href: string
}

export interface AudienceSelectorProps {
  cards: AudienceCardData[]
  helperLinks?: HelperLink[]
  showGradient?: boolean
}

export function AudienceSelector({ cards, helperLinks, showGradient = true }: AudienceSelectorProps) {
  return (
    <>
      {/* Radial gradient backplate */}
      {showGradient && (
        <div
          className="pointer-events-none absolute inset-x-0 top-0 h-[600px] opacity-40"
          style={{
            background: 'radial-gradient(ellipse 80% 50% at 50% 0%, hsl(var(--primary) / 0.05), transparent)',
          }}
          aria-hidden="true"
        />
      )}

      {/* Grid with responsive 2→3 column layout and equal heights */}
      <div
        className="mx-auto grid max-w-6xl grid-cols-1 content-start gap-6 sm:grid-cols-2 xl:grid-cols-3 xl:gap-8"
        aria-label="Audience options: Developers, GPU Owners, Enterprise"
      >
        {cards.map((card, index) => (
          <div key={index} className="flex h-full">
            <AudienceCard
              icon={card.icon}
              category={card.category}
              title={card.title}
              description={card.description}
              features={card.features}
              href={card.href}
              ctaText={card.ctaText}
              color={card.color}
              imageSlot={card.imageSlot}
              badgeSlot={card.badgeSlot}
              decisionLabel={card.decisionLabel}
            />
          </div>
        ))}
      </div>

      {/* Bottom helper links */}
      {helperLinks && helperLinks.length > 0 && (
        <div className="mx-auto mt-12 text-center">
          {helperLinks.map((link, index) => (
            <span key={index}>
              {index > 0 && <span className="mx-3 text-muted-foreground/50">·</span>}
              <Link
                href={link.href}
                className="inline-flex items-center gap-2 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary/40"
              >
                {link.label}
                <ChevronRight className="h-3.5 w-3.5" />
              </Link>
            </span>
          ))}
        </div>
      )}
    </>
  )
}
