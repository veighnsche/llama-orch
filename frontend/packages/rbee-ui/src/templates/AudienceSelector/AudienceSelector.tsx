'use client'

import { AudienceCard, type AudienceCardColor } from '@rbee/ui/organisms/AudienceCard'
import type { ReactNode } from 'react'

export interface AudienceCardData {
  icon: ReactNode
  category: string
  title: string
  description: string
  features: string[]
  href: string
  ctaText: string
  color?: AudienceCardColor
  imageSlot?: ReactNode
  badgeSlot?: ReactNode
  decisionLabel?: string
  showGradient?: boolean
}

export interface AudienceSelectorProps {
  cards: AudienceCardData[]
}

export function AudienceSelector({ cards }: AudienceSelectorProps) {
  return (
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
  )
}
