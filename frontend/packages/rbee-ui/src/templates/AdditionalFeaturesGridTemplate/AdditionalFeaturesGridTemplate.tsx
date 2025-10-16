'use client'

import { Badge, Card, CardContent } from '@rbee/ui/atoms'
import { IconCardHeader } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'

export interface FeatureGridCard {
  href: string
  ariaLabel: string
  icon: ReactNode
  iconTone: 'chart-2' | 'chart-3' | 'primary' | 'muted'
  title: string
  subtitle: string
  borderColor: string
  featured?: boolean
}

export interface FeatureRow {
  categoryLabel: string
  cards: FeatureGridCard[]
}

export interface AdditionalFeaturesGridTemplateProps {
  rows: FeatureRow[]
  className?: string
}

export function AdditionalFeaturesGridTemplate({ rows, className }: AdditionalFeaturesGridTemplateProps) {
  return (
    <div className={cn('', className)}>
      <div className="mx-auto max-w-6xl space-y-8">
        {rows.map((row, rowIdx) => (
          <div key={rowIdx}>
            <div className="flex items-center gap-2 mb-4">
              <Badge variant="secondary">{row.categoryLabel}</Badge>
            </div>
            <div
              className={cn('grid gap-4', row.cards.length === 3 ? 'md:grid-cols-2 lg:grid-cols-3' : 'md:grid-cols-3')}
            >
              {row.cards.map((card, cardIdx) => (
                <a
                  key={cardIdx}
                  href={card.href}
                  className={cn(
                    'group block transition-transform hover:-translate-y-0.5 animate-in fade-in slide-in-from-bottom-2 focus-visible:outline-none',
                    card.featured && 'lg:col-span-1 md:col-span-2 lg:col-span-1',
                  )}
                  style={{ animationDelay: `${rowIdx * 100 + cardIdx * 100}ms` }}
                  aria-label={card.ariaLabel}
                >
                  <Card
                    className={cn(
                      'h-full relative overflow-hidden',
                      `before:absolute before:inset-x-0 before:top-0 before:rounded-t-xl ${card.borderColor}`,
                    )}
                  >
                    <CardContent className="p-6">
                      <IconCardHeader
                        icon={card.icon}
                        iconTone={card.iconTone}
                        iconSize="sm"
                        title={card.title}
                        subtitle={card.subtitle}
                        titleClassName="text-lg"
                        subtitleClassName="text-sm mt-2"
                        useCardHeader={false}
                      />
                      <div className="mt-4 text-sm text-primary inline-flex items-center gap-1 group-hover:gap-2 transition-all">
                        <span>Learn more</span>
                        <ChevronRight className="size-3" />
                      </div>
                    </CardContent>
                  </Card>
                </a>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
