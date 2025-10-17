import { Card, CardContent, QuoteBlock } from '@rbee/ui/atoms'
import { IconCardHeader } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type * as React from 'react'

export type ProvidersCaseCardProps = {
  icon: React.ReactNode
  title: string
  subtitle?: string
  quote: string
  facts: { label: string; value: string }[]
  highlight?: string
  index?: number
  className?: string
}

export function ProvidersCaseCard({
  icon,
  title,
  subtitle,
  quote,
  facts,
  highlight,
  index = 0,
  className,
}: ProvidersCaseCardProps) {
  const delays = ['delay-75', 'delay-150', 'delay-200', 'delay-300']
  const delay = delays[index % delays.length]

  return (
    <Card
      className={cn(
        'group min-h-[320px] bg-gradient-to-b from-card/70 to-background/60 p-6 backdrop-blur transition-transform hover:translate-y-0.5 supports-[backdrop-filter]:bg-background/60 sm:p-7',
        'animate-in fade-in slide-in-from-bottom-2',
        delay,
        className,
      )}
    >
      <IconCardHeader
        icon={icon}
        title={title}
        subtitle={subtitle}
        iconSize="lg"
        iconTone="primary"
        titleClassName="text-lg font-semibold"
        subtitleClassName="text-xs"
        align="start"
      />

      <CardContent className="p-0 space-y-4">
        {/* Optional highlight badge */}
        {highlight && (
          <div className="inline-flex items-center rounded-full bg-primary/10 px-2.5 py-1 text-[11px] text-primary">
            {highlight}
          </div>
        )}

        {/* Quote block */}
        <QuoteBlock>{quote}</QuoteBlock>

        {/* Facts list */}
        <div className="space-y-2 text-sm">
          {facts.map((fact, idx) => {
            const isEarnings = fact.label.toLowerCase().includes('monthly')
            return (
              <div key={idx} className="flex justify-between">
                <span className="text-muted-foreground">{fact.label}</span>
                <span className={cn('tabular-nums text-foreground', isEarnings && 'font-semibold text-primary')}>
                  {fact.value}
                </span>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}
