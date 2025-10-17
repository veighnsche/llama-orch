import { Card } from '@rbee/ui/atoms'
import { IconCardHeader } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type * as React from 'react'

export type ProvidersSecurityCardProps = {
  icon: React.ReactNode
  title: string
  subtitle?: string
  body: string
  points: string[]
  index?: number
  className?: string
}

export function ProvidersSecurityCard({
  icon,
  title,
  subtitle,
  body,
  points,
  index = 0,
  className,
}: ProvidersSecurityCardProps) {
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
        iconTone="success"
        titleClassName="text-lg font-semibold"
        subtitleClassName="text-xs"
        align="start"
      />

      {/* Body text */}
      <p className="mb-4 line-clamp-3 text-sm leading-relaxed text-muted-foreground">{body}</p>

      {/* Points list */}
      <ul className="space-y-2">
        {points.map((point, idx) => (
          <li key={idx} className="flex items-center gap-2 text-sm text-muted-foreground">
            <div className="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400" />
            {point}
          </li>
        ))}
      </ul>
    </Card>
  )
}
