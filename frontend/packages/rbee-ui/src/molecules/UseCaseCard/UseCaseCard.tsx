import { Badge } from '@rbee/ui/atoms/Badge'
import { Card, CardContent, CardFooter } from '@rbee/ui/atoms/Card'
import { IconCardHeader } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import Link from 'next/link'
import type * as React from 'react'

export interface UseCaseCardProps {
  icon: React.ReactNode
  color?: 'primary' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5' | 'muted' | 'success' | 'warning'
  iconTone?: 'primary' | 'muted' | 'success' | 'warning' | 'chart-1' | 'chart-2' | 'chart-3' | 'chart-4' | 'chart-5'
  iconSize?: 'sm' | 'md' | 'lg'
  title: string
  scenario: string
  solution: string
  outcome?: string
  tags?: string[]
  cta?: { label: string; href: string }
  anchor?: string
  badge?: string
  className?: string
  style?: React.CSSProperties
}

export function UseCaseCard({
  icon,
  color,
  iconTone = 'primary',
  iconSize = 'md',
  title,
  scenario,
  solution,
  outcome,
  tags,
  cta,
  anchor,
  badge,
  className,
  style,
}: UseCaseCardProps) {
  // Use color if provided, otherwise fall back to iconTone
  const resolvedIconTone = color || iconTone
  return (
    <Card id={anchor} tabIndex={0} className={cn('group transition-all', className)} style={style}>
      <div className="flex items-start justify-between pr-6">
        <IconCardHeader
          icon={icon}
          title={title}
          iconSize={iconSize}
          iconTone={resolvedIconTone}
          titleClassName="text-base font-semibold tracking-tight text-card-foreground"
          titleId={anchor ? `${anchor}-title` : undefined}
          className="flex-1 min-w-0"
        />
        {badge && (
          <Badge variant="secondary" className="ml-2 mt-4 text-xs shrink-0">
            {badge}
          </Badge>
        )}
      </div>

      <CardContent className="space-y-4">
        {/* Scenario */}
        <div className="min-h-24">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Scenario</div>
          <div className="mt-1 text-sm leading-relaxed text-muted-foreground">{scenario}</div>
        </div>

        <div className="h-px bg-border/60" />

        {/* Solution */}
        <div className="min-h-24">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Solution</div>
          <div className="mt-1 text-sm leading-relaxed text-muted-foreground">{solution}</div>
        </div>

        {outcome && (
          <>
            <div className="h-px bg-border/60" />

            {/* Outcome callout */}
            <div className="rounded border border-primary/30 bg-primary/10 p-3 min-h-28 items-center flex">
              <div>
                <div className="text-xs font-semibold uppercase tracking-wide text-primary font-sans">Outcome</div>
                <div className="mt-1 text-sm text-foreground">{outcome}</div>
              </div>
            </div>
          </>
        )}
      </CardContent>

      {/* Optional footer */}
      {(tags || cta) && (
        <CardFooter className="flex-wrap gap-2">
          {tags?.map((tag, idx) => (
            <Badge
              key={idx}
              variant="outline"
              className="rounded-full border px-2 py-0.5 text-[11px] text-muted-foreground"
            >
              {tag}
            </Badge>
          ))}
          {cta && (
            <Link href={cta.href} className="ml-auto text-sm font-medium text-primary hover:underline">
              {cta.label}
            </Link>
          )}
        </CardFooter>
      )}
    </Card>
  )
}
