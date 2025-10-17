import { Badge, Card, CardContent } from '@rbee/ui/atoms'
import { IconCardHeader, ListCard } from '@rbee/ui/molecules'
import Link from 'next/link'
import type { ReactNode } from 'react'

export interface IndustryCaseCardProps {
  /** Lucide icon component (e.g., Building2, Heart, Scale, Shield) */
  icon: ReactNode
  /** Industry name (e.g., "Financial Services") */
  industry: string
  /** Segment description (e.g., "Banks, Insurance, FinTech") */
  segments: string
  /** Brief summary of the use case */
  summary: string
  /** List of challenges */
  challenges: string[]
  /** List of solutions */
  solutions: string[]
  /** Optional compliance badges */
  badges?: string[]
  /** Optional link to industry page */
  href?: string
  /** Additional CSS classes */
  className?: string
}

/**
 * IndustryCaseCard organism for regulated industry use cases
 * with Challenge → Solution contrast and compliance badges
 *
 * Uses Card atom, IconCardHeader, Badge, and ListCard molecules
 */
export function IndustryCaseCard({
  icon,
  industry,
  segments,
  summary,
  challenges,
  solutions,
  badges,
  href,
  className,
}: IndustryCaseCardProps) {
  const industryId = `industry-${industry.toLowerCase().replace(/\s+/g, '-')}`

  return (
    <Card className="p-8">
      <IconCardHeader
        icon={icon}
        title={industry}
        subtitle={segments}
        titleId={industryId}
        iconSize="lg"
        iconTone="primary"
        titleClassName="text-xl"
      />

      <CardContent className="space-y-4 p-0">
        {/* Compliance badges */}
        {badges && badges.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {badges.map((badge) => (
              <Badge key={badge} variant="outline">
                {badge}
              </Badge>
            ))}
          </div>
        )}

        {/* Summary */}
        <p className="text-sm text-muted-foreground">{summary}</p>

        {/* Challenge panel */}
        <ListCard title="Challenge" items={challenges} variant="dot" color="muted" cardClassName="border-border" />

        {/* Solution panel */}
        <ListCard
          title="Solution with rbee"
          items={solutions}
          variant="check"
          color="chart-3"
          cardClassName="border-chart-3/50 bg-chart-3/10"
          titleClassName="text-chart-3"
        />

        {/* Optional footer link */}
        {href && (
          <div className="pt-2">
            <Link
              href={href}
              className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
              aria-label={`Learn more about ${industry}`}
            >
              Learn more →
            </Link>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
