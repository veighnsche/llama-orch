import type { LucideIcon } from 'lucide-react'
import { cn } from '@/lib/utils'
import { IconBox } from '@/components/molecules'
import { Badge } from '@/components/atoms/Badge/Badge'

export interface UseCaseCardProps {
  icon: LucideIcon
  color: 'primary' | 'chart-2' | 'chart-3' | 'chart-4'
  title: string
  scenario: string
  solution: string
  highlights: string[]
  anchor?: string
  badge?: string
  className?: string
  style?: React.CSSProperties
}

export function UseCaseCard({
  icon,
  color,
  title,
  scenario,
  solution,
  highlights,
  anchor,
  badge,
  className,
  style,
}: UseCaseCardProps) {
  return (
    <article
      id={anchor}
      role="article"
      aria-labelledby={anchor ? `${anchor}-title` : undefined}
      tabIndex={-1}
      className={cn(
        'bg-card border border-border/80 rounded-xl p-6 md:p-8 shadow-sm hover:shadow-md transition-shadow scroll-mt-28',
        'hover:outline hover:outline-1 hover:outline-muted/40',
        'animate-in fade-in-50 slide-in-from-bottom-4',
        className
      )}
      style={style}
    >
      {/* Top row: Icon + Badge */}
      <div className="flex items-start justify-between mb-4">
        <IconBox icon={icon} color={color} size="lg" />
        {badge && (
          <Badge variant="secondary" className="text-xs">
            {badge}
          </Badge>
        )}
      </div>

      {/* Title */}
      <h3 id={anchor ? `${anchor}-title` : undefined} className="text-2xl font-semibold tracking-tight text-foreground mb-3">
        {title}
      </h3>

      {/* Scenario & Solution as definition list */}
      <dl className="space-y-3 text-sm leading-relaxed">
        <div>
          <dt className="sr-only">Scenario</dt>
          <dd className="text-muted-foreground">
            <span className="text-foreground font-medium">Scenario:</span> {scenario}
          </dd>
        </div>

        <div>
          <dt className="sr-only">Solution</dt>
          <dd className="text-muted-foreground">
            <span className="text-foreground font-medium">Solution:</span> {solution}
          </dd>
        </div>
      </dl>

      {/* Highlights */}
      <ul className="mt-4 space-y-1 text-sm font-medium text-chart-3">
        {highlights.map((highlight, index) => (
          <li key={index} className="pl-5 relative before:content-['✓'] before:absolute before:left-0 before:text-chart-3">
            {highlight}
          </li>
        ))}
      </ul>
    </article>
  )
}
