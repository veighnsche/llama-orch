import { UseCaseCard } from '@rbee/ui/molecules'
import { cn } from '@rbee/ui/utils'
import type { LucideIcon } from 'lucide-react'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type UseCase = {
  icon: LucideIcon
  title: string
  scenario: string
  solution: string
  outcome: string
  tags?: string[]
  cta?: { label: string; href: string }
  illustrationSrc?: string
}

/**
 * UseCasesTemplate displays a grid of use case cards.
 * 
 * @example
 * ```tsx
 * <UseCasesTemplate
 *   items={[
 *     { icon: Code, title: 'AI Coding', scenario: '...', solution: '...', outcome: '...' },
 *     { icon: Server, title: 'API Generation', scenario: '...', solution: '...', outcome: '...' },
 *   ]}
 *   columns={3}
 * />
 * ```
 */
export type UseCasesTemplateProps = {
  /** Array of use case items to display */
  items: UseCase[]
  /** Number of columns in the grid (2 or 3) */
  columns?: 2 | 3
  /** Custom class name for the root element */
  className?: string
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function UseCasesTemplate({ items, columns = 3, className }: UseCasesTemplateProps) {
  const gridCols = columns === 2 ? 'sm:grid-cols-2' : 'sm:grid-cols-2 lg:grid-cols-3'

  return (
    <div className={className}>
      {/* Cards grid */}
      <div className={cn('mx-auto grid max-w-6xl gap-6 animate-in fade-in-50 duration-400', gridCols)}>
        {items.map((item, i) => (
          <UseCaseCard
            key={i}
            icon={item.icon}
            title={item.title}
            scenario={item.scenario}
            solution={item.solution}
            outcome={item.outcome}
            tags={item.tags}
            cta={item.cta}
            iconSize="md"
            iconTone="primary"
            style={{ animationDelay: `${i * 80}ms` }}
            className="animate-in fade-in slide-in-from-bottom-2 duration-400"
          />
        ))}
      </div>
    </div>
  )
}
