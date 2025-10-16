import type { Sector } from '@rbee/ui/data/testimonials'
import { TestimonialsRail } from '@rbee/ui/organisms'

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

export type EnterpriseTestimonialsTemplateProps = {
  sectorFilter: Sector | Sector[]
  layout: 'grid' | 'carousel'
  showStats: boolean
}

// ──────────────────────────────────────────────────────────────────────────────
// Main Component
// ──────────────────────────────────────────────────────────────────────────────

export function EnterpriseTestimonialsTemplate({
  sectorFilter,
  layout,
  showStats,
}: EnterpriseTestimonialsTemplateProps) {
  return (
    <div>
      {/* Testimonials Rail */}
      <TestimonialsRail sectorFilter={sectorFilter} layout={layout} showStats={showStats} />
    </div>
  )
}
